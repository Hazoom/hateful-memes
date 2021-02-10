import logging
import ntpath
from typing import Iterable, Union, Tuple, Dict

import srsly

import torch
from torch import Tensor

from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import TextField, MetadataField, LabelField, ArrayField, Field
from allennlp.data.instance import Instance
from allennlp_models.vision.dataset_readers.vision_reader import VisionReader

logger = logging.getLogger(__name__)


@DatasetReader.register("memereader")
class HatefulMemeDatasetReader(VisionReader):
    def __init__(
        self,
        source_max_tokens: int,
        truncate_long_sequences: bool = None,
        uncased: bool = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._source_max_tokens = source_max_tokens

        self._source_max_truncated = 0
        self._source_max_skipped = 0

        self._num_images_skipped = 0

        self._truncate_long_sequences = True if truncate_long_sequences is None else truncate_long_sequences
        logger.info("truncate_long_sequences=%s", str(self._truncate_long_sequences))

        self._uncased = True if uncased is None else uncased
        logger.info("uncased=%s", str(self._uncased))

        self._use_cache = True

    def _validate_line(self, text: str, image_path: Union[str, None]) -> bool:
        if not text:
            return False

        # check length of source
        tokenized_source = self._tokenizer.tokenize(text)
        if (
            self._source_max_tokens
            and len(tokenized_source) > self._source_max_tokens
            and not self._truncate_long_sequences
        ):
            self._source_max_skipped += 1
            return False

        if image_path:
            try:
                self.image_loader([image_path])
            except RuntimeError:
                self._num_images_skipped += 1
                return False

        return True

    def _read(self, file_path: str) -> Iterable[Instance]:
        # Reset truncated/skipped counts
        self._source_max_truncated = 0
        self._source_max_skipped = 0
        self._num_images_skipped = 0

        lines = srsly.read_jsonl(file_path)
        lines = list(self.shard_iterable(lines))

        if self.produce_featurized_images:
            filenames = [ntpath.basename(info_dict["img"]) for info_dict in lines]
            image_paths = [self.images[filename] for filename in filenames]
        else:
            image_paths = [None] * len(lines)

        batch = []
        for index, (line, image_path) in enumerate(zip(lines, image_paths)):
            text = line["text"]
            line_is_valid = self._validate_line(text, image_path)
            if not line_is_valid:
                continue
            batch.append((image_path, text, line.get("label", None), line))
            if len(batch) == self.image_processing_batch_size or index == len(lines) - 1:
                # It would be much easier to just process one image at a time, but it's faster to process
                # them in batches. So this code gathers up instances until it has enough to fill up a batch
                # that needs processing, and then processes them all.
                batch_image_paths = [item[0] for item in batch]
                if batch_image_paths == [None] * len(batch_image_paths):
                    processed_images = batch_image_paths  # all nones
                else:
                    processed_images = self._process_image_paths(batch_image_paths, use_cache=self._use_cache)

                assert len(batch) == len(processed_images)

                for item, processed_image in zip(batch, processed_images):
                    yield self.text_to_instance(
                        image=processed_image,
                        text=item[1],
                        label=item[2],
                        metadata=item[3],
                    )

                # initialize batch items for next batch
                batch = []

        if self._source_max_tokens and (self._source_max_truncated or self._source_max_skipped):
            logger.info(
                "In %d instances, the source token length exceeded the max limit (%d) and were %s.",
                self._source_max_truncated if self._truncate_long_sequences else self._source_max_skipped,
                self._source_max_tokens,
                "truncated" if self._truncate_long_sequences else "skipped",
            )
        if self._num_images_skipped:
            logger.info(
                "In %d instances, the image was non RGB image.",
                self._num_images_skipped,
            )

    def _preprocess(self, query: str) -> str:
        return query.lower() if self._uncased else query

    def text_to_instance(
        self,
        image: Union[Tuple[Tensor, Tensor], None],
        text: str,
        label: int = None,
        metadata: dict = None,
        use_cache: bool = True,
    ) -> Instance:
        if not metadata:
            metadata = dict()

        source = self._preprocess(text)

        tokenized_source = self._tokenizer.tokenize(source)
        if self._source_max_tokens and len(tokenized_source) > self._source_max_tokens:
            self._source_max_truncated += 1
            if self._truncate_long_sequences:
                tokenized_source = tokenized_source[: self._source_max_tokens]
        source_field = TextField(tokenized_source, self._token_indexers)

        fields: Dict[str, Field] = {"source_tokens": source_field, "metadata": MetadataField(metadata)}

        if image is not None:
            features, coords = image

            fields["box_features"] = ArrayField(features)
            fields["box_coordinates"] = ArrayField(coords)
            fields["box_mask"] = ArrayField(
                features.new_ones((features.shape[0],), dtype=torch.bool),
                padding_value=False,
                dtype=torch.bool,
            )

        if label is not None:
            fields["label"] = LabelField(label, skip_indexing=True)

        return Instance(fields)
