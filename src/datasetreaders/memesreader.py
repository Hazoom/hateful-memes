import logging
from typing import Iterable, Dict

import srsly
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import TextField, MetadataField, LabelField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer
from allennlp.data.tokenizers import Tokenizer

logger = logging.getLogger(__name__)


@DatasetReader.register("memereader")
class HatefulMemeDatasetReader(DatasetReader):
    def __init__(
        self,
        source_tokenizer: Tokenizer,
        source_token_indexers: Dict[str, TokenIndexer],
        source_max_tokens: int,
        truncate_long_sequences: bool = None,
        uncased: bool = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._source_tokenizer = source_tokenizer
        self._source_token_indexers = source_token_indexers
        self._source_max_tokens = source_max_tokens

        self._source_max_truncated = 0
        self._source_max_skipped = 0

        self._truncate_long_sequences = True if truncate_long_sequences is None else truncate_long_sequences
        logger.info("truncate_long_sequences=%s", str(self._truncate_long_sequences))

        self._uncased = True if uncased is None else uncased
        logger.info("uncased=%s", str(self._uncased))

    def _validate_line(self, text: str) -> bool:
        if not text:
            return False

        # check length of source
        tokenized_source = self._source_tokenizer.tokenize(text)
        if (
            self._source_max_tokens
            and len(tokenized_source) > self._source_max_tokens
            and not self._truncate_long_sequences
        ):
            self._source_max_skipped += 1
            return False

        return True

    def _read(self, file_path: str) -> Iterable[Instance]:
        # Reset truncated/skipped counts
        self._source_max_truncated = 0
        self._source_max_skipped = 0

        lines = srsly.read_jsonl(file_path)
        for line in lines:
            text = line["text"]
            line_is_valid = self._validate_line(text)
            if not line_is_valid:
                continue
            yield self.text_to_instance(
                text,
                label=line.get("label", None),
                metadata=line,
            )

        if self._source_max_tokens and (self._source_max_truncated or self._source_max_skipped):
            logger.info(
                "In %d instances, the source token length exceeded the max limit (%d) and were %s.",
                self._source_max_truncated if self._truncate_long_sequences else self._source_max_skipped,
                self._source_max_tokens,
                "truncated" if self._truncate_long_sequences else "skipped",
            )

    def _preprocess(self, query: str) -> str:
        return query.lower() if self._uncased else query

    def text_to_instance(self, text: str, label: int = None, metadata: dict = None) -> Instance:
        if not metadata:
            metadata = dict()

        source = self._preprocess(text)

        tokenized_source = self._source_tokenizer.tokenize(source)
        if self._source_max_tokens and len(tokenized_source) > self._source_max_tokens:
            self._source_max_truncated += 1
            if self._truncate_long_sequences:
                tokenized_source = tokenized_source[: self._source_max_tokens]
        source_field = TextField(tokenized_source, self._source_token_indexers)

        fields = {"source_tokens": source_field, "metadata": MetadataField(metadata)}
        if label is not None:
            fields["label"] = LabelField(label, skip_indexing=True)

        return Instance(fields)
