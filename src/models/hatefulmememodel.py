from typing import Dict

from overrides import overrides

import torch
from allennlp.models.model import Model
from allennlp.data import Vocabulary, TextFieldTensors
from allennlp.training.metrics import Average

from torch import Tensor

from transformers import BertModel


@Model.register("hatefulmememodel")
class HatefulMemeModel(Model):
    def __init__(self, vocab: Vocabulary, text_model_name: str):
        super().__init__(vocab)
        self._text_model = BertModel.from_pretrained(text_model_name)
        self._num_labels = vocab.get_vocab_size()

        self._accuracy = Average()

    def forward(
        self, source_tokens: TextFieldTensors, label: Tensor = None, metadata: Dict = None
    ) -> Dict[str, torch.Tensor]:
        input_ids, input_mask = source_tokens["tokens"]["token_ids"], source_tokens["tokens"]["mask"]

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return self._accuracy.get_metric(reset=reset)
