from typing import Dict, Optional

from overrides import overrides

import torch
from allennlp.models.model import Model
from allennlp.data import Vocabulary, TextFieldTensors
from allennlp.training.metrics import Average, Auc

from torch import Tensor

from transformers import BertModel, BertForSequenceClassification


@Model.register("hatefulmememodel")
class HatefulMemeModel(Model):
    def __init__(self, vocab: Vocabulary, text_model_name: str):
        super().__init__(vocab)
        self._text_model = BertForSequenceClassification.from_pretrained(text_model_name)
        self._num_labels = vocab.get_vocab_size()

        self._accuracy = Average()
        self._auc = Auc()

        self._softmax = torch.nn.Softmax(dim=1)

    def forward(
        self,
        source_tokens: TextFieldTensors,
        box_features: Optional[Tensor] = None,
        box_coordinates: Optional[Tensor] = None,
        box_mask: Optional[Tensor] = None,
        label: Optional[Tensor] = None,
        metadata: Optional[Dict] = None,
    ) -> Dict[str, torch.Tensor]:
        input_ids = source_tokens["tokens"]["token_ids"]
        input_mask = source_tokens["tokens"]["mask"]
        token_type_ids = source_tokens["tokens"]["type_ids"]
        outputs = self._text_model(
            input_ids=input_ids,
            attention_mask=input_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
            labels=label,
        )

        if label is not None:
            predictions = torch.argmax(self._softmax(outputs.logits), dim=-1)
            for index in range(predictions.shape[0]):
                correct = float((predictions[index] == label[index]))
                self._accuracy(int(correct))

            self._auc(predictions, label)

        return outputs

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics: Dict[str, float] = {}
        if not self.training:
            metrics["accuracy"] = self._accuracy.get_metric(reset=reset)
            metrics["auc"] = self._auc.get_metric(reset=reset)
        return metrics
