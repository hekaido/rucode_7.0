import torch.nn as nn
from transformers import BertConfig, BertForSequenceClassification

from settings import VOCAB_SIZE, MAX_WORD_LEN


class BERT52(nn.Module):
    def __init__(
        self,
        hidden_dim=128,
        out_dim=15,
        vocab_size=VOCAB_SIZE,
        hidden_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        intermediate_size=128,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=MAX_WORD_LEN,
    ):
        super().__init__()
        bert_config = BertConfig(
            vocab_size=vocab_size,
            hidden_size=hidden_dim,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            hidden_act=hidden_act,
            hidden_dropout_prob=hidden_dropout_prob,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            max_position_embeddings=max_position_embeddings,
        )
        self.bert = BertForSequenceClassification(bert_config)
        self.bert.classifier = nn.Linear(hidden_dim, out_dim)

    def forward(self, batch):
        items = batch["items"]
        mask = items > 0
        x = self.bert(items, mask).logits
        return x
