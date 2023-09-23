import torch.nn as nn
import torch

from settings import VOCAB_SIZE
from .transformer import TransformerBlock


class AttnLSTM52(nn.Module):
    def __init__(
        self,
        vocab_size=VOCAB_SIZE,
        embedding_dim=128,
        hidden_dim=128,
        output_dim=15,
        n_layers=4,
        bidirectional=False,
        dropout=0.1,
    ):
        super(AttnLSTM52, self).__init__()
        self.embedding = nn.Embedding(vocab_size + 2, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=n_layers,
            bidirectional=bidirectional,
            dropout=dropout,
            batch_first=True,
        )
        self.attention = TransformerBlock(
            hidden=hidden_dim,
            attn_heads=1,
            feed_forward_hidden=hidden_dim,
            dropout=dropout,
        )
        self.classifier = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, batch):
        items = batch["items"]
        seq = add_cls(items)
        embedded = self.embedding(seq)
        output, (hidden, cell_state) = self.lstm(embedded)
        mask = torch.ones(seq.size()[0], 1, seq.size()[1], seq.size()[1]).to('cuda')
        mask[:, :, -1, :] = 0
        attn_out = self.attention(output, mask)
        logits = self.classifier(attn_out[:, -1, :])
        return logits


def add_cls(items, vocab_size=VOCAB_SIZE):
    cls_tensor = torch.tensor([vocab_size + 1] * items.size()[0]).reshape(-1, 1).to('cuda')
    return torch.cat([items, cls_tensor], dim=1)
