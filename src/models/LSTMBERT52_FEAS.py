import torch.nn as nn
import torch

from settings import VOCAB_SIZE
from .transformer import TransformerBlock


class LSTMBERT52_FEAS(nn.Module):
    def __init__(
        self,
        vocab_size=VOCAB_SIZE,
        embedding_dim=128,
        hidden_dim=128,
        output_dim=15,
        n_layers=4,
        attn_heads=4,
        n_attn_layers=4,
        dropout=0.1,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ):
        super(LSTMBERT52_FEAS, self).__init__()
        self.device = device
        self.embedding = nn.Embedding(vocab_size + 2, embedding_dim, padding_idx=0)
        self.vowels_num = nn.Embedding(output_dim, hidden_dim, padding_idx=0)
        self.is_norm = nn.Embedding(2, hidden_dim, padding_idx=0)
        self.part = nn.Embedding(9, hidden_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=n_layers,
            bidirectional=False,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(hidden_dim, attn_heads, hidden_dim, dropout)
                for _ in range(n_attn_layers)
            ]
        )
        self.classifier = nn.Linear(hidden_dim, output_dim)

        self.lstm.apply(weight_init)
        self.transformer_blocks.apply(weight_init)
        self.classifier.apply(weight_init)
        self.embedding.apply(weight_init)
        self.is_norm.apply(weight_init)
        self.part.apply(weight_init)
        self.vowels_num.apply(weight_init)

    def forward(self, batch):
        items = batch["items"]
        seq = add_cls(items, self.device)
        embedded = self.embedding(seq)
        output, (hidden, cell_state) = self.lstm(embedded)
        mask = torch.ones(seq.size()[0], 1, seq.size()[1], seq.size()[1]).to(
            self.device
        )
        mask[:, :, -1, :] = 0
        for transformer in self.transformer_blocks:
            output = transformer.forward(output, mask)
        outputs = (
            output[:, -1, :]
            + self.is_norm(batch["is_norm"])
            + self.part(batch["part"])
            + self.vowels_num(batch["vowels_num"])
        )
        logits = self.classifier(outputs)
        return logits


def add_cls(items, vocab_size=VOCAB_SIZE, device=torch.cpu()):
    cls_tensor = (
        torch.tensor([vocab_size + 1] * items.size()[0]).reshape(-1, 1).to(device)
    )
    return torch.cat([items, cls_tensor], dim=1)


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight)
        m.bias.data.zero_()