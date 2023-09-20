import torch.nn as nn

from settings import VOCAB_SIZE


class BiLSTM52(nn.Module):
    def __init__(
        self,
        vocab_size=VOCAB_SIZE,
        embedding_dim=128,
        hidden_dim=128,
        output_dim=15,
        n_layers=4,
        bidirectional=True,
        dropout=0.1,
    ):
        super(BiLSTM52, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=n_layers,
            bidirectional=bidirectional,
            dropout=dropout,
            batch_first=True,
        )

        self.classifier = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, batch):
        items = batch["items"]
        embedded = self.embedding(items)
        output, (hidden, cell_state) = self.lstm(embedded)
        print(hidden.size())
        logits = self.classifier(hidden[-1, :, :])
        return logits
