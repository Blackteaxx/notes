from torch import nn

class Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size,
                 drop_out=0.1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(vocab_size, embedding_size)

        self.gru = nn.GRU(embedding_size, hidden_size,
                          dropout=drop_out,
                          batch_first=True)

    def forward(self, X):
        """

        :param X: [batch_size , seq_len]
        :return:
        """
        # [batch_size , seq_len] -> [batch_size , seq_len, embedding_size]
        embedded = self.embedding(X)

        # hidden: [num_layers, batch_size, hidden_size]
        output, hidden = self.gru(embedded)
        return output, hidden

