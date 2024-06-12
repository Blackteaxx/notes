import torch
from torch import nn
import torch.nn.functional as F
from ProcessData import SOS_token, EOS_token, MAX_LENGTH, device

class DecoderRNN(nn.Module):
    def __init__(self, embedding_size, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.embedding = nn.Embedding(output_size, embedding_size)
        self.gru = nn.GRU(embedding_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, encoder_outputs, encoder_hidden, target_tensor=None):
        """

        :param encoder_outputs: [batch_size, seq_len, embedding_size]
        :param encoder_hidden: [num_layers, batch_size, hidden_size]
        :param target_tensor: [batch_size, seq_len(output)]
        :return:
        """
        batch_size = encoder_outputs.size(0)
        # [batch, 1]
        decoder_input = torch.empty(batch_size, 1,
                                    dtype=torch.long,
                                    device=device).fill_(SOS_token)
        decoder_hidden = encoder_hidden
        decoder_outputs = []

        for i in range(MAX_LENGTH):
            # decoder_input: [batch, 1]
            # decoder_hidden: [batch_size, hidden_size]
            decoder_output, decoder_hidden = self.forward_step(decoder_input, decoder_hidden)
            decoder_outputs.append(decoder_output)

            if target_tensor is not None:
                # Teacher forcing: Feed the target as the next input
                decoder_input = target_tensor[:, i].unsqueeze(1)  # Teacher forcing
            else:
                # Without teacher forcing: use its own predictions as the next input
                _, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze(-1).detach()  # detach from history as input

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
        return decoder_outputs, decoder_hidden, None  # We return `None` for consistency in the training loop

    def forward_step(self, input, hidden):
        output = self.embedding(input)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.out(output)
        return output, hidden

if __name__ == '__main__':
    decoder = DecoderRNN(10, 10, 5)

    input = torch.ones((5, 1), dtype=torch.long)
    hidden = torch.ones((1, 5, 10))

    print(decoder.forward_step(input, hidden))