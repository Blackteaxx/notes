## 参考

[Pytorch Tutorial](https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html)

## Introduction

翻译任务有别于情感分析这样的分类任务，它主要关注将一个**变长的源语言序列**转录成一个**变长的目标语言序列**。

而为了实现这个目标，我们需要一个更加 general 的模型框架，就是 encoder-decoder， encoder 目的在于将输入的变长序列转换为一个 latent vector，decoder 目的在于将这个 latent vector 转换为一个变长的输出序列。

值得注意的是，在训练过程中，latent vector 一起与 true decoder sequence 一起被输入到 decoder 中，这样可以使得 decoder 在生成每个词的时候都能够考虑到整个输入序列。

而在生成过程中，decoder 会根据上一个生成的词以及 latent vector 生成下一个词，这样就可以逐步生成整个输出序列。

而在这个教程中，我们将会使用一个简单的 encoder-decoder 模型来完成一个简单的翻译任务。

## Preparing Data

需要将输入输出序列对转换为 idx，这里我们手工制作 Lang 类来完成这个任务。

```python
class Lang:
    def __init__(self, name):
        self.name  = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {
            0: "SOS",
            1: "EOS"
        }
        self.num_words = 2

    def addSentence(self, sentence):
        """
        add Sentence to Vocab,
        update word2index, word2count, index2word
        :param sentence:
        :return:
        """

        for word in sentence.split(" "):
            self.addWord(word)

    def addWord(self, word):
        if word in self.word2count.keys():
            self.word2count[word] += 1
        else:
            self.index2word[self.num_words] = word
            self.word2index[word] = self.num_words
            self.word2count[word] = 1

            self.num_words += 1

    def __len__(self):
        return self.num_words

    def __getitem__(self, item):
        return self.index2word[item]
```

```python
def prepareData(lang1, lang2, reverse=False):
    input_lang, output_lang, pairs = readLangs(lang1, lang2,
                                               reverse)

    pairs = filterPairs(pairs)

    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])

    print(f"input_lang size: {len(input_lang)}")
    print(f"output_lang size: {len(output_lang)}")

    return input_lang, output_lang, pairs
```

## Encoder-Decoder Architecture

![alt text](image.png)

## Encoder

我们需要使用 RNN 将输入序列转换为一个融合信息的 latent vector，也叫 context vector。

在 pytorch 中，`nn.RNN` 的输入在设置`batch_first=True`的情况下，输入的形状是 `[batch_size, seq_len, input_size]`，输出的形状是 `[batch_size, seq_len, hidden_size], [num_layers, batch_size, hidden_size]`。其中 seq_len 是输入序列的长度，虽然输入的是矩阵形式，但是 RNN 依旧是时序化处理的，并非并行。

如果`seq_len = 1`, 我们就能进行单步的处理。

值得注意的是，`nn.RNN` 可以指定 `hidden_input`，这个参数是用来初始化 hidden state 的，如果不指定，那么 hidden state 就会被初始化为 0。

同时，如果使用`drop_out`, 那么 RNN 必须是要多层的，因为 drop_out 是在每一层之间进行的。

```python
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
        output, hidden = self.gru(X)
        return output, hidden
```

## Decoder

分为两种模式：训练模式和生成模式。

在训练模式中，使用上一层的 hidden state 以及上一层真实的输出作为输入。

在生成模式中，使用上一层的 hidden state 以及**上一层生成**的输出（top1）作为输入。

因此，我们需要按序列生成。而真实输出是`[batch_size, seq_len]`，因此我们需要按序列生成， 并`cat(dim=1)`。

```python
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

                # return value, indices
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
```

## train

在前文已经将 encoder 和 decoder 定义好了，并且和期望的输出格式也是一致的，因此我们可以直接进行训练，并使用`nn.NLLLoss`作为损失函数。

```python
def train_epoch(dataloader, encoder, decoder, encoder_optimizer, decoder_optimizer,
                criterion):

    total_loss = 0
    for input_tensor, target_tensor in tqdm(dataloader, desc="training..."):

        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        encoder_outputs, encoder_hidden = encoder(input_tensor)
        decoder_outputs, _, _ = decoder(encoder_outputs, encoder_hidden, target_tensor)

        loss = criterion(
            decoder_outputs.view(-1, decoder_outputs.size(-1)),
            target_tensor.view(-1)
        )

        loss.backward()

        encoder_optimizer.step()
        decoder_optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)
```
