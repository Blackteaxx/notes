from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import re
import torch

SOS_token = 0
EOS_token = 1
MAX_LENGTH = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

eng_prefixes = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s ",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re "
)

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

# Turn a Unicode string to plain ASCII, thanks to
# https://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# Lowercase, trim, and remove non-letter characters
def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z!?]+", r" ", s)
    return s.strip()

def readLangs(lang1, lang2, reverse=True):
    print("Reading File into Lines...")

    # Read file
    lines = open('data/%s-%s.txt' % (lang1, lang2), encoding='utf-8'). \
        read().strip().split('\n')

    # split sentences and normalize
    pairs = [[normalizeString(s) for s in l.split("\t")] for l in lines]

    # Reverse pairs, make sentences
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs

def filterPair(p):
    return len(p[0].split(' ')) < MAX_LENGTH and \
        len(p[1].split(' ')) < MAX_LENGTH

def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]

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

if __name__ == '__main__':
    input_lang, output_lang, pairs = prepareData("eng", "fra", True)
    print(input_lang[3], output_lang[3])

    import random

    print(random.choice(pairs))