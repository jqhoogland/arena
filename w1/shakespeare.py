#%%

import random
import re

import torch as t
import torch.nn.functional as F
import transformers
from arena.w1.attention import DecoderOnlyTransformer, TransformerConfig
from torch import nn
from torch.utils.data import DataLoader, Dataset

# %%

with open("./shakespeare.txt", "r") as f:
    shakespeare = f.read()
    print(shakespeare[:100])    

shakespeare_tokens = list(set(re.split(r"\b", shakespeare)))

def tokenize(text: str) -> t.Tensor:
    return t.tensor([shakespeare_tokens.index(token) for token in re.split(r"\b", text) if token])

def detokenize(tokens: t.Tensor) -> str:
    return "".join([shakespeare_tokens[token] for token in tokens])

# %%

config = TransformerConfig(
    vocab_size=len(shakespeare_tokens),
    hidden_size=512,
    num_layers=6,
    num_heads=8,
    max_seq_len=512,
    dropout=0.1,
)

transformer = DecoderOnlyTransformer(config)

#%%

class ShakespeareDataset(Dataset):
    def __init__(self, corpus: str, seq_len: int):
        self.inputs = []
        self.targets = []

        tokens = tokenize(corpus)

        for i in range(0, len(corpus) - seq_len, seq_len):
            self.inputs.append(tokens[i : i + seq_len])
            self.targets.append(tokens[i + 1 : i + seq_len + 1])

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx: int):
        input_ = self.inputs[idx]
        target = self.targets[idx]
        return input_, target

# %%

train_corpus = shakespeare[:int(len(shakespeare) * 0.8)]
test_corpus = shakespeare[int(len(shakespeare) * 0.8) :]

train_data = ShakespeareDataset(train_corpus, 128)
test_data = ShakespeareDataset(test_corpus, 128)

train_dataloader = DataLoader(train_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

# %%

config = TransformerConfig(
    vocab_size=100,
    hidden_size=128,
    num_layers=4,
    num_heads=4,
    max_seq_len=10
)

transformer = DecoderOnlyTransformer(config)
# %%

loss = t.tensor(0.)
loss_fn = nn.CrossEntropyLoss()
optimizer = t.optim.Adam(transformer.parameters(), lr=1e-3)

for epoch in range(10):
    for batch in train_dataloader:
        input_, target = batch

        output = transformer(input_)
        loss = loss_fn(output.reshape(-1, 100), target.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


    for batch in test_dataloader:
        input_, target = batch

        output = transformer(input_)
        loss = loss_fn(output.reshape(-1, 100), target.reshape(-1))
        print(loss)


 # %%

for batch in test_dataloader:
    input_, target = batch

    output = transformer(input_)

    print(output.argmax(dim=2)[0], target[0])
    loss = loss_fn(output.reshape(-1, 100), target.reshape(-1))
    print("\n")
# %%

# TODO: The model is just learning to repeat guesses. I need to play around with longer training runs & different hyperparameter combos
