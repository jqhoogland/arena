#%% 
import torch as t
from arena.w1.attention import DecoderOnlyTransformer, TransformerConfig
from torch import nn
from torch.utils.data import DataLoader, Dataset

#%%

class ReverseASequenceDataset(Dataset):
    def __init__(self, seq_len, total_size):
        self.seq_len = seq_len
        self.total_size = total_size

        self.inputs = t.randint(0, 100, (total_size, seq_len))
        self.targets = self.inputs[:, :].flip(1)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx: int):
        input_ = self.inputs[idx]
        target = self.targets[idx]
        return input_, target
# %%

train_data = ReverseASequenceDataset(10, 10000)
test_data = ReverseASequenceDataset(10, 100)

train_dataloader = DataLoader(train_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

# %%

config = TransformerConfig(
    vocab_size=100,
    hidden_size=32,
    num_layers=4,
    num_heads=4,
    max_seq_len=10
)

transformer = DecoderOnlyTransformer(config)
# %%

loss = t.tensor(0.)
loss_fn = nn.CrossEntropyLoss()
optimizer = t.optim.Adam(transformer.parameters(), lr=1e-3)

for epoch in range(20):
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