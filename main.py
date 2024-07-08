import os
import torch
from tokenizer import Tokenizer
import config as cfg
from transformer import LanguageModel
from tqdm import tqdm
from textwrap import wrap
import csv

files = ['../sherlock/'+x for x in os.listdir('../sherlock')]
files.sort()

text = ""

for file in files:
    with open(file, 'r', encoding='utf-8') as f:
        text += f.read()

text = wrap(text, 1024)
# text[-1] = text[-1] + " "*(1024 - len(text[-1]))

tokenizer = Tokenizer(name=cfg.tokenizer_model)
vocab_size = tokenizer.vocab_size

print(f"Vocab size of {vocab_size}")

encoding = []

for t in text:
    encoding += list(tokenizer.encode(t))

encoding = torch.cat(encoding, dim = 0)
# encoding = torch.tensor(encoding)

print(f"number of tokens after encoding: {len(encoding)*1024}")

n = int(len(encoding) * cfg.train_val_split)
train = encoding[:n]
val = encoding[n:]

model = LanguageModel(vocab_size=vocab_size)

print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')
# print(f"Number of parameters is {torch.numel(model.parameters())}")

def get_batch(split):
    data = train if split == "train" else val
    ix = torch.randint(len(data) - cfg.block_size, (cfg.batch_size, ))
    x = torch.stack([data[i:i+cfg.block_size] for i in ix])
    y = torch.stack([data[i+1:i+cfg.block_size+1] for i in ix])    
    return x, y

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

sentence = "I went to the clinic today and Sherlock came in running."
# prediction
idx = torch.tensor(tokenizer.encode(sentence)[0], dtype = torch.long).view(1, -1)
# print(idx)
# idx = torch.zeros((1, 1))
out_tokens = model.generate_sequence(idx, max_new_tokens = 20)[0].tolist()
output= tokenizer.decode(out_tokens)
print()
print("OUTPUT:", "".join(output))
print()

losses = []
for step in tqdm(range(cfg.epochs)):
    
    xb, yb = get_batch('train')
    
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    if step%(cfg.epochs//50) == 0:
        print(f"step: {step}")
        # out = tokenizer.
        res = tokenizer.decode(model.generate_sequence(idx, max_new_tokens = 20)[0].tolist())
        print(f"res : {''.join(res)}")
        print(f"loss: {loss.item()}")
        print()
        losses.append([step, loss.item(), ''.join(res)])

with open("log.csv", "w") as f:
    wr = csv.writer(f)
    wr.writerows(losses)