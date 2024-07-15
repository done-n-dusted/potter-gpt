import os
import torch
from tokenizer import Tokenizer
from transformer import LanguageModel
from tqdm import tqdm
import csv
import yaml

with open("config.yml") as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)

files = [cfg['data_path']+x for x in os.listdir(cfg['data_path'])]
files.sort()

text = ""

for file in files:
    with open(file, 'r', encoding='utf-8') as f:
        text += f.read()

# text = wrap(text, 1024)
# text[-1] = text[-1] + " "*(1024 - len(text[-1]))
print(len(text))
tokenizer = Tokenizer(name=cfg['tokenizer_model'])
vocab_size = tokenizer.vocab_size

print(f"Vocab size of {vocab_size}")

# encoding = []

# for t in text:
#     encoding += list(tokenizer.encode(t))

# encoding = torch.cat(encoding, dim = 0)
# encoding = torch.tensor(encoding)
encoding = tokenizer.encode(text)
print(type(encoding))
print(f"encoding shape: {encoding.shape}")
n = int(len(encoding) * cfg['train_val_split'])
train = encoding[:n]
val = encoding[n:]

model = model = LanguageModel(vocab_size=vocab_size,
                          n_blocks = cfg['n_blocks'], 
                          n_embed=cfg['n_embed'],
                          num_heads=cfg['num_heads'],
                          block_size=cfg['block_size'],
                          dropout_rate=cfg['dropout_rate'])
# model.summary()
print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')
# print(f"Number of parameters is {torch.numel(model.parameters())}")
# assert False

def get_batch(split):
    data = train if split == "train" else val
    ix = torch.randint(len(data) - cfg['block_size'], (cfg['batch_size'], ))
    x = torch.stack([data[i:i+cfg['block_size']] for i in ix])
    y = torch.stack([data[i+1:i+cfg['block_size']+1] for i in ix])    
    return x, y

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

sentence = "I went to the clinic today and Sherlock came in running."
# prediction
idx = torch.tensor(tokenizer.encode(sentence), dtype = torch.long).view(1, -1)
# print(idx)
# print(tokenizer.decode(idx.view(-1,)))
# idx = torch.zeros((1, 1))
out_tokens = model.generate_sequence(idx, max_new_tokens = 20)[0].tolist()
output= tokenizer.decode(out_tokens)
print()
print("OUTPUT:", "".join(output))
print()
# assert False

losses = []
for step in tqdm(range(cfg['epochs'])):
    
    xb, yb = get_batch('train')
    xval, yval = get_batch('val')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    _, val_loss = model(xval, yval)
    loss.backward()
    optimizer.step()

    mod = cfg['epochs'] // 50
    if step%(max(mod, 1)) == 0:
        print(f"step: {step}")
        # out = tokenizer.
        res = tokenizer.decode(model.generate_sequence(idx, max_new_tokens = 150)[0].tolist())
        print(f"res : {''.join(res)}")
        print(f"training loss: {loss.item()}")
        print(f"validation loss: {val_loss.item()}")
        print()
        losses.append([step, loss.item(), ''.join(res)])

if cfg['log_name']:
    if not os.path.exists('results/'):
        os.makedirs('results/')
    with open('results/' + cfg['log_name'] + '.csv', "w") as f:
        wr = csv.writer(f)
        wr.writerows(losses)

if cfg['save_model']:
    if not os.path.exists('results/'):
        os.makedirs('results/')
    torch.save(model.state_dict(), 'results/' + cfg['save_model'] + '.pth' )