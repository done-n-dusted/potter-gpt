import os
import torch
from tokenizer import Tokenizer
import config as cfg

files = ['../harrypotter/'+x for x in os.listdir('harrypotter')]
files.sort()

text = ""

for file in files[:1]:
    with open(file, 'r', encoding='utf-8') as f:
        text += f.read()

tokenizer = Tokenizer(name=cfg.tokenizer_model)
