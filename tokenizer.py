from transformers import AutoTokenizer
import torch
import config as cfg

class Tokenizer:

    def __init__(self, name = "gpt2"):
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.vocab_size = self.tokenizer.vocab_size

    def encode(self, input_string):
        return self.tokenizer([input_string], return_tensors="pt")['input_ids']

    def decode(self, encodings):
        # assert(len(encodings.shape) == 2)
        return self.tokenizer.batch_decode(encodings)

if __name__ == "__main__":

    tokenizer = Tokenizer(name=cfg.tokenizer_model)
    print(tokenizer.vocab_size)
    strings = ["This is a sample string", "this is the second string"]
    print("sentence:", strings)
    encoding = tokenizer.encode(strings)
    print(f"encoding:", encoding)
    decoding = tokenizer.decode(encoding)
    print("decoding:", decoding)

