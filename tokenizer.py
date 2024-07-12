from transformers import AutoTokenizer
import torch
from textwrap import wrap

class Tokenizer:

    def __init__(self, name = "gpt2"):
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.tokenizer.pad_token = "[PAD]"
        self.vocab_size = self.tokenizer.vocab_size

    def encode(self, input_string: str):
        max_pos = self.tokenizer.model_max_length
        text = wrap(input_string, max_pos)
        encoding = []

        for t in text:
            encoding += list(self.tokenizer(t, return_tensors="pt", padding=False)['input_ids'])
        encoding = torch.cat(encoding, dim = 0)
        return encoding

    def decode(self, encodings):
        # assert(len(encodings.shape) == 2)
        return self.tokenizer.batch_decode(encodings, skip_special_tokens=True)

if __name__ == "__main__":

    tokenizer = Tokenizer(name="gpt2")
    print(tokenizer.vocab_size)
    strings = ["This is a sample string", "this is the second string"]
    print("sentence:", strings)
    encoding = tokenizer.encode(strings)
    print(f"encoding:", encoding)
    decoding = tokenizer.decode(encoding)
    print("decoding:", decoding)

