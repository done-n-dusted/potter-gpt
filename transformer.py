import torch
from torch import nn
from torch.nn import functional as F
import config as cfg

class Head(nn.Module):
    """One self attention head class"""
    # C is n_embed look up. Will be defined in final transformer model
    def __init__(self, head_size, n_embed=cfg.n_embed, is_encoder=False):
        super().__init__()
        self.is_encoder = is_encoder
        self.hs = head_size
        self.key = nn.Linear(n_embed, head_size, bias = False) # C to HS
        self.query = nn.Linear(n_embed, head_size, bias = False) # C to HS
        self.value = nn.Linear(n_embed, head_size, bias = False) # C to HS

        self.register_buffer("tril", torch.tril(torch.ones(cfg.block_size, cfg.block_size))) # T x T
        self.dropout = nn.Dropout(cfg.dropout_rate)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x) # (B, T, HS)
        q = self.key(x) # (B, T, HS)
        v = self.key(x) # (B, T, HS)

        weights = q @ k.transpose(-2, -1) * C**(-0.5) # (B, T, C) * (B, C, T) -> (B, T, T)
        if not self.is_encoder:
            weights = weights.masked_fill(self.tril[:T, :T] == 0, -float('inf')) # cut context to future tokens
        weights = F.softmax(weights, dim=-1)
        weights = self.dropout(weights) # B x T x T

        out = weights @ v # (B, T, T) x (B, T, Hs) -> (B, T, Hs)
        return out

class MultiHead(nn.Module):

    def __init__(self, head_size, num_heads=cfg.num_heads):
        super().__init__()

        self.heads = nn.ModuleList([Head(n_embed=cfg.n_embed, head_size=head_size) for _ in range(num_heads)])
        self.projection = nn.Linear(cfg.n_embed, cfg.n_embed)
        self.dropout = nn.Dropout(cfg.dropout_rate)


    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.projection(out))
        return out

    

class FeedForward(nn.Module):

    def __init__(self, n_embed = cfg.n_embed):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4*n_embed,n_embed),
            nn.Dropout(cfg.dropout_rate),
        )
    
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embed = cfg.n_embed, num_heads = cfg.num_heads):
        super().__init__()

        head_size = n_embed // num_heads
        self.multihead = MultiHead(num_heads=num_heads, head_size=head_size)
        self.feedforward = FeedForward(n_embed)
        self.layer_norm1 = nn.LayerNorm(n_embed)
        self.layer_norm2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        x = x + self.multihead(self.layer_norm1(x))
        x = x + self.feedforward(self.layer_norm2(x))

        return x

class LanguageModel(nn.Module):

    def __init__(self, vocab_size, n_blocks = cfg.n_blocks, 
                 n_embed = cfg.n_embed, num_heads = cfg.num_heads, 
                 block_size=cfg.block_size):
        super().__init__()

        self.vocab_size = vocab_size
        self.n_blocks = n_blocks
        self.n_embed = n_embed
        self.num_heads = num_heads
        self.block_size = block_size

        self.token_embedding = nn.Embedding(self.vocab_size, self.n_embed)
        self.position_embedding = nn.Embedding(self.block_size, self.n_embed)
        self.blocks = nn.Sequential(*[Block(self.n_embed, self.num_heads) for _ in range(self.n_blocks)])

        self.linear = nn.Linear(self.n_embed, self.vocab_size)
    
    def forward(self, idx, targets = None):
        B, T = idx.shape

        tokens = self.token_embedding(idx)
        positions = self.position_embedding(torch.arange(T, device=cfg.device))

        x = tokens + positions
        x = self.blocks(x)
        logits = self.linear(x)

        if targets == None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate_sequence(self, idx, max_new_tokens):
        B, T = idx.shape
        for _ in range(max_new_tokens):
            idx_context = idx[:, -self.block_size:] # take last T tokens
            logits, loss = self(idx_context)
            # _, _, C = logits.shape
            logits = logits[:, -1, :] # B, C
            probs = F.softmax(logits, dim = -1)
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # print("idx_next shape:", idx_next.shape)
            # print("idx shape", idx.shape)
            idx = torch.cat([idx, idx_next], dim = 1)
        return idx

if __name__ == '__main__':
    model = LanguageModel(vocab_size=50)
    X = torch.randint(0, 50, (1, cfg.block_size), dtype=torch.int)
    print(X)
    logits, _ = model(X)
    print("logits:", logits)
    print(logits.shape)
    pass

