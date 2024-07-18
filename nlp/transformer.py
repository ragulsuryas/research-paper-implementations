import torch
import torch.nn as nn

"""
B(batch size), T(sequence length), C(embedding dimension), T(head dimension), N(number of heads)
"""

class FeedForward(nn.Module):
    def __init__(self, embd_dim):
        super(FeedForward, self).__init__()
        self.f1 = nn.Linear(embd_dim, 4*embd_dim)
        self.f2 = nn.Linear(4*embd_dim, embd_dim)
    
    def forward(self, x):
        x = torch.relu(self.f1(x)) # B, T, C
        out = torch.relu(self.f2(x))
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, emdb_dim):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = emdb_dim // num_heads
        
        assert emdb_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads"
        
        self.k = nn.Linear(emdb_dim, emdb_dim)
        self.q = nn.Linear(emdb_dim, emdb_dim)
        self.v = nn.Linear(emdb_dim, emdb_dim)
        
    def forward(self, key, query, value, mask=None):
        batch_size, seq_len, _ = key.shape
        
        key = self.k(key)  # B, T, C
        query = self.q(query)
        value = self.v(value)
        
        key = key.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # B, T, C -> B, N, T, H
        query = query.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        scores = query @ key.transpose(-1, -2) // (self.head_dim ** 0.5)  # B, N, T, H @ B, N, H, T -> B, N, T, T
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
            
        attn = torch.softmax(scores, dim=-1)  # B, N, T, T
        
        out = attn @ value  # B, N, T, T @ B, N, T, H -> B, N, T, H
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.num_heads * self.head_dim)  # B, N, T, H -> B, T, N, H -> B, T, N*H
        
        return out

class Encoder(nn.Module):
    def __init__(self, num_heads, emdb_dim):
        super(Encoder, self).__init__()
        self.sa = MultiHeadAttention(num_heads, emdb_dim)
        self.ff = FeedForward(emdb_dim)
        self.ln1 = nn.LayerNorm(emdb_dim)
        self.ln2 = nn.LayerNorm(emdb_dim)
    
    def forward(self, x):
        x = self.ln1(x + self.sa(x, x, x))
        x = self.ln2(x + self.ff(x))
        return x

class Decoder(nn.Module):
    def __init__(self, num_heads, emdb_dim):
        super(Decoder, self).__init__()
        self.msa = MultiHeadAttention(num_heads, emdb_dim)
        self.sa = MultiHeadAttention(num_heads, emdb_dim)
        self.ff = FeedForward(emdb_dim)
        self.ln1 = nn.LayerNorm(emdb_dim)
        self.ln2 = nn.LayerNorm(emdb_dim)
        self.ln3 = nn.LayerNorm(emdb_dim)
    
    def forward(self, enc_out, x):
        _, seq_len, _ = x.shape
        x = self.ln1(x + self.msa(x, x, x))
        
        mask = torch.tril(torch.ones((seq_len, seq_len)))
        x = self.ln2(x + self.sa(enc_out, enc_out, x, mask))
        x = self.ln3(x + self.ff(x))
        return x

class Transformer(nn.Module):
    def __init__(self, num_layers, num_heads, vocab_size, emdb_dim, context_len):
        super(Transformer, self).__init__()
        self.tok_embedding_enc = nn.Embedding(vocab_size, emdb_dim)
        self.pos_embedding_enc = nn.Embedding(context_len, emdb_dim)
        self.encoders = nn.ModuleList([Encoder(num_heads, emdb_dim) for _ in range(num_layers)])
        
        self.tok_embedding_dec = nn.Embedding(vocab_size, emdb_dim)
        self.pos_embedding_dec = nn.Embedding(context_len, emdb_dim)
        self.decoders = nn.ModuleList([Decoder(num_heads, emdb_dim) for _ in range(num_layers)])
        
        self.fc = nn.Linear(emdb_dim, vocab_size)
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, src, trg):
        B, src_len = src.shape
        B, trg_len = trg.shape
        src_pos = torch.arange(0, src_len)
        trg_pos = torch.arange(0, trg_len)
        
        tok = self.tok_embedding_enc(src)
        pos = self.pos_embedding_dec(src_pos)
        src = tok + pos
        for encoder in self.encoders:
            src = encoder(src)
            
        tok = self.tok_embedding_dec(trg)
        pos = self.pos_embedding_dec(trg_pos)
        trg = tok + pos
        for decoder in self.decoders:
            trg = decoder(src, trg)
            
        out = self.fc(trg)
        return self.softmax(out)
        
if __name__ == "__main__":
    src = torch.rand((32, 128)).long()
    trg = torch.rand((32, 128)).long()
    
    model = Transformer(num_layers=6, num_heads=8, vocab_size=20000, emdb_dim=512, context_len=128)
    out = model(src, trg)
    print(out.shape) # (32, 128, 20000) -> the next token in the target lenguage