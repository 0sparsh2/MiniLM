import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
import urllib.request

# --- BITNET 1.58b IMPLEMENTATION ---

class RoundWithSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return torch.round(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

class BitLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=False):
        super(BitLinear, self).__init__(in_features, out_features, bias=bias)

    def forward(self, x):
        # 1. Weight Quantization to {-1, 0, 1}
        # scale = mean of absolute values
        weight_scale = self.weight.abs().mean()
        
        # Scale, round, clamp
        w_scaled = self.weight / (weight_scale + 1e-8)
        w_quant = torch.clamp(RoundWithSTE.apply(w_scaled), -1.0, 1.0)
        
        # 2. Activation Quantization to 8-bit [-128, 127]
        # per-token max absolute value
        x_scale = x.abs().max(dim=-1, keepdim=True)[0]
        x_scaled = x * (127.0 / (x_scale + 1e-8))
        x_quant = torch.clamp(RoundWithSTE.apply(x_scaled), -128.0, 127.0)
        
        # 3. Linear Projection (in a real engine this is int8 * int2 bitwise ops)
        out = F.linear(x_quant, w_quant)
        
        # 4. Dequantize
        out = out * (weight_scale * x_scale / 127.0)
        
        if self.bias is not None:
            out += self.bias
            
        return out

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps
        
    def forward(self, x):
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x

class BitTransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.ln_1 = RMSNorm(embed_dim)
        self.q_proj = BitLinear(embed_dim, embed_dim, bias=False)
        self.k_proj = BitLinear(embed_dim, embed_dim, bias=False)
        self.v_proj = BitLinear(embed_dim, embed_dim, bias=False)
        self.o_proj = BitLinear(embed_dim, embed_dim, bias=False)
        
        self.ln_2 = RMSNorm(embed_dim)
        self.gate_proj = BitLinear(embed_dim, embed_dim * 4, bias=False)
        self.up_proj = BitLinear(embed_dim, embed_dim * 4, bias=False)
        self.down_proj = BitLinear(embed_dim * 4, embed_dim, bias=False)
        
    def forward(self, x):
        batch, seq, _ = x.shape
        
        # --- Attention ---
        norm_x = self.ln_1(x)
        Q = self.q_proj(norm_x).view(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(norm_x).view(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(norm_x).view(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        mask = torch.triu(torch.ones(seq, seq, device=x.device), diagonal=1).bool()
        scores.masked_fill_(mask, float('-inf'))
        attn = torch.softmax(scores, dim=-1)
        
        context = torch.matmul(attn, V).transpose(1, 2).contiguous().view(batch, seq, self.embed_dim)
        x = x + self.o_proj(context)
        
        # --- SwiGLU FFN ---
        norm_x2 = self.ln_2(x)
        gate = F.silu(self.gate_proj(norm_x2))
        up = self.up_proj(norm_x2)
        x = x + self.down_proj(gate * up)
        
        return x

class BitGPT(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_layers, num_heads):
        super().__init__()
        # Continuous embeddings are usually kept continuous in BitNet
        self.vocab_embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Embedding(1024, embed_dim)
        
        self.layers = nn.ModuleList([BitTransformerBlock(embed_dim, num_heads) for _ in range(num_layers)])
        
        self.ln_f = RMSNorm(embed_dim)
        # Head is usually continuous or bitlinear, we'll use BitLinear for maximum compression!
        self.head = BitLinear(embed_dim, vocab_size, bias=False)
        
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear) or isinstance(module, BitLinear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x):
        batch, seq = x.shape
        pos = torch.arange(seq, device=x.device).unsqueeze(0)
        
        x = self.vocab_embed(x) + self.pos_embed(pos)
        
        for layer in self.layers:
            x = layer(x)
            
        x = self.ln_f(x)
        return self.head(x)

def get_batch(text, seq_length, batch_size, char_to_ix):
    ixs = torch.randint(0, len(text) - seq_length - 1, (batch_size,))
    x = torch.zeros(batch_size, seq_length, dtype=torch.long)
    y = torch.zeros(batch_size, seq_length, dtype=torch.long)
    for i, idx in enumerate(ixs):
        x[i] = torch.tensor([char_to_ix[ch] for ch in text[idx:idx+seq_length]])
        y[i] = torch.tensor([char_to_ix[ch] for ch in text[idx+1:idx+seq_length+1]])
    return x, y

def main():
    if not os.path.exists("tinyshakespeare.txt"):
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        urllib.request.urlretrieve(url, "tinyshakespeare.txt")

    with open("tinyshakespeare.txt", "r") as f:
        text = f.read()

    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    char_to_ix = {ch: i for i, ch in enumerate(chars)}
    ix_to_char = {i: ch for i, ch in enumerate(chars)}
    
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 256 dims, 4 layers, 4 heads
    model = BitGPT(vocab_size, embed_dim=256, num_layers=4, num_heads=4).to(device)
    
    # Count parameters
    params = sum(p.numel() for p in model.parameters())
    print(f"Total Model Parameters: {params}")
    print(f"File size if 16-bit Float: {params * 2 / 1024 / 1024:.2f} MB")
    print(f"File size at 1.58 bits: {params * 1.58 / 8 / 1024 / 1024:.2f} MB")
    
    # BitNet requires a slightly higher learning rate
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    
    batch_size = 128
    seq_length = 64
    
    print("Training the 1.58b BitNet Transformer...")
    for step in range(600):
        x, y = get_batch(text, seq_length, batch_size, char_to_ix)
        x, y = x.to(device), y.to(device)
        
        optimizer.zero_grad()
        logits = model(x)
        
        loss = criterion(logits.reshape(-1, vocab_size), y.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        if step % 50 == 0:
            perplexity = torch.exp(loss)
            print(f"Step {step} | Loss: {loss.item():.4f} | Perplexity: {perplexity.item():.4f}")

    print("Saving model weights to bitnet_model.pt...")
    torch.save(model.state_dict(), "bitnet_model.pt")

    print("\n--- GENERATING TEXT ---")
    model.eval()
    with torch.no_grad():
        x = torch.tensor([[char_to_ix['T']]]).to(device)
        out_text = 'T'
        for _ in range(300):
            logits = model(x)
            # Take the last logit
            logits = logits[:, -1, :]
            probs = F.softmax(logits / 0.8, dim=-1)
            next_ix = torch.multinomial(probs, 1).item()
            out_text += ix_to_char[next_ix]
            
            # Append next char
            next_tensor = torch.tensor([[next_ix]]).to(device)
            x = torch.cat([x, next_tensor], dim=1)
            # Truncate context if too long
            if x.size(1) > seq_length:
                x = x[:, -seq_length:]
                
        print(out_text)

if __name__ == '__main__':
    main()
