import torch
import torch.nn as nn
import os
import urllib.request
import math

class RowHyperNet(nn.Module):
    def __init__(self, layer_count, row_count, embed_dim, hidden_dim=64):
        super().__init__()
        self.layer_count = layer_count
        self.row_count = row_count
        self.embed_dim = embed_dim
        
        # DNA: The highly compressed embeddings
        # We need distinct weight matrices per layer: Q, K, V, O, FFN1, FFN2
        # So we have layer_count * 6 matrices
        self.matrix_embeds = nn.Embedding(layer_count * 6, 16)
        self.row_embeds = nn.Embedding(row_count, 16)
        
        # DNA: The Synthesis Function
        self.generator = nn.Sequential(
            nn.Linear(32, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim) # Outputs one full row of the virtual matrix
        )
        
        # Initialization
        with torch.no_grad():
            self.generator[-1].weight.uniform_(-0.01, 0.01)
            
    def generate_matrix(self, matrix_id):
        # Generate a (row_count, embed_dim) matrix
        m_emb = self.matrix_embeds(torch.tensor(matrix_id, device=self.matrix_embeds.weight.device)) # (16,)
        m_emb = m_emb.unsqueeze(0).expand(self.row_count, -1) # (row_count, 16)
        r_emb = self.row_embeds.weight # (row_count, 16)
        
        x = torch.cat([m_emb, r_emb], dim=-1) # (row_count, 32)
        
        matrix = self.generator(x) # (row_count, embed_dim)
        
        scale = 1.0 / math.sqrt(self.embed_dim)
        return matrix * scale

class VirtualTransformerLayer(nn.Module):
    def __init__(self, hypernet, layer_id):
        super().__init__()
        self.hypernet = hypernet
        self.layer_id = layer_id
        self.m_id = layer_id * 6
        
        self.ln1 = nn.LayerNorm(hypernet.embed_dim)
        self.ln2 = nn.LayerNorm(hypernet.embed_dim)
        
    def forward(self, x):
        # Generate matrices dynamically!
        W_Q = self.hypernet.generate_matrix(self.m_id + 0)
        W_K = self.hypernet.generate_matrix(self.m_id + 1)
        W_V = self.hypernet.generate_matrix(self.m_id + 2)
        W_O = self.hypernet.generate_matrix(self.m_id + 3)
        
        norm_x = self.ln1(x)
        Q = torch.matmul(norm_x, W_Q.t())
        K = torch.matmul(norm_x, W_K.t())
        V = torch.matmul(norm_x, W_V.t())
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(Q.size(-1))
        mask = torch.triu(torch.ones(scores.size(-2), scores.size(-1), device=x.device), diagonal=1).bool()
        scores.masked_fill_(mask, float('-inf'))
        attn = torch.softmax(scores, dim=-1)
        
        context = torch.matmul(attn, V)
        x = x + torch.matmul(context, W_O.t())
        
        W_F1 = self.hypernet.generate_matrix(self.m_id + 4)
        W_F2 = self.hypernet.generate_matrix(self.m_id + 5)
        
        norm_x2 = self.ln2(x)
        ffn = torch.nn.functional.gelu(torch.matmul(norm_x2, W_F1.t()))
        x = x + torch.matmul(ffn, W_F2.t())
        
        return x

class HyperLLM(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_layers):
        super().__init__()
        self.vocab_embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Embedding(512, embed_dim)
        
        self.hypernet = RowHyperNet(layer_count=num_layers, row_count=embed_dim, embed_dim=embed_dim)
        self.layers = nn.ModuleList([VirtualTransformerLayer(self.hypernet, i) for i in range(num_layers)])
        
        self.ln_f = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size)
        
        params = sum(p.numel() for p in self.parameters())
        virtual_params = num_layers * 6 * embed_dim * embed_dim
        print(f"--- HYPERNETWORK TRANSFORMER INITIALIZED ---")
        print(f"Total Stored DNA Model Size: {params} params ({params*4/1024:.2f} KB)")
        print(f"Virtual Parameter Count: {virtual_params} params ({virtual_params*4/1024/1024:.2f} MB)")
        print(f"Compression Ratio: {virtual_params/params:.2f}x")
        
    def forward(self, x):
        seq = x.size(1)
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
    
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 256 dims, 4 layers
    model = HyperLLM(vocab_size, embed_dim=256, num_layers=4).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    batch_size = 128
    seq_length = 64
    
    print("Training the HyperNetwork DNA...")
    for step in range(500):
        x, y = get_batch(text, seq_length, batch_size, char_to_ix)
        x, y = x.to(device), y.to(device)
        
        optimizer.zero_grad()
        logits = model(x)
        
        loss = criterion(logits.reshape(-1, vocab_size), y.reshape(-1))
        loss.backward()
        # Gradient clipping is vital for hypernets
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        if step % 50 == 0:
            perplexity = torch.exp(loss)
            print(f"Step {step} | Loss: {loss.item():.4f} | Perplexity: {perplexity.item():.4f}")

if __name__ == '__main__':
    main()
