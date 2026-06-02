import torch
import torch.nn as nn
import numpy as np
import os
import urllib.request
import math

class KrylovReservoir(nn.Module):
    def __init__(self, vocab_size, hidden_size, degree=50, spectral_radius=0.95):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.degree = degree
        
        # 1. PRNG Embeddings (Fixed)
        torch.manual_seed(42)
        E = torch.randn(vocab_size, hidden_size) * 0.1
        self.register_buffer('E', E)
        
        # 2. PRNG Reservoir Matrix (Fixed)
        W_res = torch.randn(hidden_size, hidden_size)
        # Scale by spectral radius
        eigenvalues = torch.linalg.eigvals(W_res)
        max_eig = torch.max(torch.abs(eigenvalues))
        W_res = W_res * (spectral_radius / max_eig)
        self.register_buffer('W_res', W_res)
        
        # 3. Trainable DNA: The 50 polynomial coefficients!
        self.alpha = nn.Parameter(torch.zeros(degree))
        nn.init.normal_(self.alpha, mean=0.0, std=0.02)
        
        # Precompute the Krylov basis for the embeddings to save time during training
        # We want to compute: E_c^T W_res^k
        # Shape: (degree, vocab_size, hidden_size)
        print("Precomputing Krylov Subspace basis for the vocabulary...")
        basis = []
        curr_E = self.E  # (vocab_size, hidden_size)
        for k in range(degree):
            basis.append(curr_E)
            # Multiply E * W_res (which is E^T W_res^T conceptually)
            curr_E = torch.matmul(curr_E, self.W_res.t())
        
        self.register_buffer('krylov_basis', torch.stack(basis, dim=0)) # (degree, vocab_size, hidden_size)

        num_params = sum(p.numel() for p in self.parameters())
        print(f"--- KRYLOV RESERVOIR INITIALIZED ---")
        print(f"Stored DNA parameters: {num_params} (Exactly {num_params * 4} bytes)")
        
    def forward(self, x, hidden=None):
        batch, seq = x.shape
        if hidden is None:
            hidden = torch.zeros(batch, self.hidden_size, device=x.device)
            
        out = []
        for t in range(seq):
            # 1. Reservoir State Update
            xt = x[:, t]
            in_vec = self.E[xt] # (batch, hidden_size)
            hidden = torch.tanh(in_vec + torch.matmul(hidden, self.W_res.t()))
            
            # 2. Compute 50 Krylov Features for all candidate tokens
            # krylov_basis is (degree, vocab, hidden)
            # hidden is (batch, hidden)
            # We want dot products: features = dot(krylov_basis, hidden) 
            # => (degree, vocab, batch)
            # Reshape for bmm: basis (degree, vocab, hidden), hidden (batch, hidden, 1) -> wait, einsum is cleaner
            # f_{k, v, b} = sum_h basis_{k, v, h} * hidden_{b, h}
            features = torch.einsum('kvh,bh->bvk', self.krylov_basis, hidden) # (batch, vocab, degree)
            
            # 3. Multiply by trainable alpha (degree) to get logits (batch, vocab)
            logits = torch.einsum('bvk,k->bv', features, self.alpha)
            out.append(logits)
            
        return torch.stack(out, dim=1), hidden

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
    
    # 2048 hidden size, 128 degree polynomial
    model = KrylovReservoir(vocab_size, hidden_size=2048, degree=128).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    batch_size = 128
    seq_length = 32
    
    print("Training the DNA...")
    for step in range(500):
        x, y = get_batch(text, seq_length, batch_size, char_to_ix)
        x, y = x.to(device), y.to(device)
        
        optimizer.zero_grad()
        logits, _ = model(x)
        
        loss = criterion(logits.reshape(-1, vocab_size), y.reshape(-1))
        loss.backward()
        optimizer.step()
        
        if step % 50 == 0:
            perplexity = torch.exp(loss)
            print(f"Step {step} | Loss: {loss.item():.4f} | Perplexity: {perplexity.item():.4f}")

if __name__ == '__main__':
    main()
