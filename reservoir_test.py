import numpy as np
import urllib.request
import os

try:
    from sklearn.linear_model import SGDClassifier
    from sklearn.metrics import log_loss
except ImportError:
    import sys
    print("scikit-learn not installed. Please run: pip install scikit-learn")
    sys.exit(1)

def main():
    if not os.path.exists("tinyshakespeare.txt"):
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        print("Downloading Tiny Shakespeare...")
        urllib.request.urlretrieve(url, "tinyshakespeare.txt")

    with open("tinyshakespeare.txt", "r") as f:
        text = f.read()

    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    char_to_ix = {ch: i for i, ch in enumerate(chars)}
    ix_to_char = {i: ch for i, ch in enumerate(chars)}

    hidden_size = 2048
    spectral_radius = 0.95
    np.random.seed(42)

    print(f"Vocab size: {vocab_size}, Reservoir size: {hidden_size}")
    print(f"Generating Reservoir Matrix...")

    W_res = np.random.randn(hidden_size, hidden_size)
    # Scale to desired spectral radius
    eigenvalues = np.linalg.eigvals(W_res)
    max_eigenvalue = np.max(np.abs(eigenvalues))
    W_res = W_res * (spectral_radius / max_eigenvalue)
    
    # Cast to float32 for speed
    W_res = W_res.astype(np.float32)

    # PRNG Embeddings
    W_in = (np.random.randn(hidden_size, vocab_size) * 0.1).astype(np.float32)

    print("Collecting states...")
    train_len = 50000 # Reduced for fast iteration
    X_states = np.zeros((train_len, hidden_size), dtype=np.float32)
    Y_targets = np.zeros((train_len,), dtype=np.int32)

    state = np.zeros((hidden_size,), dtype=np.float32)

    for t in range(train_len):
        char_id = char_to_ix[text[t]]
        in_vec = W_in[:, char_id]
        state = np.tanh(in_vec + np.dot(W_res, state))
        
        X_states[t] = state
        if t < train_len - 1:
            Y_targets[t] = char_to_ix[text[t+1]]

    # Discard warmup
    warmup = 1000
    X_train = X_states[warmup:train_len-1]
    Y_train = Y_targets[warmup:train_len-1]

    print("Training linear readout via SGD...")
    clf = SGDClassifier(loss='log_loss', penalty='l2', alpha=1e-4, max_iter=20, random_state=42, n_jobs=-1)
    
    # Needs classes array for log_loss compatibility
    classes = np.arange(vocab_size)
    clf.partial_fit(X_train, Y_train, classes=classes)
    
    # We can do a few more iterations manually
    for _ in range(10):
        clf.partial_fit(X_train, Y_train)

    probs = clf.predict_proba(X_train)
    loss = log_loss(Y_train, probs, labels=classes)
    perplexity = np.exp(loss)
    print(f"Train Perplexity: {perplexity:.4f}")

    # Test set
    test_len = 10000
    X_test = np.zeros((test_len, hidden_size), dtype=np.float32)
    Y_test = np.zeros((test_len,), dtype=np.int32)

    print("Evaluating on test set...")
    for t in range(train_len, train_len + test_len):
        char_id = char_to_ix[text[t]]
        in_vec = W_in[:, char_id]
        state = np.tanh(in_vec + np.dot(W_res, state))
        X_test[t - train_len] = state
        if t < train_len + test_len - 1:
            Y_test[t - train_len] = char_to_ix[text[t+1]]

    X_test = X_test[:-1]
    Y_test = Y_test[:-1]

    test_probs = clf.predict_proba(X_test)
    test_loss = log_loss(Y_test, test_probs, labels=classes)
    test_perplexity = np.exp(test_loss)
    print(f"Test Perplexity: {test_perplexity:.4f}")

    # Baseline perplexity (random guessing)
    print(f"Random Guessing Perplexity: {vocab_size:.4f}")

if __name__ == "__main__":
    main()
