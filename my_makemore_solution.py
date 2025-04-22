import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random

# Data loading and preprocessing
def load_names(file_path='names.txt'):
    """
    Load name data from file
    
    Args:
        file_path (str): Path to text file containing names
        
    Returns:
        list: List of names
    """
    with open(file_path, 'r') as f:
        return f.read().splitlines()

def build_vocabulary(words):
    """
    Build character vocabulary and mappings
    
    Args:
        words (list): List of words
        
    Returns:
        tuple: (stoi, itos, vocab_size)
            - stoi (dict): Character to integer mapping
            - itos (dict): Integer to character mapping
            - vocab_size (int): Size of vocabulary
    """
    chars = sorted(list(set(''.join(words))))
    stoi = {s: i+1 for i, s in enumerate(chars)}
    stoi['.'] = 0  # Add special end-of-word token
    itos = {i: s for s, i in stoi.items()}
    vocab_size = len(itos)
    return stoi, itos, vocab_size

def build_dataset(words, block_size, stoi):
    """
    Build dataset from words
    
    Args:
        words (list): List of words
        block_size (int): Context length
        stoi (dict): Character to integer mapping
        
    Returns:
        tuple: (X, Y) where:
            - X (torch.Tensor): Input contexts, shape [N, block_size]
            - Y (torch.Tensor): Target next characters, shape [N]
    """
    X, Y = [], []
    
    for w in words:
        context = [0] * block_size
        for ch in w + '.':
            ix = stoi[ch]
            X.append(context)
            Y.append(ix)
            context = context[1:] + [ix]  # crop and append
    
    X = torch.tensor(X)
    Y = torch.tensor(Y)
    return X, Y

def split_dataset(words, block_size, stoi, train_ratio=0.8, val_ratio=0.1):
    """
    Split dataset into training, validation, and test sets
    
    Args:
        words (list): List of words
        block_size (int): Context length
        stoi (dict): Character to integer mapping
        train_ratio (float): Proportion for training set
        val_ratio (float): Proportion for validation set
        
    Returns:
        tuple: (Xtr, Ytr, Xdev, Ydev, Xte, Yte)
            - Xtr, Ytr: Training inputs and targets
            - Xdev, Ydev: Validation inputs and targets
            - Xte, Yte: Test inputs and targets
    """
    n1 = int(train_ratio * len(words))
    n2 = int((train_ratio + val_ratio) * len(words))
    
    Xtr, Ytr = build_dataset(words[:n1], block_size, stoi)
    Xdev, Ydev = build_dataset(words[n1:n2], block_size, stoi)
    Xte, Yte = build_dataset(words[n2:], block_size, stoi)
    
    return Xtr, Ytr, Xdev, Ydev, Xte, Yte

# Model layer definitions
class Linear:
    """
    Linear layer implementation
    
    Args:
        fan_in (int): Input feature dimension
        fan_out (int): Output feature dimension
        bias (bool): Whether to use bias
    """
    
    def __init__(self, fan_in, fan_out, bias=True):
        """Initialize linear layer parameters"""
        self.weight = torch.randn((fan_in, fan_out)) / fan_in**0.5  # Kaiming initialization
        self.bias = torch.zeros(fan_out) if bias else None
    
    def __call__(self, x):
        """
        Forward pass
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Output tensor
        """
        self.out = x @ self.weight
        if self.bias is not None:
            self.out += self.bias
        return self.out
    
    def parameters(self):
        """
        Return trainable parameters of the layer
        
        Returns:
            list: List containing weights and bias (if present)
        """
        return [self.weight] + ([] if self.bias is None else [self.bias])

class BatchNorm1d:
    """
    1D Batch Normalization layer
    
    Args:
        dim (int): Feature dimension
        eps (float): Numerical stability constant
        momentum (float): Momentum parameter for running averages
    """
    
    def __init__(self, dim, eps=1e-5, momentum=0.1):
        """Initialize batch normalization layer parameters"""
        self.eps = eps
        self.momentum = momentum
        self.training = True
        # Parameters (trained with backprop)
        self.gamma = torch.ones(dim)
        self.beta = torch.zeros(dim)
        # Buffers (trained with running momentum update)
        self.running_mean = torch.zeros(dim)
        self.running_var = torch.ones(dim)
    
    def __call__(self, x):
        """
        Forward pass
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Normalized output tensor
        """
        # Calculate forward pass
        if self.training:
            if x.ndim == 2:
                dim = 0
            elif x.ndim == 3:
                dim = (0, 1)
            xmean = x.mean(dim, keepdim=True)  # Batch mean
            xvar = x.var(dim, keepdim=True)  # Batch variance
        else:
            xmean = self.running_mean
            xvar = self.running_var
        
        xhat = (x - xmean) / torch.sqrt(xvar + self.eps)  # Normalize to unit variance
        self.out = self.gamma * xhat + self.beta
        
        # Update buffers
        if self.training:
            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * xmean
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * xvar
        
        return self.out
    
    def parameters(self):
        """
        Return trainable parameters of the layer
        
        Returns:
            list: List containing gamma and beta parameters
        """
        return [self.gamma, self.beta]

class Tanh:
    """
    Hyperbolic tangent activation function
    """
    
    def __call__(self, x):
        """
        Forward pass
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Output tensor after applying tanh
        """
        self.out = torch.tanh(x)
        return self.out
    
    def parameters(self):
        """
        Return trainable parameters of the layer (none)
        
        Returns:
            list: Empty list
        """
        return []

class Embedding:
    """
    Embedding layer
    
    Args:
        num_embeddings (int): Number of embedding vectors
        embedding_dim (int): Dimension of each embedding vector
    """
    
    def __init__(self, num_embeddings, embedding_dim):
        """Initialize embedding layer parameters"""
        self.weight = torch.randn((num_embeddings, embedding_dim))
    
    def __call__(self, IX):
        """
        Forward pass
        
        Args:
            IX (torch.Tensor): Index tensor
            
        Returns:
            torch.Tensor: Corresponding embedding vectors
        """
        self.out = self.weight[IX]
        return self.out
    
    def parameters(self):
        """
        Return trainable parameters of the layer
        
        Returns:
            list: List containing embedding weights
        """
        return [self.weight]

class FlattenConsecutive:
    """
    Flatten consecutive layer, used to flatten consecutive time steps
    
    Args:
        n (int): Number of consecutive time steps to flatten
    """
    
    def __init__(self, n):
        """Initialize flatten consecutive layer"""
        self.n = n
    
    def __call__(self, x):
        """
        Forward pass
        
        Args:
            x (torch.Tensor): Input tensor of shape [B, T, C]
            
        Returns:
            torch.Tensor: Flattened output tensor of shape [B, T//n, C*n]
        """
        B, T, C = x.shape
        x = x.view(B, T//self.n, C*self.n)
        if x.shape[1] == 1:
            x = x.squeeze(1)
        self.out = x
        return self.out
    
    def parameters(self):
        """
        Return trainable parameters of the layer (none)
        
        Returns:
            list: Empty list
        """
        return []

class Sequential:
    """
    Sequential container, holds a list of layers applied in sequence
    
    Args:
        layers (list): List of layers
    """
    
    def __init__(self, layers):
        """Initialize sequential container"""
        self.layers = layers
    
    def __call__(self, x):
        """
        Forward pass
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Output tensor after passing through all layers
        """
        for layer in self.layers:
            x = layer(x)
        self.out = x
        return self.out
    
    def parameters(self):
        """
        Return trainable parameters of all layers
        
        Returns:
            list: List containing all layer parameters
        """
        return [p for layer in self.layers for p in layer.parameters()]

# Model training functions
def train_model(model, Xtr, Ytr, max_steps=200000, batch_size=32, learning_rate=0.1, lr_decay_step=150000, lr_decay_rate=0.1):
    """
    Train the model
    
    Args:
        model (Sequential): Model to train
        Xtr (torch.Tensor): Training input data
        Ytr (torch.Tensor): Training target data
        max_steps (int): Maximum training steps
        batch_size (int): Batch size
        learning_rate (float): Learning rate
        lr_decay_step (int): Step at which to decay learning rate
        lr_decay_rate (float): Learning rate decay factor
        
    Returns:
        list: List of loss values during training
    """
    parameters = model.parameters()
    for p in parameters:
        p.requires_grad = True
    
    lossi = []
    
    for i in range(max_steps):
        # Minibatch construction
        ix = torch.randint(0, Xtr.shape[0], (batch_size,))
        Xb, Yb = Xtr[ix], Ytr[ix]  # Batch X, Y
        
        # Forward pass
        logits = model(Xb)
        loss = F.cross_entropy(logits, Yb)  # Loss function
        
        # Backward pass
        for p in parameters:
            p.grad = None
        loss.backward()
        
        # Update: simple SGD
        lr = learning_rate if i < lr_decay_step else learning_rate * lr_decay_rate
        for p in parameters:
            p.data += -lr * p.grad
        
        # Track stats
        if i % 10000 == 0:
            print(f'{i:7d}/{max_steps:7d}: {loss.item():.4f}')
        lossi.append(loss.log10().item())
    
    return lossi

# Model evaluation functions
@torch.no_grad()
def evaluate_model(model, split_data):
    """
    Evaluate model performance on given datasets
    
    Args:
        model (Sequential): Model to evaluate
        split_data (dict): Dictionary containing 'train', 'val', 'test' data
        
    Returns:
        dict: Loss values for each dataset
    """
    results = {}
    
    for split, (X, Y) in split_data.items():
        logits = model(X)
        loss = F.cross_entropy(logits, Y)
        results[split] = loss.item()
    
    return results

# Sample generation function
def generate_samples(model, itos, block_size, num_samples=10):
    """
    Generate samples using the trained model
    
    Args:
        model (Sequential): Trained model
        itos (dict): Integer to character mapping
        block_size (int): Context length
        num_samples (int): Number of samples to generate
        
    Returns:
        list: List of generated samples
    """
    samples = []
    
    for _ in range(num_samples):
        out = []
        context = [0] * block_size  # Initialize with all '.'
        
        while True:
            # Forward pass the neural net
            logits = model(torch.tensor([context]))
            probs = F.softmax(logits, dim=1)
            # Sample from the distribution
            ix = torch.multinomial(probs, num_samples=1).item()
            # Shift the context window and track the samples
            context = context[1:] + [ix]
            out.append(ix)
            # If we sample the special '.' token, break
            if ix == 0:
                break
        
        samples.append(''.join(itos[i] for i in out))
    
    return samples

# Main function
def main():
    """
    Main function to run the complete training and generation process
    """
    # Set random seed for reproducibility
    torch.manual_seed(42)
    random.seed(42)
    
    # Load data
    words = load_names()
    print(f"Loaded {len(words)} names")
    print(f"Max name length: {max(len(w) for w in words)}")
    print(f"Examples: {words[:8]}")
    
    # Build vocabulary
    stoi, itos, vocab_size = build_vocabulary(words)
    print(f"Vocabulary size: {vocab_size}")
    
    # Set context length
    block_size = 8
    
    # Shuffle data
    random.shuffle(words)
    
    # Split dataset
    Xtr, Ytr, Xdev, Ydev, Xte, Yte = split_dataset(words, block_size, stoi)
    print(f"Training set size: {len(Xtr)}")
    print(f"Validation set size: {len(Xdev)}")
    print(f"Test set size: {len(Xte)}")
    
    # Build hierarchical network model
    n_embd = 24  # dimension of character embedding vectors
    n_hidden = 128  # number of neurons in hidden layer
    
    model = Sequential([
        Embedding(vocab_size, n_embd),
        FlattenConsecutive(2), Linear(n_embd * 2, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
        FlattenConsecutive(2), Linear(n_hidden*2, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
        FlattenConsecutive(2), Linear(n_hidden*2, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
        Linear(n_hidden, vocab_size),
    ])
    
    # Parameter initialization
    with torch.no_grad():
        model.layers[-1].weight *= 0.1  # make last layer less confident
    
    parameters = model.parameters()
    print(f"Total parameters: {sum(p.nelement() for p in parameters)}")
    
    # Train model
    lossi = train_model(model, Xtr, Ytr)
    
    # Plot loss curve
    plt.figure(figsize=(10, 5))
    plt.plot(torch.tensor(lossi).view(-1, 1000).mean(1))
    plt.title('Training Loss')
    plt.xlabel('Iterations (x1000)')
    plt.ylabel('Loss (log10)')
    plt.savefig('training_loss.png')
    plt.close()
    
    # Set model to evaluation mode
    for layer in model.layers:
        if hasattr(layer, 'training'):
            layer.training = False
    
    # Evaluate model
    split_data = {
        'train': (Xtr, Ytr),
        'val': (Xdev, Ydev),
        'test': (Xte, Yte),
    }
    losses = evaluate_model(model, split_data)
    print(f"Train loss: {losses['train']:.4f}")
    print(f"Validation loss: {losses['val']:.4f}")
    print(f"Test loss: {losses['test']:.4f}")
    
    # Generate samples
    samples = generate_samples(model, itos, block_size, num_samples=20)
    print("\nGenerated name samples:")
    for sample in samples:
        print(sample)

if __name__ == "__main__":
    main()
