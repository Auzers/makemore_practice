"""
my_makemore_template.py - Character-level language model implementation template

This file is a template created based on makemore.ipynb, preserving the code structure
and function interfaces, but leaving the main implementation for the user to complete.
Places where code needs to be filled in are marked with "your code here" comments.
"""

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
    # your code here
    pass

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
    # your code here
    pass

def build_dataset(words, block_size, stoi):
    """
    Build training, validation, and test datasets
    
    Args:
        words (list): List of words
        block_size (int): Context length
        stoi (dict): Character to integer mapping
        
    Returns:
        tuple: (X, Y) where:
            - X (torch.Tensor): Input contexts, shape [N, block_size]
            - Y (torch.Tensor): Target next characters, shape [N]
    """
    # your code here
    pass

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
    # your code here
    pass

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
        # your code here
        pass
    
    def __call__(self, x):
        """
        Forward pass
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Output tensor
        """
        # your code here
        pass
    
    def parameters(self):
        """
        Return trainable parameters of the layer
        
        Returns:
            list: List containing weights and bias (if present)
        """
        # your code here
        pass

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
        # your code here
        pass
    
    def __call__(self, x):
        """
        Forward pass
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Normalized output tensor
        """
        # your code here
        pass
    
    def parameters(self):
        """
        Return trainable parameters of the layer
        
        Returns:
            list: List containing gamma and beta parameters
        """
        # your code here
        pass

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
        # your code here
        pass
    
    def parameters(self):
        """
        Return trainable parameters of the layer (none)
        
        Returns:
            list: Empty list
        """
        # your code here
        pass

class Embedding:
    """
    Embedding layer
    
    Args:
        num_embeddings (int): Number of embedding vectors
        embedding_dim (int): Dimension of each embedding vector
    """
    
    def __init__(self, num_embeddings, embedding_dim):
        """Initialize embedding layer parameters"""
        # your code here
        pass
    
    def __call__(self, IX):
        """
        Forward pass
        
        Args:
            IX (torch.Tensor): Index tensor
            
        Returns:
            torch.Tensor: Corresponding embedding vectors
        """
        # your code here
        pass
    
    def parameters(self):
        """
        Return trainable parameters of the layer
        
        Returns:
            list: List containing embedding weights
        """
        # your code here
        pass

class FlattenConsecutive:
    """
    Flatten consecutive layer, used to flatten consecutive time steps
    
    Args:
        n (int): Number of consecutive time steps to flatten
    """
    
    def __init__(self, n):
        """Initialize flatten consecutive layer"""
        # your code here
        pass
    
    def __call__(self, x):
        """
        Forward pass
        
        Args:
            x (torch.Tensor): Input tensor of shape [B, T, C]
            
        Returns:
            torch.Tensor: Flattened output tensor of shape [B, T//n, C*n]
        """
        # your code here
        pass
    
    def parameters(self):
        """
        Return trainable parameters of the layer (none)
        
        Returns:
            list: Empty list
        """
        # your code here
        pass

class Sequential:
    """
    Sequential container, holds a list of layers applied in sequence
    
    Args:
        layers (list): List of layers
    """
    
    def __init__(self, layers):
        """Initialize sequential container"""
        # your code here
        pass
    
    def __call__(self, x):
        """
        Forward pass
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Output tensor after passing through all layers
        """
        # your code here
        pass
    
    def parameters(self):
        """
        Return trainable parameters of all layers
        
        Returns:
            list: List containing all layer parameters
        """
        # your code here
        pass

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
    # your code here
    pass

# Model evaluation functions
def evaluate_model(model, split_data):
    """
    Evaluate model performance on given datasets
    
    Args:
        model (Sequential): Model to evaluate
        split_data (dict): Dictionary containing 'train', 'val', 'test' data
        
    Returns:
        dict: Loss values for each dataset
    """
    # your code here
    pass

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
    # your code here
    pass

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
