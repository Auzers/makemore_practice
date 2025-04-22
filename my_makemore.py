import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random

# 数据加载和预处理部分
def load_names(file_path='names.txt'):
    """
    从文件中加载名字数据
    
    参数:
        file_path (str): 包含名字的文本文件路径
        
    返回:
        list: 名字列表
    """
    # your code here
    pass

def build_vocabulary(words):
    """
    构建字符词汇表和映射
    
    参数:
        words (list): 单词列表
        
    返回:
        tuple: (stoi, itos, vocab_size)
            - stoi (dict): 字符到整数的映射
            - itos (dict): 整数到字符的映射
            - vocab_size (int): 词汇表大小
    """
    # your code here
    pass

def build_dataset(words, block_size, stoi):
    """
    构建训练、验证和测试数据集
    
    参数:
        words (list): 单词列表
        block_size (int): 上下文长度
        stoi (dict): 字符到整数的映射
        
    返回:
        tuple: (X, Y) 其中:
            - X (torch.Tensor): 输入上下文，形状为 [N, block_size]
            - Y (torch.Tensor): 目标下一个字符，形状为 [N]
    """
    # your code here
    pass

def split_dataset(words, block_size, stoi, train_ratio=0.8, val_ratio=0.1):
    """
    将数据集分割为训练、验证和测试集
    
    参数:
        words (list): 单词列表
        block_size (int): 上下文长度
        stoi (dict): 字符到整数的映射
        train_ratio (float): 训练集比例
        val_ratio (float): 验证集比例
        
    返回:
        tuple: (Xtr, Ytr, Xdev, Ydev, Xte, Yte)
            - Xtr, Ytr: 训练集输入和目标
            - Xdev, Ydev: 验证集输入和目标
            - Xte, Yte: 测试集输入和目标
    """
    # your code here
    pass

# 模型层定义
class Linear:
    """
    线性层实现
    
    参数:
        fan_in (int): 输入特征维度
        fan_out (int): 输出特征维度
        bias (bool): 是否使用偏置项
    """
    
    def __init__(self, fan_in, fan_out, bias=True):
        """初始化线性层参数"""
        # your code here
        pass
    
    def __call__(self, x):
        """
        前向传播
        
        参数:
            x (torch.Tensor): 输入张量
            
        返回:
            torch.Tensor: 输出张量
        """
        # your code here
        pass
    
    def parameters(self):
        """
        返回层的可训练参数
        
        返回:
            list: 包含权重和偏置(如果有)的列表
        """
        # your code here
        pass

class BatchNorm1d:
    """
    一维批量归一化层
    
    参数:
        dim (int): 特征维度
        eps (float): 数值稳定性常数
        momentum (float): 运行平均值的动量参数
    """
    
    def __init__(self, dim, eps=1e-5, momentum=0.1):
        """初始化批量归一化层参数"""
        # your code here
        pass
    
    def __call__(self, x):
        """
        前向传播
        
        参数:
            x (torch.Tensor): 输入张量
            
        返回:
            torch.Tensor: 归一化后的输出张量
        """
        # your code here
        pass
    
    def parameters(self):
        """
        返回层的可训练参数
        
        返回:
            list: 包含gamma和beta参数的列表
        """
        # your code here
        pass

class Tanh:
    """
    双曲正切激活函数
    """
    
    def __call__(self, x):
        """
        前向传播
        
        参数:
            x (torch.Tensor): 输入张量
            
        返回:
            torch.Tensor: 应用tanh后的输出张量
        """
        # your code here
        pass
    
    def parameters(self):
        """
        返回层的可训练参数(无参数)
        
        返回:
            list: 空列表
        """
        # your code here
        pass

class Embedding:
    """
    嵌入层
    
    参数:
        num_embeddings (int): 嵌入向量的数量
        embedding_dim (int): 每个嵌入向量的维度
    """
    
    def __init__(self, num_embeddings, embedding_dim):
        """初始化嵌入层参数"""
        # your code here
        pass
    
    def __call__(self, IX):
        """
        前向传播
        
        参数:
            IX (torch.Tensor): 索引张量
            
        返回:
            torch.Tensor: 对应的嵌入向量
        """
        # your code here
        pass
    
    def parameters(self):
        """
        返回层的可训练参数
        
        返回:
            list: 包含嵌入权重的列表
        """
        # your code here
        pass

class FlattenConsecutive:
    """
    连续展平层，用于将连续的时间步骤展平
    
    参数:
        n (int): 要展平的连续时间步数
    """
    
    def __init__(self, n):
        """初始化连续展平层"""
        # your code here
        pass
    
    def __call__(self, x):
        """
        前向传播
        
        参数:
            x (torch.Tensor): 形状为[B, T, C]的输入张量
            
        返回:
            torch.Tensor: 展平后的输出张量，形状为[B, T//n, C*n]
        """
        # your code here
        pass
    
    def parameters(self):
        """
        返回层的可训练参数(无参数)
        
        返回:
            list: 空列表
        """
        # your code here
        pass

class Sequential:
    """
    顺序容器，按顺序包含多个层
    
    参数:
        layers (list): 层的列表
    """
    
    def __init__(self, layers):
        """初始化顺序容器"""
        # your code here
        pass
    
    def __call__(self, x):
        """
        前向传播
        
        参数:
            x (torch.Tensor): 输入张量
            
        返回:
            torch.Tensor: 通过所有层后的输出张量
        """
        # your code here
        pass
    
    def parameters(self):
        """
        返回所有层的可训练参数
        
        返回:
            list: 包含所有层参数的列表
        """
        # your code here
        pass

# 模型训练函数
def train_model(model, Xtr, Ytr, max_steps=200000, batch_size=32, learning_rate=0.1, lr_decay_step=150000, lr_decay_rate=0.1):
    """
    训练模型
    
    参数:
        model (Sequential): 要训练的模型
        Xtr (torch.Tensor): 训练输入数据
        Ytr (torch.Tensor): 训练目标数据
        max_steps (int): 最大训练步数
        batch_size (int): 批量大小
        learning_rate (float): 学习率
        lr_decay_step (int): 学习率衰减步数
        lr_decay_rate (float): 学习率衰减率
        
    返回:
        list: 训练过程中的损失值列表
    """
    # your code here
    pass

# 模型评估函数
def evaluate_model(model, split_data):
    """
    评估模型在给定数据集上的性能
    
    参数:
        model (Sequential): 要评估的模型
        split_data (dict): 包含'train', 'val', 'test'数据的字典
        
    返回:
        dict: 各数据集上的损失值
    """
    # your code here
    pass

# 生成样本函数
def generate_samples(model, itos, block_size, num_samples=10):
    """
    使用训练好的模型生成样本
    
    参数:
        model (Sequential): 训练好的模型
        itos (dict): 整数到字符的映射
        block_size (int): 上下文长度
        num_samples (int): 要生成的样本数量
        
    返回:
        list: 生成的样本列表
    """
    # your code here
    pass

# 主函数
def main():
    """
    主函数，运行完整的训练和生成过程
    """
    # 设置随机种子以确保可重复性
    torch.manual_seed(42)
    random.seed(42)
    
    # 加载数据
    words = load_names()
    print(f"加载了 {len(words)} 个名字")
    print(f"最长名字长度: {max(len(w) for w in words)}")
    print(f"示例: {words[:8]}")
    
    # 构建词汇表
    stoi, itos, vocab_size = build_vocabulary(words)
    print(f"词汇表大小: {vocab_size}")
    
    # 设置上下文长度
    block_size = 8
    
    # 打乱数据
    random.shuffle(words)
    
    # 分割数据集
    Xtr, Ytr, Xdev, Ydev, Xte, Yte = split_dataset(words, block_size, stoi)
    print(f"训练集大小: {len(Xtr)}")
    print(f"验证集大小: {len(Xdev)}")
    print(f"测试集大小: {len(Xte)}")
    
    # 构建分层网络模型
    n_embd = 24  # 字符嵌入向量的维度
    n_hidden = 128  # 隐藏层神经元数量
    
    model = Sequential([
        Embedding(vocab_size, n_embd),
        FlattenConsecutive(2), Linear(n_embd * 2, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
        FlattenConsecutive(2), Linear(n_hidden*2, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
        FlattenConsecutive(2), Linear(n_hidden*2, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
        Linear(n_hidden, vocab_size),
    ])
    
    # 参数初始化
    with torch.no_grad():
        model.layers[-1].weight *= 0.1  # 最后一层权重缩小，使输出更不确定
    
    parameters = model.parameters()
    print(f"模型参数总数: {sum(p.nelement() for p in parameters)}")
    
    # 训练模型
    lossi = train_model(model, Xtr, Ytr)
    
    # 绘制损失曲线
    plt.figure(figsize=(10, 5))
    plt.plot(torch.tensor(lossi).view(-1, 1000).mean(1))
    plt.title('训练损失')
    plt.xlabel('迭代次数 (x1000)')
    plt.ylabel('损失 (log10)')
    plt.savefig('training_loss.png')
    plt.close()
    
    # 将模型设置为评估模式
    for layer in model.layers:
        if hasattr(layer, 'training'):
            layer.training = False
    
    # 评估模型
    split_data = {
        'train': (Xtr, Ytr),
        'val': (Xdev, Ydev),
        'test': (Xte, Yte),
    }
    losses = evaluate_model(model, split_data)
    print(f"训练集损失: {losses['train']:.4f}")
    print(f"验证集损失: {losses['val']:.4f}")
    print(f"测试集损失: {losses['test']:.4f}")
    
    # 生成样本
    samples = generate_samples(model, itos, block_size, num_samples=20)
    print("\n生成的名字样本:")
    for sample in samples:
        print(sample)

if __name__ == "__main__":
    main()
