import numpy as np
from syntorch.torch.module import Module, Parameter
from syntorch.torch.tensor import Tensor
import syntorch.torch.nn.functional as F

class Embedding(Module):
    """임베딩 레이어 구현"""
    
    def __init__(self, num_embeddings, embedding_dim, padding_idx=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        
        # 가중치 초기화
        weight_data = np.random.randn(num_embeddings, embedding_dim) * 0.02
        
        # 패딩 인덱스 처리
        if padding_idx is not None:
            weight_data[padding_idx] = 0
        
        self.weight = Parameter(weight_data)
    
    def forward(self, input):
        """순전파"""
        return F.embedding(input, self.weight)
    
    def extra_repr(self):
        """추가 표현 문자열"""
        return f'num_embeddings={self.num_embeddings}, embedding_dim={self.embedding_dim}, padding_idx={self.padding_idx}' 