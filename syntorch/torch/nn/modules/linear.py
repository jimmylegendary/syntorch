import numpy as np
from syntorch.torch.module import Module, Parameter
from syntorch.torch.tensor import Tensor
import syntorch.torch.nn.functional as F

class Linear(Module):
    """선형 레이어 구현"""
    
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # 가중치 초기화
        weight_data = np.random.randn(out_features, in_features) * 0.02
        self.weight = Parameter(weight_data)
        
        if bias:
            bias_data = np.zeros(out_features)
            self.bias = Parameter(bias_data)
        else:
            self.bias = None
    
    def forward(self, input):
        """순전파"""
        return F.linear(input, self.weight, self.bias)
    
    def extra_repr(self):
        """추가 표현 문자열"""
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}' 