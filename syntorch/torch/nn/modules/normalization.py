import numpy as np
from syntorch.torch.module import Module, Parameter
from syntorch.torch.tensor import Tensor
import syntorch.torch.nn.functional as F

class LayerNorm(Module):
    """레이어 정규화 구현"""
    
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        
        if elementwise_affine:
            weight_data = np.ones(normalized_shape)
            self.weight = Parameter(weight_data)
            bias_data = np.zeros(normalized_shape)
            self.bias = Parameter(bias_data)
        else:
            self.weight = None
            self.bias = None
    
    def forward(self, input):
        """순전파"""
        return F.layer_norm(
            input, 
            self.normalized_shape, 
            self.weight, 
            self.bias, 
            self.eps
        )
    
    def extra_repr(self):
        """추가 표현 문자열"""
        return f'normalized_shape={self.normalized_shape}, eps={self.eps}, elementwise_affine={self.elementwise_affine}' 