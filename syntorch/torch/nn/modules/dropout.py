from syntorch.torch.module import Module
import syntorch.torch.nn.functional as F

class Dropout(Module):
    """드롭아웃 레이어 구현"""
    
    def __init__(self, p=0.5, inplace=False):
        super().__init__()
        self.p = p
        self.inplace = inplace  # 현재 구현에서는 무시됨
    
    def forward(self, input):
        """순전파"""
        return F.dropout(input, self.p, self.training)
    
    def extra_repr(self):
        """추가 표현 문자열"""
        return f'p={self.p}, inplace={self.inplace}' 