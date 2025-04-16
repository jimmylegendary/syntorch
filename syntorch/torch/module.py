from syntorch.core.trace import SyntorchLayer
from syntorch.torch.tensor import Tensor
import inspect


class Parameter(Tensor):
    """모듈 파라미터 클래스"""

    def __init__(self, data, requires_grad=True):
        super().__init__(data, requires_grad=requires_grad)


class Module(SyntorchLayer):
    """PyTorch nn.Module과 호환되는 기본 모듈 클래스"""

    def __init__(self):
        self._modules = {}
        self._parameters = {}
        self._buffers = {}
        self.training = True

    def __setattr__(self, name, value):
        """속성 설정 시 자동으로 파라미터, 모듈, 버퍼 등록"""
        if isinstance(value, Parameter):
            self._parameters[name] = value
        elif isinstance(value, Module):
            self._modules[name] = value
        elif isinstance(value, Tensor) and not name.startswith("_"):
            self._buffers[name] = value

        object.__setattr__(self, name, value)

    def __getattr__(self, name):
        """속성 접근 시 파라미터, 모듈, 버퍼 확인"""
        if "_parameters" in self.__dict__ and name in self._parameters:
            return self._parameters[name]
        if "_modules" in self.__dict__ and name in self._modules:
            return self._modules[name]
        if "_buffers" in self.__dict__ and name in self._buffers:
            return self._buffers[name]

        raise AttributeError(f"'{self.__class__.__name__}' 객체에 '{name}' 속성이 없습니다")

    def add_module(self, name, module):
        """하위 모듈 추가"""
        if not isinstance(module, Module) and module is not None:
            raise TypeError(f"{type(module)}는 모듈 타입이 아닙니다")
        self._modules[name] = module

    def parameters(self):
        """모듈의 모든 파라미터 반환"""
        for name, param in self.named_parameters():
            yield param

    def named_parameters(self, prefix=""):
        """이름이 있는 모든 파라미터 반환"""
        for name, param in self._parameters.items():
            yield prefix + ("." if prefix else "") + name, param

        for mname, module in self._modules.items():
            submodule_prefix = prefix + ("." if prefix else "") + mname
            for name, param in module.named_parameters(submodule_prefix):
                yield name, param

    def modules(self):
        """모든 하위 모듈 반환"""
        for name, module in self.named_modules():
            yield module

    def named_modules(self, prefix=""):
        """이름이 있는 모든 모듈 반환"""
        yield prefix, self

        for name, module in self._modules.items():
            submodule_prefix = prefix + ("." if prefix else "") + name
            for m_name, m in module.named_modules(submodule_prefix):
                yield m_name, m

    def eval(self):
        """평가 모드로 전환"""
        self.training = False
        for module in self.modules():
            module.training = False
        return self

    def train(self, mode=True):
        """학습 모드로 전환"""
        self.training = mode
        for module in self.modules():
            module.training = mode
        return self

    def to(self, device=None, dtype=None):
        """모든 파라미터와 버퍼를 지정된 디바이스/데이터 타입으로 변환"""
        for param in self.parameters():
            param.to(device=device, dtype=dtype)

        for buffer_name, buffer in self._buffers.items():
            self._buffers[buffer_name] = buffer.to(device=device, dtype=dtype)

        return self

    def forward(self, *args, **kwargs):
        """순전파 (서브클래스에서 구현해야 함)"""
        raise NotImplementedError("Module 서브클래스는 forward 메서드를 구현해야 합니다")

    def __call__(self, *args, **kwargs):
        """모듈 호출 시 forward 메서드 실행"""
        return self.forward(*args, **kwargs)

    def state_dict(self):
        """모듈의 상태를 딕셔너리로 반환"""
        state = {}
        for name, param in self._parameters.items():
            state[name] = param.data
        for name, buffer in self._buffers.items():
            state[name] = buffer.data
        for name, module in self._modules.items():
            state[name] = module.state_dict()
        return state

    def load_state_dict(self, state_dict):
        """모듈 상태 로드"""
        for name, param in self._parameters.items():
            if name in state_dict:
                param.data = state_dict[name]

        for name, buffer in self._buffers.items():
            if name in state_dict:
                self._buffers[name] = Tensor(state_dict[name])

        for name, module in self._modules.items():
            if name in state_dict:
                module.load_state_dict(state_dict[name])

    def apply(self, fn):
        """모든 하위 모듈에 함수 적용"""
        for module in self.modules():
            fn(module)
        return self
