from .tensor import Tensor, tensor, zeros, ones, randn, arange, cat, sort, topk, cumsum, randint
from .module import Module, Parameter
import syntorch.torch.nn as nn
import syntorch.torch.nn.functional as F


# 데이터 타입 정의
float32 = "float32"  # 기본 부동소수점 타입
float = float32  # float32의 별칭
double = "float64"  # 두 배 정밀도 부동소수점
half = "float16"  # 반 정밀도 부동소수점

int32 = "int32"  # 32비트 정수
int = int32  # int32의 별칭
long = "int64"  # 64비트 정수
short = "int16"  # 16비트 정수
int8 = "int8"  # 8비트 정수

uint8 = "uint8"  # 부호 없는 8비트 정수
bool = "bool"  # 불리언 타입


# CUDA 기능 추가
class _CudaModule:
    def is_available(self):
        """CUDA 사용 가능 여부 확인 (syntorch는 기본적으로 CPU만 지원)"""
        return False


cuda = _CudaModule()


# no_grad 컨텍스트 매니저
class no_grad:
    """그래디언트 계산을 비활성화하는 컨텍스트 매니저"""

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


# 함수 추가
def softmax(input, dim=-1):
    """소프트맥스 함수"""
    return input.softmax(dim)


# 하위 모듈 내보내기
__all__ = [
    "Tensor",
    "tensor",
    "zeros",
    "ones",
    "randn",
    "randint",
    "arange",
    "cat",
    "sort",
    "topk",
    "cumsum",
    "softmax",
    "Module",
    "Parameter",
    "nn",
    "functional",
    "cuda",
    "no_grad",
    "long",
    "float",
    "float32",
    "double",
    "half",
    "int",
    "int32",
    "short",
    "int8",
    "uint8",
    "bool",
]

# 함수형 모듈 참조
functional = F
