"""
syntorch: LLM 추론을 위한 합성 트레이스 생성 프레임워크

이 패키지는 PyTorch와 호환되는 인터페이스를 제공하면서
하드웨어 레이어까지의 합성 트레이스를 생성할 수 있는 기능을 제공합니다.
"""

# 버전 정보
__version__ = '0.1.0'

# 주요 모듈 가져오기
from syntorch.core.trace import trace_manager

# PyTorch 호환 모듈
import syntorch.torch as torch

# 병렬 처리 모듈
import syntorch.parallel as parallel

# 하위 레이어
import syntorch.runtime as runtime
import syntorch.os as os
import syntorch.hw as hw

__all__ = [
    'trace_manager',
    'torch',
    'parallel',
    'runtime',
    'os',
    'hw'
] 