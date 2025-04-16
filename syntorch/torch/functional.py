"""
SynTorch 함수형 API 모듈

이 모듈은 텐서 연산을 위한 함수형 API를 제공합니다.
NumPy 연산을 SynTorch 텐서 연산으로 래핑하여 트레이싱을 가능하게 합니다.
"""

import numpy as np
from syntorch.core.trace import trace_manager
from syntorch.torch.tensor import Tensor


def tile(input, reps):
    """텐서를 반복 (NumPy의 tile에 해당)

    Args:
        input: 입력 텐서
        reps: 각 차원을 몇 번 반복할지 지정하는 정수 또는 시퀀스

    Returns:
        반복된 텐서
    """
    if not isinstance(input, Tensor):
        raise TypeError("입력은 Tensor 타입이어야 합니다")

    result_data = np.tile(input.data, reps)
    result = Tensor(result_data, input.dtype, input.device)

    # 트레이싱
    trace_manager.trace_tensor(
        "tile",
        [input],
        result,
        {"input_shape": input.shape, "output_shape": result.shape, "reps": reps},
    )

    return result


def reshape(input, shape):
    """텐서 모양 변경 (NumPy의 reshape에 해당)

    Args:
        input: 입력 텐서
        shape: 새로운 모양

    Returns:
        모양이 변경된 텐서
    """
    if not isinstance(input, Tensor):
        raise TypeError("입력은 Tensor 타입이어야 합니다")

    result_data = np.reshape(input.data, shape)
    result = Tensor(result_data, input.dtype, input.device)

    # 트레이싱
    trace_manager.trace_tensor(
        "reshape",
        [input],
        result,
        {"input_shape": input.shape, "output_shape": result.shape, "target_shape": shape},
    )

    return result


def transpose(input, axes=None):
    """텐서 전치 (NumPy의 transpose에 해당)

    Args:
        input: 입력 텐서
        axes: 차원 순서 (None이면 모든 차원 반전)

    Returns:
        전치된 텐서
    """
    if not isinstance(input, Tensor):
        raise TypeError("입력은 Tensor 타입이어야 합니다")

    result_data = np.transpose(input.data, axes)
    result = Tensor(result_data, input.dtype, input.device)

    # 트레이싱
    trace_manager.trace_tensor(
        "transpose",
        [input],
        result,
        {"input_shape": input.shape, "output_shape": result.shape, "axes": axes},
    )

    return result


def matmul(input, other):
    """행렬 곱셈 (NumPy의 matmul에 해당)

    Args:
        input: 첫 번째 입력 텐서
        other: 두 번째 입력 텐서

    Returns:
        행렬 곱셈 결과 텐서
    """
    if not isinstance(input, Tensor) or not isinstance(other, Tensor):
        raise TypeError("입력은 Tensor 타입이어야 합니다")

    result_data = np.matmul(input.data, other.data)
    result = Tensor(result_data, input.dtype, input.device)

    # 트레이싱
    trace_manager.trace_tensor(
        "matmul",
        [input, other],
        result,
        {"input_shape": input.shape, "other_shape": other.shape, "output_shape": result.shape},
    )

    return result


def zeros(shape, dtype=None, device=None):
    """0으로 채워진 텐서 생성 (NumPy의 zeros에 해당)

    Args:
        shape: 생성할 텐서의 모양
        dtype: 데이터 타입 (기본값: None)
        device: 장치 (기본값: None)

    Returns:
        0으로 채워진 텐서
    """
    result_data = np.zeros(shape)
    result = Tensor(result_data, dtype, device)

    # 트레이싱
    trace_manager.trace_tensor(
        "zeros", [], result, {"shape": shape, "dtype": dtype, "device": device}
    )

    return result


def ones(shape, dtype=None, device=None):
    """1로 채워진 텐서 생성 (NumPy의 ones에 해당)

    Args:
        shape: 생성할 텐서의 모양
        dtype: 데이터 타입 (기본값: None)
        device: 장치 (기본값: None)

    Returns:
        1로 채워진 텐서
    """
    result_data = np.ones(shape)
    result = Tensor(result_data, dtype, device)

    # 트레이싱
    trace_manager.trace_tensor(
        "ones", [], result, {"shape": shape, "dtype": dtype, "device": device}
    )

    return result


def max(input, axis=None, keepdims=False):
    """최대값 계산 (NumPy의 max에 해당)

    Args:
        input: 입력 텐서
        axis: 계산할 축 (기본값: None, 전체 텐서)
        keepdims: 차원 유지 여부 (기본값: False)

    Returns:
        최대값 텐서
    """
    if not isinstance(input, Tensor):
        raise TypeError("입력은 Tensor 타입이어야 합니다")

    result_data = np.max(input.data, axis=axis, keepdims=keepdims)
    if not isinstance(result_data, np.ndarray):
        result_data = np.array(result_data)

    result = Tensor(result_data, input.dtype, input.device)

    # 트레이싱
    trace_manager.trace_tensor(
        "max",
        [input],
        result,
        {
            "input_shape": input.shape,
            "output_shape": result.shape,
            "axis": axis,
            "keepdims": keepdims,
        },
    )

    return result


def exp(input):
    """지수 함수 (NumPy의 exp에 해당)

    Args:
        input: 입력 텐서

    Returns:
        지수 함수 적용된 텐서
    """
    if not isinstance(input, Tensor):
        raise TypeError("입력은 Tensor 타입이어야 합니다")

    result_data = np.exp(input.data)
    result = Tensor(result_data, input.dtype, input.device)

    # 트레이싱
    trace_manager.trace_tensor(
        "exp", [input], result, {"input_shape": input.shape, "output_shape": result.shape}
    )

    return result


def sum(input, axis=None, keepdims=False):
    """합계 계산 (NumPy의 sum에 해당)

    Args:
        input: 입력 텐서
        axis: 계산할 축 (기본값: None, 전체 텐서)
        keepdims: 차원 유지 여부 (기본값: False)

    Returns:
        합계 텐서
    """
    if not isinstance(input, Tensor):
        raise TypeError("입력은 Tensor 타입이어야 합니다")

    result_data = np.sum(input.data, axis=axis, keepdims=keepdims)
    if not isinstance(result_data, np.ndarray):
        result_data = np.array(result_data)

    result = Tensor(result_data, input.dtype, input.device)

    # 트레이싱
    trace_manager.trace_tensor(
        "sum",
        [input],
        result,
        {
            "input_shape": input.shape,
            "output_shape": result.shape,
            "axis": axis,
            "keepdims": keepdims,
        },
    )

    return result
