import numpy as np
from syntorch.core.trace import trace_manager
from syntorch.torch.tensor import Tensor
import syntorch.torch.functional as F


def relu(input):
    """ReLU 활성화 함수"""
    # 직접 구현: max(0, x) 연산
    if input.data.ndim == 0:  # 스칼라인 경우
        result_data = max(0, input.data)
    else:  # 1차원 이상인 경우
        # 결과 데이터를 담을 리스트 구조 생성
        result_data = create_nested_list(input.data.shape)

        # 모든 요소에 대해 ReLU 적용
        apply_function_to_tensor(input.data, result_data, lambda x: max(0, x))

    result = Tensor(result_data, input.dtype, input.device)
    trace_manager.trace_tensor("relu", [input], result)
    return result


def gelu(input):
    """GELU 활성화 함수"""
    # GELU 근사 구현 - 직접 계산
    # GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))

    # 필요한 상수
    sqrt_2_over_pi = 0.7978845608028654  # sqrt(2/pi)
    coeff = 0.044715

    # GELU 함수 정의
    def gelu_func(x):
        tanh_arg = sqrt_2_over_pi * (x + coeff * x**3)
        # tanh 계산 (Python math 모듈 대신 수동 계산)
        exp_pos = min(88.0, tanh_arg)  # 오버플로우 방지
        exp_neg = max(-88.0, -tanh_arg)  # 언더플로우 방지

        exp_pos_val = pow(2.718281828459045, exp_pos)
        exp_neg_val = pow(2.718281828459045, exp_neg)

        tanh_val = (exp_pos_val - exp_neg_val) / (exp_pos_val + exp_neg_val)

        # GELU 값 계산
        return 0.5 * x * (1 + tanh_val)

    if input.data.ndim == 0:  # 스칼라인 경우
        result_data = gelu_func(input.data)
    else:  # 1차원 이상인 경우
        # 결과 데이터를 담을 리스트 구조 생성
        result_data = create_nested_list(input.data.shape)

        # 모든 요소에 대해 GELU 적용
        apply_function_to_tensor(input.data, result_data, gelu_func)

    result = Tensor(result_data, input.dtype, input.device)
    trace_manager.trace_tensor("gelu", [input], result)
    return result


def softmax(input, dim=-1):
    """Softmax 함수"""
    # 직접 구현 - NumPy 사용하지 않음
    # softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))

    # 차원 처리
    shape = input.data.shape
    ndim = input.data.ndim

    # 음수 차원 처리
    if dim < 0:
        dim = ndim + dim

    # 스칼라인 경우 간단히 처리
    if ndim == 0:
        result_data = 1.0  # 스칼라의 softmax는 항상 1
        return Tensor(result_data, input.dtype, input.device)

    # 결과 데이터를 담을 구조 생성
    result_data = create_nested_list(shape)

    # 모든 슬라이스에 대해 softmax 계산
    for indices in iterate_tensor_indices(shape, dim):
        # 이 슬라이스의 인덱스들 가져오기
        slice_indices = generate_slice_indices(shape, dim, indices)

        # 최댓값 계산
        max_val = float("-inf")
        for idx in slice_indices:
            val = get_value_at_indices(input.data, idx)
            max_val = max(max_val, val)

        # 합계 계산
        sum_exp = 0.0
        for idx in slice_indices:
            val = get_value_at_indices(input.data, idx)
            exp_val = min(88.0, val - max_val)  # 오버플로우 방지
            sum_exp += pow(2.718281828459045, exp_val)

        # 각 요소의 softmax 값 계산 및 결과 저장
        for idx in slice_indices:
            val = get_value_at_indices(input.data, idx)
            exp_val = min(88.0, val - max_val)  # 오버플로우 방지
            softmax_val = pow(2.718281828459045, exp_val) / sum_exp
            set_value_at_indices(result_data, idx, softmax_val)

    result = Tensor(result_data, input.dtype, input.device)
    trace_manager.trace_tensor("softmax", [input], result, {"dim": dim})
    return result


def linear(input, weight, bias=None):
    """선형 레이어"""
    # 입력 및 가중치 확인
    if not isinstance(input, Tensor) or not isinstance(weight, Tensor):
        raise TypeError("입력과 가중치는 Tensor 타입이어야 합니다")

    # 디버깅: 입력 형태 출력
    print(f"Input shape: {input.shape}")
    print(f"Weight shape: {weight.shape}")
    print(f"Weight ndim: {weight.data.ndim}")

    # 가중치의 차원 확인
    if weight.data.ndim == 1:
        # 1차원 가중치는 2차원으로 변환 (1, N) 형태
        weight_2d = view(weight, (1, -1))
        print(f"Converted 1D weight to 2D shape: {weight_2d.shape}")

        # 가중치 전치 수행 - 차원 교환
        weight_transposed = transpose(weight_2d, 0, 1)
    else:
        # 가중치 전치 수행 (마지막 두 차원 교환)
        weight_transposed = transpose(weight, 0, 1)  # 두 개의 정수 인수로 차원 교환

    # 디버깅: 전치된 가중치 형태 출력
    print(f"Transposed weight shape: {weight_transposed.shape}")

    # 행렬 곱셈 수행
    try:
        output = input.matmul(weight_transposed)
        print(f"Output shape: {output.shape}")
    except ValueError as e:
        print(f"MatMul error: {str(e)}")
        print(
            f"Input last dim: {input.shape[-1]}, Weight_T first dim: {weight_transposed.shape[0]}"
        )
        raise

    # 편향 더하기 (있는 경우)
    if bias is not None:
        output = output + bias

    trace_manager.trace_tensor("linear", [input, weight, bias], output)
    return output


def embedding(input, weight):
    """임베딩 레이어"""
    # 입력 및 가중치 확인
    if not isinstance(input, Tensor) or not isinstance(weight, Tensor):
        raise TypeError("입력과 가중치는 Tensor 타입이어야 합니다")

    # 임베딩 룩업 수행
    indices = input.data.astype(np.int32)
    result_data = weight.data[indices]
    result = Tensor(result_data, weight.dtype, weight.device)

    trace_manager.trace_tensor("embedding", [input, weight], result)
    return result


def layer_norm(input, normalized_shape, weight=None, bias=None, eps=1e-5):
    """레이어 정규화"""
    # 차원 계산
    ndim = input.data.ndim
    if isinstance(normalized_shape, int):
        normalized_shape = (normalized_shape,)

    # 정규화할 차원 결정
    norm_dims = tuple(range(ndim - len(normalized_shape), ndim))

    # 평균 및 분산 계산
    mean = np.mean(input.data, axis=norm_dims, keepdims=True)
    var = np.var(input.data, axis=norm_dims, keepdims=True)

    # 정규화
    x_norm = (input.data - mean) / np.sqrt(var + eps)

    # 스케일 및 시프트 적용
    if weight is not None:
        x_norm = x_norm * weight.data
    if bias is not None:
        x_norm = x_norm + bias.data

    result = Tensor(x_norm, input.dtype, input.device)
    trace_manager.trace_tensor(
        "layer_norm", [input], result, {"normalized_shape": normalized_shape, "eps": eps}
    )

    return result


def dropout(input, p=0.5, training=True):
    """드롭아웃"""
    if not training or p == 0:
        return input

    # 마스크 생성
    mask = np.random.binomial(1, 1 - p, input.data.shape) / (1 - p)
    result_data = input.data * mask
    result = Tensor(result_data, input.dtype, input.device)

    trace_manager.trace_tensor("dropout", [input], result, {"p": p})
    return result


def view(input_tensor, shape):
    """텐서 형태 변경 (torch.view에 해당)"""
    if not isinstance(input_tensor, Tensor):
        raise TypeError("input_tensor must be a Tensor")

    result = Tensor(input_tensor.data.reshape(shape), input_tensor.dtype, input_tensor.device)

    # 트레이싱
    trace_manager.trace_tensor(
        "view",
        [input_tensor, shape],
        result,
        {"input_shape": input_tensor.shape, "output_shape": result.shape, "target_shape": shape},
    )
    return result


def transpose(input_tensor, dim0, dim1):
    """두 차원 간의 전치 (torch.transpose에 해당)"""
    if not isinstance(input_tensor, Tensor):
        raise TypeError("input_tensor must be a Tensor")

    # 입력 텐서의 형태 가져오기
    shape = input_tensor.shape
    ndim = len(shape)

    print(f"TRANSPOSE: Input shape: {shape}, ndim: {ndim}, dim0: {dim0}, dim1: {dim1}")

    # 차원 유효성 검사
    if dim0 < 0 or dim0 >= ndim or dim1 < 0 or dim1 >= ndim:
        raise ValueError(
            f"차원 인덱스가 범위를 벗어났습니다: dim0={dim0}, dim1={dim1}, ndim={ndim}"
        )

    # 동일한 차원이면 입력 그대로 반환
    if dim0 == dim1:
        print(f"TRANSPOSE: Same dimensions, returning clone")
        return input_tensor.clone()

    # 디버깅: 항상 NumPy transpose 사용
    axes = list(range(input_tensor.data.ndim))
    axes[dim0], axes[dim1] = axes[dim1], axes[dim0]
    print(f"TRANSPOSE: Reordered axes: {axes}")

    # NumPy로 정확한 전치 수행
    result_data = np.transpose(input_tensor.data, axes)
    result = Tensor(result_data, input_tensor.dtype, input_tensor.device)
    print(f"TRANSPOSE: Result shape: {result.shape}")

    # 트레이싱
    trace_manager.trace_tensor(
        "transpose",
        [input_tensor, dim0, dim1],
        result,
        {
            "input_shape": input_tensor.shape,
            "output_shape": result.shape,
            "dim0": dim0,
            "dim1": dim1,
        },
    )
    return result


def chunk(input_tensor, chunks, dim=0):
    """지정된 차원을 따라 텐서를 여러 청크로 분할 (torch.chunk에 해당)"""
    if not isinstance(input_tensor, Tensor):
        raise TypeError("input_tensor must be a Tensor")

    # 각 청크의 크기 계산
    dim_size = input_tensor.data.shape[dim]
    chunk_size = int(np.ceil(dim_size / chunks))

    # 청크로 분할
    result = []
    for i in range(0, dim_size, chunk_size):
        end = min(i + chunk_size, dim_size)

        # 슬라이싱 인덱스 생성
        indices = [slice(None)] * input_tensor.data.ndim
        indices[dim] = slice(i, end)

        # 청크 추출
        chunk_data = input_tensor.data[tuple(indices)]
        chunk_tensor = Tensor(chunk_data, input_tensor.dtype, input_tensor.device)
        result.append(chunk_tensor)

        # 각 청크에 대한 트레이싱
        trace_manager.trace_tensor(
            "chunk_part",
            [input_tensor, chunks, dim],
            chunk_tensor,
            {
                "input_shape": input_tensor.shape,
                "output_shape": chunk_tensor.shape,
                "chunks": chunks,
                "dim": dim,
                "part": i // chunk_size,
            },
        )

    # 전체 청크 연산에 대한 트레이싱
    trace_manager.trace_tensor(
        "chunk",
        [input_tensor, chunks, dim],
        result,
        {
            "input_shape": input_tensor.shape,
            "chunks": chunks,
            "dim": dim,
            "num_outputs": len(result),
        },
    )

    return result


def cat(tensors, dim=0):
    """텐서들을 지정된 차원을 따라 연결 (torch.cat에 해당)"""
    if not tensors:
        raise ValueError("tensors must be a non-empty list")

    # 모든 입력을 Tensor로 변환
    tensor_datas = []
    for t in tensors:
        if isinstance(t, Tensor):
            tensor_datas.append(t.data)
        else:
            tensor_datas.append(np.array(t))

    # 연결 수행
    result_data = np.concatenate(tensor_datas, axis=dim)

    # 결과 Tensor 생성
    result = Tensor(
        result_data,
        tensors[0].dtype if isinstance(tensors[0], Tensor) else None,
        tensors[0].device if isinstance(tensors[0], Tensor) else None,
    )

    # 트레이싱
    trace_manager.trace_tensor(
        "cat",
        tensors + [dim],
        result,
        {"input_count": len(tensors), "output_shape": result.shape, "dim": dim},
    )

    return result


# 헬퍼 함수들
def create_nested_list(shape):
    """주어진 shape에 맞는 중첩 리스트 생성"""
    if len(shape) == 1:
        return [0.0] * shape[0]
    else:
        return [create_nested_list(shape[1:]) for _ in range(shape[0])]


def apply_function_to_tensor(input_data, output_data, func):
    """텐서의 모든 요소에 함수 적용"""
    if isinstance(input_data, (int, float)):
        return func(input_data)

    # 0차원 텐서 처리
    if hasattr(input_data, "ndim") and input_data.ndim == 0:
        if hasattr(input_data, "item"):
            return func(input_data.item())
        else:
            return func(input_data)

    for i in range(len(input_data)):
        if isinstance(input_data[i], list) or (
            hasattr(input_data[i], "__len__") and not isinstance(input_data[i], (str, bytes))
        ):
            apply_function_to_tensor(input_data[i], output_data[i], func)
        else:
            output_data[i] = func(input_data[i])


def iterate_tensor_indices(shape, dim_to_skip):
    """텐서의 인덱스를 반복하되, 특정 차원은 건너뜀"""
    # 건너뛸 차원을 제외한 shape
    reduced_shape = shape[:dim_to_skip] + shape[dim_to_skip + 1 :]

    # 비어있는 형태면 빈 리스트 반환
    if not reduced_shape:
        yield []
        return

    # 모든 인덱스 조합 생성
    counters = [0] * len(reduced_shape)
    yield counters[:]

    while True:
        # 카운터 증가
        for i in range(len(reduced_shape) - 1, -1, -1):
            counters[i] += 1
            if counters[i] < reduced_shape[i]:
                break
            counters[i] = 0
        else:
            # 모든 조합을 다 생성했음
            break

        yield counters[:]


def generate_slice_indices(shape, dim, base_indices):
    """지정된 차원에 대한 모든 인덱스 조합 생성"""
    dim_size = shape[dim]

    # 기본 인덱스를 dim 위치에 맞게 확장
    result = []
    for i in range(dim_size):
        # dim 차원의 인덱스 삽입
        full_idx = base_indices[:dim] + [i] + base_indices[dim:]
        result.append(full_idx)

    return result


def get_value_at_indices(data, indices):
    """중첩된 데이터 구조에서 지정된 인덱스의 값 가져오기"""
    current = data
    for idx in indices:
        if hasattr(current, "shape"):  # 배열 형태의 객체인 경우
            if idx >= current.shape[0]:
                raise IndexError(
                    f"인덱스 {idx}가 범위를 벗어났습니다. 데이터 크기: {current.shape[0]}"
                )
            current = current[idx]
        else:  # 리스트인 경우
            if idx >= len(current):
                raise IndexError(f"인덱스 {idx}가 범위를 벗어났습니다. 데이터 크기: {len(current)}")
            current = current[idx]
    return current


def set_value_at_indices(data, indices, value):
    """중첩된 데이터 구조의 지정된 인덱스에 값 설정"""
    current = data
    for i, idx in enumerate(indices[:-1]):
        current = current[idx]

    current[indices[-1]] = value
