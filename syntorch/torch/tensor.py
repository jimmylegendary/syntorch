import numpy as np
from syntorch.core.trace import SyntorchLayer, trace_manager
from syntorch.runtime.tiling import TilingEngine
from syntorch.os.memory import MemoryManager


class Tensor(SyntorchLayer):
    """PyTorch Tensor와 호환되는 텐서 클래스"""

    def __init__(self, data, dtype=None, device=None, requires_grad=False):
        """텐서 초기화

        Args:
            data: 텐서 데이터 (리스트, 튜플, numpy 배열 등)
            dtype: 데이터 타입
            device: 디바이스 ('cpu', 'cuda' 등)
            requires_grad: gradient 계산 여부
        """
        # numpy 배열로 변환
        if isinstance(data, Tensor):
            self.data = data.data.copy()
        else:
            self.data = np.array(data)

        # 기타 속성 설정
        self.dtype = dtype
        self.device = device or "cpu"
        self.requires_grad = requires_grad
        self.grad = None if requires_grad else None

        # 메모리 관리 관련 속성
        self._id = id(self)  # 트레이싱용 고유 ID
        self.memory_addr = None  # 할당된 메모리 주소

        # CUDA 디바이스인 경우 메모리 할당
        if self.device.startswith("cuda"):
            self._allocate_gpu_memory()

    @property
    def shape(self):
        """텐서 형태"""
        return self.data.shape

    @property
    def ndim(self):
        """차원 수"""
        return self.data.ndim

    def _allocate_gpu_memory(self):
        """GPU 메모리 할당"""
        # MemoryManager 인스턴스 가져오기
        memory_manager = MemoryManager()

        # 필요한 메모리 크기 계산
        size = self.data.size * self.data.itemsize

        # 메모리 할당
        self.memory_addr = memory_manager.allocate(size)

        # 키-주소 등록
        memory_manager.register_tensor(self._id, self.memory_addr, size)

        # 데이터 복사
        memory_manager.write_memory(self.data.tobytes(), address=self.memory_addr)

    def normal_(self, mean=0.0, std=1.0):
        """텐서를 정규 분포로 초기화 (in-place)"""
        self.data = np.random.normal(mean, std, self.data.shape)
        return self

    def zero_(self):
        """텐서를 0으로 초기화 (in-place)"""
        self.data.fill(0)
        return self

    def fill_(self, value):
        """텐서를 특정 값으로 초기화 (in-place)"""
        self.data.fill(value)
        return self

    def unsqueeze(self, dim):
        """지정된 차원에 크기 1인 차원 추가"""
        return Tensor(np.expand_dims(self.data, axis=dim), self.dtype, self.device)

    def expand_as(self, other):
        """다른 텐서의 크기에 맞게 확장"""
        target_shape = other.shape
        # NumPy의 broadcast_to 함수 사용
        expanded_data = np.broadcast_to(self.data, target_shape)
        return Tensor(expanded_data, self.dtype, self.device)

    def clone(self):
        """텐서 복제"""
        return Tensor(self.data.copy(), self.dtype, self.device, self.requires_grad)

    def softmax(self, dim=-1):
        """소프트맥스 함수"""
        # 수치 안정성을 위해 최대값 빼기
        x = self.data - np.max(self.data, axis=dim, keepdims=True)
        exp_x = np.exp(x)
        softmax_x = exp_x / np.sum(exp_x, axis=dim, keepdims=True)
        return Tensor(softmax_x, self.dtype, self.device)

    def multinomial(self, num_samples, replacement=False):
        """확률 분포에서 샘플링"""
        # 확률 분포로 간주
        probs = self.data

        # 각 행의 합이 1이 되도록 정규화
        if probs.ndim > 1:
            probs = probs / np.sum(probs, axis=-1, keepdims=True)
        else:
            probs = probs / np.sum(probs)

        # 샘플링
        if probs.ndim > 1:
            samples = np.zeros((probs.shape[0], num_samples), dtype=np.int64)
            for i in range(probs.shape[0]):
                samples[i] = np.random.choice(
                    probs.shape[1], size=num_samples, replace=replacement, p=probs[i]
                )
        else:
            samples = np.random.choice(
                probs.shape[0], size=num_samples, replace=replacement, p=probs
            )

        return Tensor(samples, dtype="long", device=self.device)

    def item(self):
        """단일 요소 텐서를 Python 스칼라로 변환"""
        if self.data.size != 1:
            raise ValueError("item()은 크기가 1인 텐서에서만 호출할 수 있습니다")
        return self.data.item()

    def __add__(self, other):
        """덧셈 연산"""
        if isinstance(other, (int, float)):
            result = Tensor(self.data + other, self.dtype, self.device)
        else:
            result = Tensor(self.data + other.data, self.dtype, self.device)

        # 트레이싱
        trace_manager.trace_tensor(
            "add",
            [self, other],
            result,
            {
                "input_shape": self.shape,
                "input_dtype": self.dtype,
                "output_shape": result.shape,
                "output_dtype": result.dtype,
            },
        )
        return result

    def __mul__(self, other):
        """곱셈 연산"""
        if isinstance(other, (int, float)):
            result = Tensor(self.data * other, self.dtype, self.device)
        else:
            result = Tensor(self.data * other.data, self.dtype, self.device)

        # 트레이싱
        trace_manager.trace_tensor(
            "mul",
            [self, other],
            result,
            {
                "input_shape": self.shape,
                "input_dtype": self.dtype,
                "output_shape": result.shape,
                "output_dtype": result.dtype,
            },
        )
        return result

    def __truediv__(self, other):
        """나눗셈 연산"""
        if isinstance(other, (int, float)):
            result = Tensor(self.data / other, self.dtype, self.device)
        else:
            result = Tensor(self.data / other.data, self.dtype, self.device)

        # 트레이싱
        trace_manager.trace_tensor(
            "div",
            [self, other],
            result,
            {
                "input_shape": self.shape,
                "input_dtype": self.dtype,
                "output_shape": result.shape,
                "output_dtype": result.dtype,
            },
        )
        return result

    def __gt__(self, other):
        """초과 비교 (>)"""
        if isinstance(other, (int, float)):
            result = Tensor(self.data > other, dtype="bool", device=self.device)
        else:
            result = Tensor(self.data > other.data, dtype="bool", device=self.device)

        # 트레이싱
        trace_manager.trace_tensor("gt", [self, other], result)
        return result

    def __ge__(self, other):
        """이상 비교 (>=)"""
        if isinstance(other, (int, float)):
            result = Tensor(self.data >= other, dtype="bool", device=self.device)
        else:
            result = Tensor(self.data >= other.data, dtype="bool", device=self.device)

        # 트레이싱
        trace_manager.trace_tensor("ge", [self, other], result)
        return result

    def __lt__(self, other):
        """미만 비교 (<)"""
        if isinstance(other, (int, float)):
            result = Tensor(self.data < other, dtype="bool", device=self.device)
        else:
            result = Tensor(self.data < other.data, dtype="bool", device=self.device)

        # 트레이싱
        trace_manager.trace_tensor("lt", [self, other], result)
        return result

    def __le__(self, other):
        """이하 비교 (<=)"""
        if isinstance(other, (int, float)):
            result = Tensor(self.data <= other, dtype="bool", device=self.device)
        else:
            result = Tensor(self.data <= other.data, dtype="bool", device=self.device)

        # 트레이싱
        trace_manager.trace_tensor("le", [self, other], result)
        return result

    def __eq__(self, other):
        """동등 비교 (==)"""
        if isinstance(other, (int, float)):
            result = Tensor(self.data == other, dtype="bool", device=self.device)
        else:
            result = Tensor(self.data == other.data, dtype="bool", device=self.device)

        # 트레이싱
        trace_manager.trace_tensor("eq", [self, other], result)
        return result

    def __ne__(self, other):
        """부등 비교 (!=)"""
        if isinstance(other, (int, float)):
            result = Tensor(self.data != other, dtype="bool", device=self.device)
        else:
            result = Tensor(self.data != other.data, dtype="bool", device=self.device)

        # 트레이싱
        trace_manager.trace_tensor("ne", [self, other], result)
        return result

    def matmul(self, other):
        """행렬 곱셈"""
        # 디버깅 정보 출력
        print(f"MATMUL: Self shape: {self.shape}, Other shape: {other.shape}")

        # Runtime Layer에 위임하여 효율적인 타일링 수행 (GPU인 경우)
        if self.device.startswith("cuda") and other.device.startswith("cuda"):
            tiling_engine = TilingEngine()
            result_data = tiling_engine.tile_matmul(self.data, other.data)
            result = Tensor(result_data, self.dtype, self.device)
        else:
            # CPU 기본 구현 - NumPy 사용하지 않고 직접 구현
            # 행렬 차원 확인
            a_shape = self.shape
            b_shape = other.shape

            print(f"MATMUL: A dims: {len(a_shape)}, B dims: {len(b_shape)}")

            # 차원 확인 - 행렬 곱셈 가능 여부
            if len(a_shape) >= 1 and len(b_shape) >= 1:
                # 마지막 차원과 마지막에서 두 번째 차원이 맞아야 함
                if len(a_shape) >= 2 and len(b_shape) >= 2:
                    if a_shape[-1] != b_shape[-2]:
                        print(
                            f"MATMUL ERROR: A last dim ({a_shape[-1]}) != B second-to-last dim ({b_shape[-2]})"
                        )
                        raise ValueError(
                            f"행렬 곱셈을 위한 차원이 맞지 않습니다: {a_shape[-1]} vs {b_shape[-2]}"
                        )
                elif len(a_shape) == 1 and len(b_shape) >= 2:
                    if a_shape[0] != b_shape[-2]:
                        print(
                            f"MATMUL ERROR: A dim ({a_shape[0]}) != B second-to-last dim ({b_shape[-2]})"
                        )
                        raise ValueError(
                            f"행렬 곱셈을 위한 차원이 맞지 않습니다: {a_shape[0]} vs {b_shape[-2]}"
                        )
                elif len(a_shape) >= 2 and len(b_shape) == 1:
                    if a_shape[-1] != b_shape[0]:
                        print(f"MATMUL ERROR: A last dim ({a_shape[-1]}) != B dim ({b_shape[0]})")
                        raise ValueError(
                            f"행렬 곱셈을 위한 차원이 맞지 않습니다: {a_shape[-1]} vs {b_shape[0]}"
                        )
                elif len(a_shape) == 1 and len(b_shape) == 1:
                    if a_shape[0] != b_shape[0]:
                        print(f"MATMUL ERROR: A dim ({a_shape[0]}) != B dim ({b_shape[0]})")
                        raise ValueError(
                            f"벡터 내적을 위한 차원이 맞지 않습니다: {a_shape[0]} vs {b_shape[0]}"
                        )

            # 1차원 텐서 처리 (벡터)
            if len(a_shape) == 1 and len(b_shape) == 1:
                # 벡터 내적 (dot product)
                # 결과는 스칼라
                print(f"MATMUL: Vector dot product")
                dot_product = 0.0
                for i in range(a_shape[0]):
                    dot_product += self.data[i] * other.data[i]
                result_data = [[dot_product]]
                result = Tensor(result_data, self.dtype, self.device)
                print(f"MATMUL: Dot product result: {dot_product}")

            elif len(a_shape) == 1:
                # a가 벡터인 경우: 1차원 벡터와 행렬의 곱
                print(f"MATMUL: Vector-matrix product")
                result_data = [0.0] * b_shape[1]
                for j in range(b_shape[1]):
                    for k in range(a_shape[0]):
                        result_data[j] += self.data[k] * other.data[k, j]

                result = Tensor(result_data, self.dtype, self.device)
                print(f"MATMUL: Vec-mat result shape: {result.shape}")

            elif len(b_shape) == 1:
                # b가 벡터인 경우: 행렬과 1차원 벡터의 곱
                print(f"MATMUL: Matrix-vector product")
                result_data = [0.0] * a_shape[0]
                for i in range(a_shape[0]):
                    for k in range(a_shape[1]):
                        result_data[i] += self.data[i, k] * other.data[k]

                result = Tensor(result_data, self.dtype, self.device)
                print(f"MATMUL: Mat-vec result shape: {result.shape}")

            else:
                # 두 텐서 모두 2차원 이상인 경우: 표준 행렬 곱
                print(f"MATMUL: Matrix-matrix product")
                # 표준 행렬 곱셈 (2D x 2D)
                if len(a_shape) == 2 and len(b_shape) == 2:
                    # 결과 행렬 초기화
                    result_data = [[0.0 for _ in range(b_shape[1])] for _ in range(a_shape[0])]

                    # 직접 행렬 곱 계산
                    for i in range(a_shape[0]):
                        for j in range(b_shape[1]):
                            for k in range(a_shape[1]):
                                result_data[i][j] += self.data[i, k] * other.data[k, j]

                    result = Tensor(result_data, self.dtype, self.device)
                    print(f"MATMUL: Mat-mat result shape: {result.shape}")

                else:
                    # 다차원 텐서 곱셈 - 배치 처리
                    print(f"MATMUL: Batched tensor product")
                    # 마지막 두 차원에 대해 표준 행렬 곱 수행
                    if len(a_shape) == 3 and len(b_shape) == 3:
                        # 배치 차원 확인
                        if a_shape[0] != b_shape[0]:
                            print(
                                f"MATMUL ERROR: Batch dimensions don't match: {a_shape[0]} vs {b_shape[0]}"
                            )
                            raise ValueError(
                                f"배치 차원이 맞지 않습니다: {a_shape[0]} vs {b_shape[0]}"
                            )

                        # 결과 텐서 초기화
                        batch_size = a_shape[0]
                        m = a_shape[1]
                        n = b_shape[2]

                        # 3차원 결과 텐서 생성
                        result_data = [
                            [[0.0 for _ in range(n)] for _ in range(m)] for _ in range(batch_size)
                        ]

                        # 배치 단위로 행렬 곱 계산
                        for b in range(batch_size):
                            for i in range(m):
                                for j in range(n):
                                    for k in range(a_shape[2]):
                                        result_data[b][i][j] += (
                                            self.data[b, i, k] * other.data[b, k, j]
                                        )

                        result = Tensor(result_data, self.dtype, self.device)
                        print(f"MATMUL: Batched result shape: {result.shape}")

                    else:
                        # 4차원 이상의 텐서는 현재 지원하지 않음
                        print(f"MATMUL ERROR: 4D+ tensors not supported yet")
                        raise NotImplementedError(
                            "4차원 이상의 텐서 행렬 곱은 아직 직접 구현되지 않았습니다."
                        )

        # 트레이싱
        trace_manager.trace_tensor(
            "matmul",
            [self, other],
            result,
            {
                "input1_shape": self.shape,
                "input1_dtype": self.dtype,
                "input2_shape": other.shape,
                "input2_dtype": other.dtype,
                "output_shape": result.shape,
                "output_dtype": result.dtype,
            },
        )
        return result

    def __getitem__(self, idx):
        """인덱싱"""
        if isinstance(idx, Tensor):
            idx = idx.data  # Tensor를 NumPy 배열로 변환
        return Tensor(self.data[idx], self.dtype, self.device)

    def __setitem__(self, idx, value):
        """항목 설정"""
        if isinstance(idx, Tensor):
            idx = idx.data  # Tensor를 NumPy 배열로 변환
        if isinstance(value, Tensor):
            self.data[idx] = value.data
        else:
            self.data[idx] = value

    def to(self, device=None, dtype=None):
        """디바이스 또는 데이터 타입 변경"""
        if dtype is not None:
            # 데이터 타입 변경
            pass

        if device is not None and device != self.device:
            # 다른 디바이스로 이동
            new_tensor = Tensor(self.data, self.dtype, device, self.requires_grad)
            return new_tensor

        return self

    def __repr__(self):
        """텐서 표현"""
        return f"Tensor({self.data}, device='{self.device}')"


# PyTorch와 호환되는 함수
def tensor(data, dtype=None, device=None, requires_grad=False):
    """텐서 생성 함수"""
    return Tensor(data, dtype, device, requires_grad)


def zeros(shape, dtype=None, device=None, requires_grad=False):
    """영 텐서 생성"""
    return Tensor(np.zeros(shape), dtype, device, requires_grad)


def ones(shape, dtype=None, device=None, requires_grad=False):
    """일 텐서 생성"""
    return Tensor(np.ones(shape), dtype, device, requires_grad)


def randn(shape, dtype=None, device=None, requires_grad=False):
    """표준 정규분포 랜덤 텐서 생성"""
    return Tensor(np.random.randn(*shape), dtype, device, requires_grad)


def randint(low, high=None, size=None, device=None, dtype=None, requires_grad=False):
    """임의의 정수 텐서 생성

    Args:
        low: 최솟값 (high가 제공되면) 또는 최댓값
        high: 최댓값 (제공되지 않으면 low가 최댓값이 되고 최솟값은 0)
        size: 텐서 형태 (튜플)
        device: 디바이스
        dtype: 데이터 타입
        requires_grad: gradient 계산 여부
    """
    if high is None:
        high = low
        low = 0

    if size is None:
        size = (1,)

    # numpy의 random.randint 함수 사용
    data = np.random.randint(low, high, size=size)

    return Tensor(data, dtype, device, requires_grad)


def arange(start, end=None, step=1, dtype=None, device=None, requires_grad=False):
    """범위 텐서 생성"""
    if end is None:
        end = start
        start = 0
    return Tensor(np.arange(start, end, step), dtype, device, requires_grad)


def cat(tensors, dim=0):
    """텐서 연결"""
    tensor_datas = [t.data if isinstance(t, Tensor) else t for t in tensors]
    result_data = np.concatenate(tensor_datas, axis=dim)
    # 첫 번째 텐서의 device 및 dtype 유지
    device = tensors[0].device if isinstance(tensors[0], Tensor) else None
    dtype = tensors[0].dtype if isinstance(tensors[0], Tensor) else None
    return Tensor(result_data, dtype, device)


def sort(input, dim=-1, descending=False):
    """텐서 정렬"""
    sorted_data = np.sort(input.data, axis=dim)
    sorted_indices = np.argsort(input.data, axis=dim)

    if descending:
        sorted_data = np.flip(sorted_data, axis=dim)
        sorted_indices = np.flip(sorted_indices, axis=dim)

    return Tensor(sorted_data, input.dtype, input.device), Tensor(
        sorted_indices, "long", input.device
    )


def topk(input, k, dim=-1, largest=True, sorted=True):
    """상위 k개 값 찾기"""
    if largest:
        indices = np.argsort(-input.data, axis=dim)
    else:
        indices = np.argsort(input.data, axis=dim)

    # k개 선택
    indices = np.take(indices, np.arange(k), axis=dim)

    # 선택된 인덱스에 해당하는 값 수집
    values = np.take_along_axis(input.data, indices, axis=dim)

    return Tensor(values, input.dtype, input.device), Tensor(indices, "long", input.device)


def cumsum(input, dim):
    """누적 합계"""
    return Tensor(np.cumsum(input.data, axis=dim), input.dtype, input.device)
