"""
집합 통신(Collective Communication) 모듈

이 모듈은 분산 학습에 필요한 집합 통신 프리미티브를 제공합니다.
실제 통신은 수행하지 않지만 trace_manager를 사용하여 호출 기록을 남깁니다.
"""

import syntorch.torch.functional as F
from syntorch.core.trace import trace_manager, SyntorchLayer
from syntorch.torch.tensor import Tensor


class CollectiveCommunication(SyntorchLayer):
    """집합 통신 프리미티브 클래스"""

    def __init__(self, group_size=1, group_rank=0, backend="nccl"):
        """
        Args:
            group_size: 통신 그룹의 크기(프로세스/GPU 수)
            group_rank: 현재 프로세스/GPU의 랭크(ID)
            backend: 통신 백엔드("nccl", "gloo" 등)
        """
        self.group_size = group_size
        self.group_rank = group_rank
        self.backend = backend

    @staticmethod
    def all_reduce(tensor, op="sum", group_size=1, group_rank=0):
        """All-Reduce 연산 (각 프로세스의 값을 결합하고 모든 프로세스에 결과 배포)

        Args:
            tensor: 입력 텐서
            op: 리덕션 연산("sum", "avg", "min", "max", "prod")
            group_size: 통신 그룹의 크기
            group_rank: 현재 프로세스의 랭크

        Returns:
            Tensor: all-reduce 결과 텐서 (입력 텐서와 동일한 모양)
        """
        # 실제 구현을 시뮬레이션 - 실제로는 모든 프로세스의 데이터를 수집하고 결합
        if isinstance(tensor, Tensor):
            # 연산 유형에 따라 결과 계산
            if op == "sum":
                # 합계 시뮬레이션 - 현재 값을 group_size배
                # (실제로는 모든 랭크의 데이터 합)
                result = tensor * group_size
            elif op == "avg":
                # 평균 시뮬레이션 - 현재 값 그대로 사용
                # (실제로는 모든 랭크의 데이터 합을 group_size로 나눔)
                result = tensor.clone()
            elif op == "max":
                # 최대값 시뮬레이션
                result = tensor.clone()
            elif op == "min":
                # 최소값 시뮬레이션
                result = tensor.clone()
            elif op == "prod":
                # 곱 시뮬레이션 - 현재 값을 group_size 제곱
                # (실제로는 모든 랭크의 데이터 곱)
                # Tensor에 __pow__ 메서드가 없으므로 반복 곱셈으로 구현
                result = tensor.clone()
                for _ in range(group_size - 1):
                    result = result * tensor
            else:
                result = tensor.clone()
        else:
            result = tensor

        # 통신 연산 메타데이터
        metadata = {
            "comm_type": "all_reduce",
            "op": op,
            "group_size": group_size,
            "group_rank": group_rank,
            "input_shape": tensor.shape if hasattr(tensor, "shape") else None,
            "output_shape": result.shape if hasattr(result, "shape") else None,
            "dtype": tensor.dtype if hasattr(tensor, "dtype") else None,
        }

        # 텐서 연산 추적 - 통신 연산 전용 추적 함수 사용
        trace_manager.trace_comm_op("all_reduce", [tensor], result, metadata)

        return result

    @staticmethod
    def all_gather(tensor, group_size=1, group_rank=0):
        """All-Gather 연산 (각 프로세스의 텐서를 수집하여 모든 프로세스에 전체 결과 배포)

        Args:
            tensor: 입력 텐서
            group_size: 통신 그룹의 크기
            group_rank: 현재 프로세스의 랭크

        Returns:
            Tensor: all-gather 결과 텐서 (첫 번째 차원이 group_size배 확장됨)
        """
        # 실제 구현에서는 모든 랭크의 텐서를 수집하여 첫 번째 차원을 따라 연결
        # 여기서는 단순히 첫 번째 차원을 group_size배 확장
        if isinstance(tensor, Tensor):
            # 첫 번째 차원을 group_size배 확장
            # F.tile 함수 사용
            tile_shape = [group_size] + [1] * (len(tensor.shape) - 1)
            result = F.tile(tensor, tile_shape)
        else:
            # 텐서가 아닌 경우 그대로 반환
            result = tensor

        # 통신 연산 메타데이터
        metadata = {
            "comm_type": "all_gather",
            "group_size": group_size,
            "group_rank": group_rank,
            "input_shape": tensor.shape if hasattr(tensor, "shape") else None,
            "output_shape": result.shape if hasattr(result, "shape") else None,
            "dtype": tensor.dtype if hasattr(tensor, "dtype") else None,
        }

        # 텐서 연산 추적 - 통신 연산 전용 추적 함수 사용
        trace_manager.trace_comm_op("all_gather", [tensor], result, metadata)

        return result

    @staticmethod
    def reduce_scatter(tensor, op="sum", group_size=1, group_rank=0):
        """Reduce-Scatter 연산 (텐서를 결합한 후 결과를 분산)

        Args:
            tensor: 입력 텐서 (첫 번째 차원이 group_size로 나누어져야 함)
            op: 리덕션 연산("sum", "avg", "min", "max", "prod")
            group_size: 통신 그룹의 크기
            group_rank: 현재 프로세스의 랭크

        Returns:
            Tensor: reduce-scatter 결과 텐서 (첫 번째 차원이 group_size로 나누어짐)
        """
        # 실제 구현에서는 텐서를 group_size개의 청크로 나누고 reduce 연산 후 분산
        # 여기서는 단순히 텐서의 현재 랭크에 해당하는 부분만 추출
        if isinstance(tensor, Tensor):
            # 텐서의 첫 번째 차원을 group_size로 나눔
            chunk_size = tensor.shape[0] // group_size
            start_idx = group_rank * chunk_size
            end_idx = start_idx + chunk_size

            # 현재 랭크에 해당하는 청크 추출
            result = tensor[start_idx:end_idx]
        else:
            # 텐서가 아닌 경우 그대로 반환
            result = tensor

        # 통신 연산 메타데이터
        metadata = {
            "comm_type": "reduce_scatter",
            "op": op,
            "group_size": group_size,
            "group_rank": group_rank,
            "input_shape": tensor.shape if hasattr(tensor, "shape") else None,
            "output_shape": result.shape if hasattr(result, "shape") else None,
            "dtype": tensor.dtype if hasattr(tensor, "dtype") else None,
        }

        # 텐서 연산 추적 - 통신 연산 전용 추적 함수 사용
        trace_manager.trace_comm_op("reduce_scatter", [tensor], result, metadata)

        return result

    @staticmethod
    def broadcast(tensor, src_rank=0, group_size=1, group_rank=0):
        """Broadcast 연산 (소스 랭크의 텐서를 모든 프로세스에 복제)

        Args:
            tensor: 입력 텐서
            src_rank: 소스 랭크
            group_size: 통신 그룹의 크기
            group_rank: 현재 프로세스의 랭크

        Returns:
            Tensor: broadcast 결과 텐서 (입력 텐서와 동일)
        """
        # 실제 통신은 수행하지 않고 입력 텐서를 그대로 반환
        result = tensor

        # 통신 연산 메타데이터
        metadata = {
            "comm_type": "broadcast",
            "src_rank": src_rank,
            "group_size": group_size,
            "group_rank": group_rank,
            "input_shape": tensor.shape if hasattr(tensor, "shape") else None,
            "output_shape": result.shape if hasattr(result, "shape") else None,
            "dtype": tensor.dtype if hasattr(tensor, "dtype") else None,
        }

        # 텐서 연산 추적 - 통신 연산 전용 추적 함수 사용
        trace_manager.trace_comm_op("broadcast", [tensor], result, metadata)

        return result


# 편의를 위한 전역 함수
def all_reduce(tensor, op="sum", group_size=1, group_rank=0):
    return CollectiveCommunication.all_reduce(tensor, op, group_size, group_rank)


def all_gather(tensor, group_size=1, group_rank=0):
    return CollectiveCommunication.all_gather(tensor, group_size, group_rank)


def reduce_scatter(tensor, op="sum", group_size=1, group_rank=0):
    return CollectiveCommunication.reduce_scatter(tensor, op, group_size, group_rank)


def broadcast(tensor, src_rank=0, group_size=1, group_rank=0):
    return CollectiveCommunication.broadcast(tensor, src_rank, group_size, group_rank)
