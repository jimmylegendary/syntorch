from syntorch.core.trace import SyntorchLayer, trace_manager
from syntorch.os.memory import MemoryManager
import numpy as np

class TilingEngine(SyntorchLayer):
    """텐서 연산의 타일링 처리"""
    
    def __init__(self, memory_manager=None, compute_kernel=None):
        self.memory_manager = memory_manager or MemoryManager()
        self.compute_kernel = compute_kernel
        self.default_tile_size = (64, 64)  # 기본 타일 크기
    
    def tile_matmul(self, A, B, C=None, tile_size=None):
        """행렬 곱셈 타일링 구현
        
        Args:
            A: 첫 번째 입력 텐서 (또는 주소)
            B: 두 번째 입력 텐서 (또는 주소)
            C: 출력 텐서 (또는 주소), None이면 새로 할당
            tile_size: 타일 크기 튜플 (높이, 너비)
        
        Returns:
            C: 결과 텐서 (또는 주소)
        """
        # 입력이 주소인 경우 메모리에서 읽기
        A_data = self._get_tensor_data(A)
        B_data = self._get_tensor_data(B)
        
        # 차원 검사
        if A_data.shape[1] != B_data.shape[0]:
            raise ValueError(f"행렬 곱셈 차원 불일치: {A_data.shape} @ {B_data.shape}")
        
        # 결과 텐서 할당 (필요시)
        if C is None:
            C_shape = (A_data.shape[0], B_data.shape[1])
            C_size = np.prod(C_shape) * A_data.itemsize
            C_addr = self.memory_manager.allocate(C_size)
            C_data = np.zeros(C_shape, dtype=A_data.dtype)
        else:
            C_data = self._get_tensor_data(C)
        
        # 타일 크기 설정
        if tile_size is None:
            tile_size = self.default_tile_size
        
        # 타일링 수행
        self._tiled_matmul(A_data, B_data, C_data, tile_size)
        
        # 결과를 메모리에 저장 (주소가 있는 경우)
        if isinstance(C, int):
            self._store_tensor_data(C_data, C)
            return C
        
        # 새로 할당된 경우
        if C is None:
            self._store_tensor_data(C_data, C_addr)
            return C_addr
        
        return C_data
    
    def _tiled_matmul(self, A, B, C, tile_size):
        """타일 단위로 행렬 곱셈 수행"""
        M, K = A.shape
        K, N = B.shape
        
        # 타일 크기
        TM, TN = tile_size
        TK = min(K, max(TM, TN))  # K 차원의 타일 크기
        
        # 각 차원의 타일 수
        num_tiles_M = (M + TM - 1) // TM
        num_tiles_N = (N + TN - 1) // TN
        num_tiles_K = (K + TK - 1) // TK
        
        # 타일 단위로 곱셈 수행
        for i in range(num_tiles_M):
            for j in range(num_tiles_N):
                # 결과 타일 범위
                i_start = i * TM
                i_end = min(i_start + TM, M)
                j_start = j * TN
                j_end = min(j_start + TN, N)
                
                # 타일 곱셈 결과 누적
                for k in range(num_tiles_K):
                    k_start = k * TK
                    k_end = min(k_start + TK, K)
                    
                    # 타일 추출
                    A_tile = A[i_start:i_end, k_start:k_end]
                    B_tile = B[k_start:k_end, j_start:j_end]
                    
                    # 컴퓨트 커널 호출 (또는 직접 계산)
                    if self.compute_kernel:
                        args = {
                            'input_arrays': (A_tile, B_tile),
                            'output_shape': (A_tile.shape[0], B_tile.shape[1])
                        }
                        result = self.compute_kernel.execute_kernel('matmul', args)
                        C[i_start:i_end, j_start:j_end] += result
                    else:
                        C[i_start:i_end, j_start:j_end] += np.matmul(A_tile, B_tile)
        
        # 트레이스 기록
        trace_manager.record_call(
            self.__class__.__name__, 
            '_tiled_matmul', 
            {'A.shape': A.shape, 'B.shape': B.shape, 'tile_size': tile_size}, 
            {}, 
            {'C.shape': C.shape}
        )
    
    def _get_tensor_data(self, tensor):
        """텐서 데이터 가져오기 (주소 또는 직접 데이터)"""
        if isinstance(tensor, int):  # 주소인 경우
            # 메모리 관리자를 통해 데이터 읽기
            # 실제 구현에서는 크기와 형태 정보도 필요
            return self.memory_manager.read_memory(address=tensor)
        else:  # 직접 데이터인 경우
            return tensor
    
    def _store_tensor_data(self, data, address):
        """텐서 데이터를 메모리에 저장"""
        if isinstance(address, int):
            self.memory_manager.write_memory(data, address=address)
    
    def set_default_tile_size(self, tile_size):
        """기본 타일 크기 설정"""
        self.default_tile_size = tile_size 