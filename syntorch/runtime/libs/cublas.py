from syntorch.core.trace import SyntorchLayer
from syntorch.os.driver import DeviceDriver
from syntorch.os.kernel import ComputeKernel
import numpy as np

class CuBLAS(SyntorchLayer):
    """CuBLAS 라이브러리 추상화"""
    
    def __init__(self, device_driver=None, compute_kernel=None):
        self.device_driver = device_driver or DeviceDriver()
        self.compute_kernel = compute_kernel or ComputeKernel()
        self.handles = {}  # Handle 관리를 위한 딕셔너리
        self.next_handle = 1  # 다음에 할당할 handle ID
    
    def create_handle(self):
        """CuBLAS handle 생성"""
        handle = self.next_handle
        self.next_handle += 1
        self.handles[handle] = {
            'status': 'created',
            'stream': None
        }
        return handle
    
    def destroy_handle(self, handle):
        """CuBLAS handle 제거"""
        if handle in self.handles:
            del self.handles[handle]
            return 0  # 성공
        return -1  # 실패
    
    def set_stream(self, handle, stream):
        """CuBLAS handle에 스트림 설정"""
        if handle in self.handles:
            self.handles[handle]['stream'] = stream
            return 0  # 성공
        return -1  # 실패
    
    def get_stream(self, handle):
        """CuBLAS handle의 스트림 조회"""
        if handle in self.handles:
            return self.handles[handle]['stream']
        return None
    
    def sgemm(self, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc):
        """단정밀도 행렬 곱셈 (SGEMM) 구현
        
        Args:
            transa: A 행렬의 전치 여부 ('N', 'T')
            transb: B 행렬의 전치 여부 ('N', 'T')
            m, n, k: 행렬 차원
            alpha, beta: 스케일링 계수
            A, lda: A 행렬과 선행 차원
            B, ldb: B 행렬과 선행 차원
            C, ldc: C 행렬과 선행 차원
        
        Returns:
            상태 코드 (0: 성공)
        """
        # 전치 처리
        transpose_a = (transa == 'T' or transa == 't')
        transpose_b = (transb == 'T' or transb == 't')
        
        # 컴퓨트 커널을 통한 계산
        args = {
            'transpose_a': transpose_a,
            'transpose_b': transpose_b,
            'm': m, 'n': n, 'k': k,
            'alpha': alpha,
            'beta': beta,
            'A': A, 'lda': lda,
            'B': B, 'ldb': ldb,
            'C': C, 'ldc': ldc
        }
        
        try:
            self.compute_kernel.execute_kernel('sgemm', args)
            return 0  # 성공
        except Exception as e:
            print(f"CuBLAS SGEMM 오류: {e}")
            return -1  # 실패
    
    # CuBLAS API 호환 인터페이스
    def cublasSgemm(self, handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc):
        """CuBLAS SGEMM API 호환 인터페이스"""
        if handle not in self.handles:
            return -1  # 유효하지 않은 handle
        
        return self.sgemm(transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
    
    def cublasCreate(self):
        """CuBLAS handle 생성 API"""
        return self.create_handle()
    
    def cublasDestroy(self, handle):
        """CuBLAS handle 제거 API"""
        return self.destroy_handle(handle)
    
    def cublasSetStream(self, handle, stream):
        """CuBLAS 스트림 설정 API"""
        return self.set_stream(handle, stream) 