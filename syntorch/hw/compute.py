from syntorch.hw.base import HWComponent
from syntorch.core.trace import trace_manager
import time

class ComputeComponent(HWComponent):
    """컴퓨트 하드웨어 컴포넌트"""
    
    def __init__(self, component_id, metadata=None):
        super().__init__(component_id, metadata)
        self.ops_supported = metadata.get('ops_supported', [])
        self.performance = metadata.get('performance', {})
    
    def execute(self, op_name, arguments):
        """연산 실행"""
        if op_name not in self.ops_supported:
            raise ValueError(f"연산 {op_name}은(는) {self.component_id} 컴포넌트에서 지원되지 않습니다")
        
        # 간단한 시뮬레이션: 연산 시간 추정
        start_time = time.time()
        
        # 연산 시뮬레이션 로직
        result = self._simulate_operation(op_name, arguments)
        
        duration = time.time() - start_time
        trace_manager.trace_hw(self.component_id, op_name, duration, arguments)
        
        return result
    
    def _simulate_operation(self, op_name, arguments):
        """연산 시뮬레이션"""
        # 성능 정보를 바탕으로 계산 시뮬레이션
        if op_name == 'matmul':
            # 행렬 곱셈 시뮬레이션
            A, B = arguments.get('input_arrays', (None, None))
            if A is not None and B is not None:
                # 실제 계산 수행하거나 결과 예측
                import numpy as np
                return np.matmul(A, B)
        elif op_name == 'add':
            # 덧셈 시뮬레이션
            A, B = arguments.get('input_arrays', (None, None))
            if A is not None and B is not None:
                return A + B
        
        # 기본 반환 값
        return None
    
    def get_capabilities(self):
        """컴퓨트 컴포넌트 기능 정보"""
        return {
            'type': 'compute',
            'ops_supported': self.ops_supported,
            'performance': self.performance
        } 