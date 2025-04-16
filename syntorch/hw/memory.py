from syntorch.hw.base import HWComponent
import numpy as np

class MemoryComponent(HWComponent):
    """메모리 하드웨어 컴포넌트"""
    
    def __init__(self, component_id, start_address, size, metadata=None):
        super().__init__(component_id, metadata)
        self.start_address = start_address
        self.size = size
        # 단순 구현을 위해 numpy 배열 사용
        self.memory = np.zeros(size, dtype=np.uint8)
    
    def read(self, offset, size):
        """메모리 읽기"""
        if offset + size > self.size:
            raise ValueError(f"읽기 범위 초과: {offset}+{size} > {self.size}")
        return self.memory[offset:offset+size].copy()
    
    def write(self, offset, data):
        """메모리 쓰기"""
        data_length = len(data)
        if offset + data_length > self.size:
            raise ValueError(f"쓰기 범위 초과: {offset}+{data_length} > {self.size}")
        self.memory[offset:offset+data_length] = data
    
    def get_capabilities(self):
        """메모리 컴포넌트 기능 정보"""
        return {
            'type': 'memory',
            'size': self.size,
            'start_address': self.start_address,
            'latency': self.metadata.get('latency', 0),
            'bandwidth': self.metadata.get('bandwidth', 0)
        } 