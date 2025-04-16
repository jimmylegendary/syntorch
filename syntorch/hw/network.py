from syntorch.hw.base import HWComponent
from syntorch.core.trace import trace_manager
import time

class NetworkComponent(HWComponent):
    """네트워크 하드웨어 컴포넌트"""
    
    def __init__(self, component_id, metadata=None):
        super().__init__(component_id, metadata)
        self.bandwidth = metadata.get('bandwidth', 0)  # GB/s
        self.latency = metadata.get('latency', 0)  # ns
    
    def transfer(self, src_address, dst_address, size, src_component=None, dst_component=None):
        """데이터 전송 시뮬레이션"""
        # 전송 시간 계산 (대역폭과 지연시간 기반)
        transfer_time_ns = self.latency
        if self.bandwidth > 0:
            transfer_time_ns += (size / (self.bandwidth * 1e9)) * 1e9  # ns 단위로 변환
        
        # 전송 시뮬레이션 (실제로는 sleep)
        time.sleep(transfer_time_ns / 1e9)  # 초 단위로 변환
        
        trace_manager.trace_hw(
            self.component_id, 
            'transfer', 
            transfer_time_ns / 1e9,
            {
                'src_address': src_address,
                'dst_address': dst_address,
                'size': size,
                'src_component': src_component.component_id if src_component else None,
                'dst_component': dst_component.component_id if dst_component else None
            }
        )
        
        return True
    
    def get_capabilities(self):
        """네트워크 컴포넌트 기능 정보"""
        return {
            'type': 'network',
            'bandwidth': self.bandwidth,
            'latency': self.latency
        } 