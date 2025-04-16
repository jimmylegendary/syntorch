from syntorch.core.trace import SyntorchLayer
from syntorch.hw.base import HWComponent, HWGroup
import json

class DeviceDriver(SyntorchLayer):
    """디바이스 드라이버"""
    
    def __init__(self):
        self.devices = {}  # 등록된 디바이스 맵
        self.hw_config = None  # HW 구성 저장
    
    def load_hw_config(self, config_file=None, hw_config=None):
        """HW 설정 로드"""
        if config_file:
            # JSON 파일에서 설정 로드
            with open(config_file, 'r') as f:
                self.hw_config = json.load(f)
        elif hw_config:
            self.hw_config = hw_config
        else:
            raise ValueError("설정 파일 또는 HW 설정 객체가 필요합니다")
        
        # 디바이스 초기화
        self._initialize_devices()
    
    def _initialize_devices(self):
        """HW 설정에서 디바이스 초기화"""
        if not self.hw_config:
            return
        
        # 컴포넌트 생성 (간단한 구현)
        # 실제로는 더 복잡한 로직이 필요할 수 있음
        from syntorch.hw.memory import MemoryComponent
        from syntorch.hw.compute import ComputeComponent
        from syntorch.hw.network import NetworkComponent
        
        components = self.hw_config.get('components', {})
        self.devices = {}
        
        # 먼저 모든 컴포넌트 생성
        for comp_id, comp_info in components.items():
            comp_type = comp_info.get('type', '')
            
            if comp_type == 'memory':
                self.devices[comp_id] = MemoryComponent(
                    comp_id,
                    comp_info.get('start_address', 0),
                    comp_info.get('size', 0),
                    comp_info.get('metadata', {})
                )
            elif comp_type == 'compute':
                self.devices[comp_id] = ComputeComponent(
                    comp_id,
                    comp_info.get('metadata', {})
                )
            elif comp_type == 'network':
                self.devices[comp_id] = NetworkComponent(
                    comp_id,
                    comp_info.get('metadata', {})
                )
        
        # 그룹 생성 및 컴포넌트 연결
        for comp_id, comp_info in components.items():
            if comp_info.get('type') == 'group':
                group = HWGroup(comp_id, [], comp_info.get('metadata', {}))
                
                # 그룹에 컴포넌트 추가
                for member_id in comp_info.get('components', []):
                    if member_id in self.devices:
                        group.add_component(self.devices[member_id])
                
                self.devices[comp_id] = group
        
        # 연결 설정
        for connection in self.hw_config.get('connections', []):
            src_id = connection.get('source')
            dst_id = connection.get('target')
            # 연결 처리 (필요시 구현)
    
    def get_device(self, device_id):
        """디바이스 ID로 디바이스 객체 검색"""
        if device_id in self.devices:
            return self.devices[device_id]
        return None
    
    def get_device_by_type(self, device_type):
        """지정된 유형의 모든 디바이스 검색"""
        return [dev for dev in self.devices.values() 
                if getattr(dev, 'type', None) == device_type]
    
    def get_all_devices(self):
        """모든 등록된 디바이스 반환"""
        return self.devices 