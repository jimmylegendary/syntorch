from syntorch.core.trace import SyntorchLayer, trace_manager
from syntorch.hw.memory import MemoryComponent
import json

class MemoryManager(SyntorchLayer):
    """메모리 관리 시스템"""
    
    def __init__(self):
        self.memory_map = {}  # 주소-크기 맵핑
        self.allocated_regions = {}  # 할당된 메모리 영역 (address: (size, allocated))
        self.key_to_address = {}  # 키-주소 맵핑
        self.hw_memories = {}  # HW 메모리 컴포넌트 매핑
    
    def init_memory_map(self, hw_config):
        """HW 설정에서 메모리 맵 초기화"""
        if isinstance(hw_config, str):
            # JSON 파일에서 설정 로드
            with open(hw_config, 'r') as f:
                config = json.load(f)
        else:
            config = hw_config
        
        for comp_id, comp_info in config.get('components', {}).items():
            if comp_info.get('type') == 'memory':
                start_addr = comp_info.get('start_address', 0)
                size = comp_info.get('size', 0)
                self.memory_map[start_addr] = size
                self.allocated_regions[start_addr] = (size, False)  # (크기, 할당여부)
                
                # HW 메모리 컴포넌트 생성
                self.hw_memories[start_addr] = MemoryComponent(
                    comp_id, 
                    start_addr, 
                    size, 
                    comp_info.get('metadata', {})
                )
    
    def allocate(self, size, device=None):
        """연속적인 메모리 영역 할당"""
        # 연속된 메모리 영역 찾기
        for start_addr, (region_size, allocated) in self.allocated_regions.items():
            if not allocated and region_size >= size:
                # 할당 가능한 영역 찾음
                self.allocated_regions[start_addr] = (size, True)  # 사용 중으로 표시
                
                # 남은 공간 관리
                if region_size > size:
                    new_addr = start_addr + size
                    self.allocated_regions[new_addr] = (region_size - size, False)
                
                trace_manager.trace_memory(start_addr, size, 'allocate')
                return start_addr
        
        raise MemoryError(f"크기 {size}의 연속적인 메모리를 할당할 수 없습니다")
    
    def free(self, address):
        """메모리 해제"""
        if address not in self.allocated_regions or not self.allocated_regions[address][1]:
            raise ValueError(f"주소 {address}는 할당되지 않았습니다")
        
        size, _ = self.allocated_regions[address]
        self.allocated_regions[address] = (size, False)  # 사용 가능으로 표시
        
        trace_manager.trace_memory(address, size, 'free')
        
        # 인접한 빈 영역 병합 (간단한 구현)
        self._merge_free_regions()
    
    def _merge_free_regions(self):
        """인접한 빈 메모리 영역 병합"""
        # 주소 순서로 정렬
        addresses = sorted(self.allocated_regions.keys())
        
        i = 0
        while i < len(addresses) - 1:
            curr_addr = addresses[i]
            next_addr = addresses[i + 1]
            
            curr_size, curr_allocated = self.allocated_regions[curr_addr]
            next_size, next_allocated = self.allocated_regions[next_addr]
            
            # 두 영역이 모두 사용 가능하고 인접해 있으면 병합
            if not curr_allocated and not next_allocated and curr_addr + curr_size == next_addr:
                # 병합
                self.allocated_regions[curr_addr] = (curr_size + next_size, False)
                del self.allocated_regions[next_addr]
                addresses.pop(i + 1)
            else:
                i += 1
    
    def register_tensor(self, tensor_key, address, size):
        """텐서 키와 메모리 주소 연결"""
        self.key_to_address[tensor_key] = (address, size)
    
    def read_memory(self, key=None, address=None, size=None):
        """메모리에서 데이터 읽기"""
        if key is not None:
            if key not in self.key_to_address:
                raise KeyError(f"키 {key}가 등록되지 않았습니다")
            address, size = self.key_to_address[key]
        
        if address is None:
            raise ValueError("주소 또는 키가 필요합니다")
        
        # HW 메모리 컴포넌트에서 데이터 읽기
        for mem_start, mem_obj in self.hw_memories.items():
            if mem_start <= address < mem_start + mem_obj.size:
                # 메모리 컴포넌트에 상대적인 오프셋 계산
                offset = address - mem_start
                data = mem_obj.read(offset, size)
                trace_manager.trace_memory(address, size, 'read', data)
                return data
        
        raise ValueError(f"주소 {address}가 어떤 메모리 범위에도 속하지 않습니다")
    
    def write_memory(self, data, key=None, address=None):
        """메모리에 데이터 쓰기"""
        if key is not None:
            if key not in self.key_to_address:
                raise KeyError(f"키 {key}가 등록되지 않았습니다")
            address, _ = self.key_to_address[key]
        
        if address is None:
            raise ValueError("주소 또는 키가 필요합니다")
        
        data_size = len(data)
        
        # HW 메모리 컴포넌트에 데이터 쓰기
        for mem_start, mem_obj in self.hw_memories.items():
            if mem_start <= address < mem_start + mem_obj.size:
                # 메모리 컴포넌트에 상대적인 오프셋 계산
                offset = address - mem_start
                mem_obj.write(offset, data)
                trace_manager.trace_memory(address, data_size, 'write')
                return True
        
        raise ValueError(f"주소 {address}가 어떤 메모리 범위에도 속하지 않습니다") 