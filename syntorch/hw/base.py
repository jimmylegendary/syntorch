from abc import abstractmethod
from syntorch.core.trace import SyntorchLayer

class HWComponent(SyntorchLayer):
    """하드웨어 컴포넌트 기본 클래스"""
    
    def __init__(self, component_id, metadata=None):
        self.component_id = component_id
        self.metadata = metadata or {}
    
    @abstractmethod
    def get_capabilities(self):
        """컴포넌트 기능 정보 반환"""
        pass

class HWGroup(HWComponent):
    """하드웨어 컴포넌트 그룹 (GPU, SM 등의 논리적 그룹)"""
    
    def __init__(self, group_id, components=None, metadata=None):
        super().__init__(group_id, metadata)
        self.components = components or []
    
    def add_component(self, component):
        """컴포넌트 추가"""
        self.components.append(component)
    
    def get_capabilities(self):
        """그룹 내 모든 컴포넌트의 기능 정보 반환"""
        capabilities = {}
        for component in self.components:
            capabilities[component.component_id] = component.get_capabilities()
        return capabilities
    
    def get_component(self, component_id):
        """ID로 컴포넌트 검색"""
        for component in self.components:
            if component.component_id == component_id:
                return component
            if isinstance(component, HWGroup):
                found = component.get_component(component_id)
                if found:
                    return found
        return None 