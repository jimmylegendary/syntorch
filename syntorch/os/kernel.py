from syntorch.core.trace import SyntorchLayer
from syntorch.hw.compute import ComputeComponent

class ComputeKernel(SyntorchLayer):
    """컴퓨트 커널 추상화"""
    
    def __init__(self, hw_compute_components=None):
        self.compute_components = hw_compute_components or {}
    
    def register_compute_components(self, hw_config):
        """HW 구성에서 컴퓨트 컴포넌트 등록"""
        self.compute_components = {}
        
        # hw_config에서 compute 컴포넌트 찾아서 등록
        for comp_id, comp in hw_config.items():
            if isinstance(comp, ComputeComponent):
                self.compute_components[comp_id] = comp
    
    def execute_kernel(self, kernel_name, arguments, component_id=None):
        """커널 실행"""
        # 적합한 컴퓨트 컴포넌트 선택
        component = self._select_component(kernel_name, component_id)
        
        if component is None:
            raise ValueError(f"커널 {kernel_name}을 실행할 수 있는 컴포넌트가 없습니다")
        
        # 컴퓨트 컴포넌트에 실행 요청
        result = component.execute(kernel_name, arguments)
        
        return result
    
    def _select_component(self, kernel_name, component_id=None):
        """조건에 맞는 컴퓨트 컴포넌트 선택"""
        # 특정 컴포넌트 ID가 지정된 경우
        if component_id is not None:
            if component_id in self.compute_components:
                component = self.compute_components[component_id]
                # 해당 커널을 지원하는지 확인
                if kernel_name in component.ops_supported:
                    return component
                else:
                    raise ValueError(f"컴포넌트 {component_id}는 커널 {kernel_name}을 지원하지 않습니다")
            else:
                raise ValueError(f"컴포넌트 {component_id}가 등록되지 않았습니다")
        
        # 지정된 컴포넌트가 없으면 지원하는 첫 번째 컴포넌트 선택
        for comp_id, component in self.compute_components.items():
            if kernel_name in component.ops_supported:
                return component
        
        # 지원하는 컴포넌트가 없음
        return None
    
    def list_available_kernels(self):
        """사용 가능한 모든 커널 나열"""
        kernels = set()
        for component in self.compute_components.values():
            kernels.update(component.ops_supported)
        return sorted(list(kernels)) 