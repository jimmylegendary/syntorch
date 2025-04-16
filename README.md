# Syntorch

LLM 추론을 위한 합성 트레이스 생성 프레임워크

## 개요

Syntorch는 PyTorch와 100% 호환되는 인터페이스를 제공하면서 하드웨어 계층까지의 합성 트레이스를 생성할 수 있는 프레임워크입니다. 이 프레임워크는 다음 네 가지 계층으로 구성되어 있습니다:

1. **Torch 계층**: PyTorch와 클래스 이름, 함수 이름, 파라미터 등 모든 면에서 100% 호환됩니다.
2. **Runtime 계층**: 하드웨어에 맞는 타일링이나 하드웨어 기능을 사용할 수 있는 라이브러리를 추상화합니다 (cublas, cudart 등).
3. **OS 계층**: 메모리 관리, 디바이스 드라이버, 컴퓨트 커널 등 운영체제 기능을 추상화합니다.
4. **HW 계층**: 컴퓨트, 메모리, 네트워크 등의 하드웨어 컴포넌트를 추상화합니다.

## 주요 기능

- PyTorch와 100% 호환되는 API
- 메모리 할당 및 관리 시뮬레이션
- 하드웨어 구성 요소 모델링
- 자동 트레이싱 기능
- Megatron 스타일의 텐서 병렬화 및 파이프라인 병렬화 지원
- GPT3 모델링 예제 포함

## 설치

```bash
# 소스에서 설치
git clone https://github.com/username/syntorch.git
cd syntorch
pip install -e .
```

## 간단한 사용 예시

```python
import syntorch.torch as torch
import syntorch.torch.nn as nn

# 기본 텐서 생성 (PyTorch와 동일한 API)
x = torch.tensor([[1, 2], [3, 4]], device="cuda")
y = torch.tensor([[5, 6], [7, 8]], device="cuda")

# 연산 수행
z = x + y
w = x.matmul(y)

# 신경망 모델 정의
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 5)
        
    def forward(self, x):
        return self.linear(x)

# 병렬 모델 정의
from syntorch.parallel import ColumnParallelLinear

parallel_linear = ColumnParallelLinear(
    input_size=10, 
    output_size=5, 
    tp_size=2,
    tp_rank=0
)

# 트레이스 정보 출력
from syntorch import trace_manager
trace_manager.dump(format='console')
```

## 예제 실행

GPT3 모델 추론 예제를 실행하려면:

```bash
cd examples/gpt3
python inference.py --num_input_tokens 10 --num_output_tokens 20 --tp 1
```

## 하드웨어 구성 정의

Syntorch는 JSON 형식으로 하드웨어 구성을 정의할 수 있습니다:

```json
{
  "name": "Simple GPU",
  "components": {
    "gpu": {
      "type": "group",
      "components": ["sm_group", "memory_hierarchy"]
    },
    "sm0": {
      "type": "compute",
      "metadata": {
        "ops_supported": ["matmul", "add", "mul"]
      }
    },
    "hbm": {
      "type": "memory",
      "start_address": 0,
      "size": 16777216
    }
  }
}
```

## 프로젝트 구조

```
syntorch/
├── core/             # 기본 트레이싱 및 인터페이스 정의
├── torch/            # PyTorch 호환 구현
├── runtime/          # HW 기능 추상화 (tiling, library)
├── os/               # 메모리 관리, 디바이스 드라이버, 커널
├── hw/               # HW 컴포넌트 추상화
└── parallel/         # 병렬화 모듈
```

## 라이선스

MIT 라이선스

## 기여

이슈와 풀 리퀘스트를 환영합니다. 큰 변경사항은 먼저 이슈를 열어 논의해주세요.