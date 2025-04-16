# GPT-3 모델 예제

이 예제는 Syntorch 프레임워크를 사용하여 GPT-3 모델을 구현하고 실행하는 방법을 보여줍니다.

## 필요 라이브러리 설치

```bash
pip install networkx matplotlib
```

## 실행 방법

기본 실행:

```bash
python inference.py
```

옵션을 지정하여 실행:

```bash
python inference.py --num_input_tokens 10 --num_output_tokens 20 --tp 1 --device cpu
```

## 트레이스 그래프 시각화

Syntorch 프레임워크는 모델 실행 중 텐서 연산을 추적하고 그래프로 시각화하는 기능을 제공합니다. 이 기능을 사용하려면 다음과 같이 `--visualize_trace` 옵션을 추가하세요:

```bash
python inference.py --visualize_trace
```

이 명령은 다음 파일을 생성합니다:
- `tensor_trace_graph.png`: 텐서 연산 그래프를 시각화한 이미지 파일
- `tensor_trace_graph.json`: NetworkX 그래프 데이터가 포함된 JSON 파일

### 다양한 출력 형식 지원

트레이스 그래프는 다음과 같은 다양한 형식으로 내보낼 수 있습니다:

1. **SVG 형식** - 벡터 그래픽으로 고해상도 시각화:
```bash
python run_visualization.py --input tensor_trace_graph.json --output graph.svg --format svg
```

2. **MLCommons Chakra 형식** - ML 모델 표준 형식으로 내보내기:
```bash
python run_visualization.py --input tensor_trace_graph.json --output model.json --format chakra
```

3. **GraphML 형식** - 다른 그래프 도구와의 호환성을 위해:
```bash
python run_visualization.py --input tensor_trace_graph.json --output graph.graphml --format graphml
```

## MLCommons Chakra 형식

[MLCommons Chakra](https://github.com/mlcommons/chakra)는 기계 학습 모델을 표현하기 위한 표준화된 그래프 형식입니다. Chakra 형식은 다음과 같은 특징을 가집니다:

- **표준화된 모델 표현** - 다양한 ML 프레임워크 간의 호환성
- **계층적 그래프 구조** - 노드(연산, 텐서)와 엣지(데이터 흐름)로 구성
- **풍부한 메타데이터** - 노드 유형, 텐서 형태, 데이터 유형 등의 정보 포함
- **확장성** - 다양한 ML 도구 및 분석 시스템과의 통합

Chakra 형식으로 내보낸 그래프는 MLCommons 생태계의 다양한 도구에서 사용할 수 있습니다.

## 그래프 해석 방법

생성된 그래프에서:
- 초록색 노드: 텐서 (입력/출력 데이터)
- 파란색 노드: 연산 (덧셈, 곱셈 등)
- 화살표: 데이터 흐름 방향 (입력 → 연산 → 출력)

## 옵션 설명

- `--num_input_tokens`: 입력 시퀀스의 토큰 수
- `--num_output_tokens`: 생성할 출력 토큰 수
- `--tp`: 텐서 병렬 크기
- `--device`: 실행 디바이스 (cpu 또는 cuda)
- `--temperature`: 텍스트 생성 온도 (높을수록 다양성 증가)
- `--top_k`: 샘플링할 상위 k개 토큰 수
- `--top_p`: 핵 샘플링 임계값 (0.0~1.0)
- `--visualize_trace`: 텐서 연산 그래프 시각화 활성화
- `--hw_config`: 하드웨어 설정 파일 경로
- `--model_config`: 모델 설정 파일 경로

## 텐서 병렬화와 통신 연산 시각화

Syntorch 프레임워크는 텐서 병렬화(Tensor Parallelism)를 구현하며, 이 과정에서 발생하는 집합 통신 연산을 추적하고 시각화할 수 있습니다.

### 통신 연산 지원

다음 통신 연산들을 지원합니다:

1. **all-reduce**: 각 프로세스의 값을 결합하고 모든 프로세스에 결과 배포
   - `sum`, `avg`, `min`, `max`, `prod` 등의 리덕션 연산 지원
   - 주로 행 병렬 선형 레이어에서 사용

2. **all-gather**: 각 프로세스의 텐서를 수집하여 모든 프로세스에 배포
   - 열 병렬 선형 레이어에서 사용

3. **reduce-scatter**: 각 프로세스의 입력을 결합한 후 결과를 분산
   - 특정 병렬화 패턴에서 사용

4. **broadcast**: 소스 랭크의 텐서를 모든 프로세스에 복제
   - 바이어스 등의 공유가 필요한 값에 사용

### 텐서 병렬화 실행 및 시각화

텐서 병렬화를 활성화하려면 `--tp` 인자에 1보다 큰 값을 지정하세요:

```bash
python inference.py --tp 2 --visualize_trace
```

이 명령은 다음 파일을 생성합니다:
- `tensor_trace_graph.png`: 기본 텐서 연산 그래프
- `tensor_comm_graph.png`: 통신 연산이 강조된 그래프
- `tensor_trace_chakra.json`: MLCommons Chakra 형식의 그래프

### 그래프 해석 방법

통신 연산이 강조된 그래프에서:
- 초록색 노드: 텐서 (입력/출력 데이터)
- 파란색 노드: 연산 (덧셈, 곱셈 등)
- 빨간색 노드: 통신 연산 (all-reduce, all-gather 등)
- 빨간색 엣지: 통신 관련 데이터 흐름
- 검은색 엣지: 일반 데이터 흐름 