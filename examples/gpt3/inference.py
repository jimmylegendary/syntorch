import argparse
import os
import sys

# 프로젝트 루트 디렉토리를 시스템 경로에 추가
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from model import GPT3Config, GPT3ForCausalLM
import syntorch
import syntorch.torch as torch
from tokenizer import GPT2Tokenizer


def main():
    try:
        # 명령줄 인수 파싱
        parser = argparse.ArgumentParser(description="GPT3 모델 실행")
        parser.add_argument("--num_input_tokens", type=int, default=5, help="입력 토큰 수")
        parser.add_argument("--num_output_tokens", type=int, default=1, help="출력 토큰 수")
        parser.add_argument(
            "--hw_config",
            type=str,
            default="configs/hardware.json",
            help="하드웨어 설정 파일 경로",
        )
        parser.add_argument(
            "--model_config",
            type=str,
            default="configs/model_config.json",
            help="모델 설정 파일 경로",
        )
        parser.add_argument("--tp", type=int, default=4, help="텐서 병렬 크기")
        parser.add_argument(
            "--device",
            type=str,
            default="cpu",
            choices=["cpu", "cuda"],
            help="실행 디바이스",
        )
        parser.add_argument("--temperature", type=float, default=0.7, help="생성 온도")
        parser.add_argument("--top_k", type=int, default=50, help="상위 k개의 토큰 선택")
        parser.add_argument("--top_p", type=float, default=0.9, help="상위 p%의 토큰 선택")
        parser.add_argument("--visualize_trace", action="store_true", help="트레이스 그래프 시각화")
        parser.add_argument("--debug", action="store_true", help="디버그 정보 출력")
        args = parser.parse_args()

        # 디버그 모드
        if args.debug:
            print("\n[디버그] 파일 경로 확인:")
            import os

            print(f"- 현재 작업 디렉토리: {os.getcwd()}")
            print(
                f"- 모델 설정 파일: {os.path.abspath(args.model_config)}, 존재: {os.path.exists(args.model_config)}"
            )
            print(
                f"- 하드웨어 설정 파일: {os.path.abspath(args.hw_config)}, 존재: {os.path.exists(args.hw_config)}"
            )
            vocab_path = "configs/gpt2_vocab.json"
            merges_path = "configs/gpt2_merges.txt"
            print(f"- 어휘 파일: {os.path.abspath(vocab_path)}, 존재: {os.path.exists(vocab_path)}")
            print(
                f"- 병합 파일: {os.path.abspath(merges_path)}, 존재: {os.path.exists(merges_path)}"
            )

        # 하드웨어 초기화
        hw = initialize_hardware(args.hw_config)

        # 토크나이저 초기화
        tokenizer = GPT2Tokenizer("configs/gpt2_vocab.json", "configs/gpt2_merges.txt")

        # 모델 설정 로드
        config = GPT3Config.from_json(args.model_config)
        config.tp_size = args.tp  # 텐서 병렬 크기 설정
        config.device = args.device

        # 모델 초기화
        model = GPT3ForCausalLM(config)

        # 입력 텍스트 토큰화
        prompt = "Hello, my name is"
        input_ids = tokenizer.encode(prompt)
        input_ids = torch.tensor([input_ids])

        # Syntorch 프레임워크 정보 출력
        print(f"\nSyntorch Framework Demo")
        print(f"------------------------")
        print(f"입력 프롬프트: '{prompt}'")
        print(f"입력 토큰 수: {args.num_input_tokens}")
        print(f"출력 토큰 수: {args.num_output_tokens}")
        print(f"디바이스: {args.device}")
        print(f"텐서 병렬 크기: {args.tp}")

        # 추론 실행
        print("\n추론 실행 중...")
        with torch.no_grad():
            logits = model(input_ids)

        # 텐서 연산 트레이스 출력
        print("\n연산 트레이스:")
        syntorch.trace_manager.dump(format="console")

        # 텍스트 생성
        print("\n생성 시작...")
        generated_ids = model.generate(
            input_ids,
            max_length=args.num_input_tokens + args.num_output_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
        )
        generated_tokens = tokenizer.decode(generated_ids[0, args.num_input_tokens :].data)

        print(f"\n생성된 텍스트:\n{generated_tokens}")

        # Trace 그래프 시각화
        if args.visualize_trace:
            from syntorch.core.trace import trace_manager

            print("\nTrace 그래프 시각화 중...")
            # 일반 그래프 시각화
            trace_manager.visualize_graph(filename="tensor_trace_graph.png", show=False)
            # 통신 연산이 강조된 그래프 시각화
            trace_manager.visualize_comm_graph(filename="tensor_comm_graph.png", show=False)
            # 다양한 형식으로 내보내기
            trace_manager.export_graph(format="json", filepath="tensor_trace_graph.json")
            trace_manager.export_graph(format="chakra", filepath="tensor_trace_chakra.json")

            if args.tp > 1:
                print("텐서 병렬화 그래프가 tensor_trace_graph.png에 저장되었습니다.")
                print("통신 연산이 강조된 그래프가 tensor_comm_graph.png에 저장되었습니다.")
            else:
                print("텐서 연산 그래프가 tensor_trace_graph.png에 저장되었습니다.")
            print("Chakra 형식 그래프가 tensor_trace_chakra.json에 저장되었습니다.")

    except Exception as e:
        import traceback

        print(f"\n[에러] 실행 중 오류가 발생했습니다: {e}")
        traceback.print_exc()


def initialize_hardware(hw_config_path):
    """하드웨어 초기화"""
    from syntorch.os.driver import DeviceDriver
    from syntorch.os.memory import MemoryManager
    from syntorch.os.kernel import ComputeKernel

    print(f"하드웨어 설정 로드 중: {hw_config_path}")

    # 디바이스 드라이버 초기화
    driver = DeviceDriver()
    driver.load_hw_config(hw_config_path)

    # 메모리 관리자 초기화
    memory_manager = MemoryManager()
    memory_manager.init_memory_map(hw_config_path)

    # 컴퓨트 커널 초기화
    compute_kernel = ComputeKernel()
    compute_kernel.register_compute_components(driver.get_all_devices())

    print("하드웨어 초기화 완료")


if __name__ == "__main__":
    main()
