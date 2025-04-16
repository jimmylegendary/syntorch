import json
import syntorch.torch as torch
import syntorch.torch.nn as nn
from syntorch.parallel.megatron import ParallelTransformerBlock
import numpy as np


class GPT3Config:
    """GPT3 모델 설정"""

    @classmethod
    def from_json(cls, json_file):
        """JSON 파일에서 설정 로드"""
        with open(json_file, "r") as f:
            config_dict = json.load(f)

        config = cls()
        for key, value in config_dict.items():
            setattr(config, key, value)

        # 텐서 병렬화 설정 추가
        config.tp_size = 1
        config.tp_rank = 0

        return config


class GPT3Model(nn.Module):
    """GPT3 모델 구현"""

    def __init__(self, config):
        super().__init__()
        self.config = config

        # 임베딩 레이어
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.n_positions, config.n_embd)
        self.drop = nn.Dropout(config.embd_pdrop)

        # 변환기 블록 (병렬화 지원)
        self.transformer = ParallelTransformerBlock(
            num_layers=config.n_layer,
            hidden_size=config.n_embd,
            num_attention_heads=config.n_head,
            ff_size=config.n_inner,
            dropout=config.resid_pdrop,
            layer_norm_eps=config.layer_norm_epsilon,
            tp_size=config.tp_size,
            tp_rank=config.tp_rank,
        )

        # LM 헤드
        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # 가중치 초기화
        self.apply(self._init_weights)

        # 가중치 공유 (임베딩과 출력 레이어)
        self.lm_head.weight = self.wte.weight

    def _init_weights(self, module):
        """가중치 초기화 (NumPy 호환 방식)"""
        if isinstance(module, nn.Linear):
            # normal_ 대신 numpy 배열에 직접 설정
            module.weight.data = np.random.normal(
                0.0, self.config.initializer_range, module.weight.data.shape
            )
            if module.bias is not None:
                module.bias.data.fill(0)
        elif isinstance(module, nn.Embedding):
            module.weight.data = np.random.normal(
                0.0, self.config.initializer_range, module.weight.data.shape
            )
            if hasattr(module, "padding_idx") and module.padding_idx is not None:
                module.weight.data[module.padding_idx] = 0
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.fill(0)
            module.weight.data.fill(1.0)

    def forward(self, input_ids, position_ids=None, attention_mask=None):
        """모델 forward pass"""
        # 위치 ID 처리
        if position_ids is None:
            # 위치 ID 자동 생성
            batch_size, seq_length = input_ids.shape
            position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        # 임베딩
        inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds
        hidden_states = self.drop(hidden_states)

        # 트랜스포머 블록 통과
        hidden_states = self.transformer(hidden_states, attention_mask)

        # LM 헤드
        hidden_states = self.ln_f(hidden_states)
        logits = self.lm_head(hidden_states)

        return logits

    def load_pretrained_weights(self, weights_path):
        """사전 학습된 가중치 로드 (간단한 구현)"""
        # 실제 구현에서는 체크포인트 로드 로직 필요
        pass


class GPT3ForCausalLM(nn.Module):
    """자기회귀 언어 모델링을 위한 GPT3"""

    def __init__(self, config):
        super().__init__()
        self.transformer = GPT3Model(config)
        self.config = config

    def forward(self, input_ids, position_ids=None, attention_mask=None, past_key_values=None):
        """모델 forward pass"""
        # 변환기 출력 얻기
        outputs = self.transformer(input_ids, position_ids, attention_mask)

        return outputs

    def generate(self, input_ids, max_length, temperature=1.0, top_k=0, top_p=0.9):
        """텍스트 생성"""
        batch_size, input_length = input_ids.shape

        # 현재 시퀀스 초기화
        current_tokens = input_ids.clone()

        # 생성 루프
        for _ in range(max_length - input_length):
            # 모델로 다음 토큰 예측
            with torch.no_grad():
                outputs = self.forward(current_tokens)
                next_token_logits = outputs[:, -1, :]

            # 온도 조정
            if temperature > 0:
                next_token_logits = next_token_logits / temperature

            # Top-k 샘플링
            if top_k > 0:
                indices_to_remove = (
                    next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                )
                next_token_logits[indices_to_remove] = float("-inf")

            # Top-p (핵) 샘플링
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

                # 확률 합이 top_p를 넘는 토큰 제거
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                # 배치별로 처리하여 인덱스 오류 방지
                for i in range(next_token_logits.shape[0]):
                    indices_remove = sorted_indices[i][sorted_indices_to_remove[i]]
                    # numpy 인덱싱을 위해 indices_remove를 정수 배열로 변환
                    if hasattr(indices_remove, "data"):
                        indices_remove = indices_remove.data.astype(np.int32)
                    else:
                        indices_remove = np.array(indices_remove, dtype=np.int32)
                    next_token_logits[i, indices_remove] = float("-inf")

            # 다음 토큰 샘플링
            probs = torch.softmax(next_token_logits, dim=-1)
            # torch.multinomial 대신 Tensor 객체의 멤버 메서드 사용
            next_tokens = probs.multinomial(num_samples=1)

            # 시퀀스에 추가
            current_tokens = torch.cat([current_tokens, next_tokens], dim=-1)

            # EOS 토큰 확인
            if next_tokens[0, 0].item() == self.config.eos_token_id:
                break

        return current_tokens
