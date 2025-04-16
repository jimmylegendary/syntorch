import syntorch.torch.functional as F
from syntorch.torch.module import Module
from syntorch.torch.tensor import Tensor, cat
import syntorch.torch.nn.functional as nnF
from syntorch.parallel.tensor_parallel import ColumnParallelLinear, RowParallelLinear
from syntorch.parallel.comm import all_reduce, all_gather, reduce_scatter, broadcast


class ParallelTransformerLayer(Module):
    """Megatron 스타일의 병렬 변환기 레이어"""

    def __init__(
        self,
        hidden_size,
        num_attention_heads,
        ff_size,
        dropout=0.1,
        layer_norm_eps=1e-5,
        tp_size=1,
        tp_rank=0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.ff_size = ff_size
        self.dropout = dropout
        self.tp_size = tp_size
        self.tp_rank = tp_rank

        # 1. 입력 정규화
        self.input_layernorm = self._get_layer_norm(hidden_size, layer_norm_eps)

        # 2. 셀프 어텐션 블록
        # Q, K, V 프로젝션 (컬럼 병렬)
        self.attention_qkv = ColumnParallelLinear(
            hidden_size,
            3 * hidden_size,
            bias=True,
            gather_output=False,
            tp_size=tp_size,
            tp_rank=tp_rank,
        )

        # 출력 프로젝션 (행 병렬)
        self.attention_output = RowParallelLinear(
            hidden_size,
            hidden_size,
            bias=True,
            input_is_parallel=True,
            tp_size=tp_size,
            tp_rank=tp_rank,
        )

        # 3. 포스트 어텐션 정규화
        self.post_attention_layernorm = self._get_layer_norm(hidden_size, layer_norm_eps)

        # 4. MLP 블록 (FF 네트워크)
        # FF 입력 프로젝션 (컬럼 병렬)
        self.ff1 = ColumnParallelLinear(
            hidden_size, ff_size, bias=True, gather_output=False, tp_size=tp_size, tp_rank=tp_rank
        )

        # FF 출력 프로젝션 (행 병렬)
        self.ff2 = RowParallelLinear(
            ff_size,
            hidden_size,
            bias=True,
            input_is_parallel=True,
            tp_size=tp_size,
            tp_rank=tp_rank,
        )

        # 5. 드롭아웃
        self.attention_dropout = dropout
        self.hidden_dropout = dropout

    def _get_layer_norm(self, hidden_size, eps):
        """레이어 정규화 인스턴스 생성
        (층 정규화는 병렬화하지 않음)"""
        from syntorch.torch.nn.modules.normalization import LayerNorm

        return LayerNorm(hidden_size, eps=eps)

    def forward(self, x, attention_mask=None):
        """순전파"""
        # 1. 입력 정규화
        ln_x = self.input_layernorm(x)

        # 2. 셀프 어텐션 블록
        # Q, K, V 프로젝션
        qkv = self.attention_qkv(ln_x)

        # 병렬 어텐션 계산
        attn_output = self._parallel_attention(qkv, attention_mask)

        # 출력 프로젝션
        attn_output = self.attention_output(attn_output)

        # 잔차 연결
        attn_output = F.dropout(attn_output, self.hidden_dropout, self.training)
        x = x + attn_output

        # 3. 포스트 어텐션 정규화
        ln_x = self.post_attention_layernorm(x)

        # 4. MLP 블록
        ff_output = self.ff1(ln_x)
        ff_output = F.gelu(ff_output)
        ff_output = self.ff2(ff_output)

        # 잔차 연결
        ff_output = F.dropout(ff_output, self.hidden_dropout, self.training)
        output = x + ff_output

        return output

    def _reshape_to_attention_shape(self, x, batch_size, seq_len, num_heads, head_dim):
        """텐서를 멀티헤드 어텐션 계산을 위한 형태로 변환"""
        # [batch_size, seq_len, num_heads * head_dim] ->
        # [batch_size, num_heads, seq_len, head_dim]

        # 차원 변경을 위한 reshape
        x_3d = F.reshape(x, (batch_size, seq_len, num_heads, head_dim))

        # 차원 순서 변경
        x_4d = F.transpose(x_3d, (0, 2, 1, 3))

        return x_4d

    def _reshape_from_attention_shape(self, x, batch_size, seq_len, num_heads, head_dim):
        """멀티헤드 어텐션 결과를 원래 형태로 변환"""
        # [batch_size, num_heads, seq_len, head_dim] ->
        # [batch_size, seq_len, num_heads * head_dim]

        # 차원 순서 변경
        x_transposed = F.transpose(x, (0, 2, 1, 3))

        # 마지막 두 차원 병합
        x_merged = F.reshape(x_transposed, (batch_size, seq_len, num_heads * head_dim))

        return x_merged

    def _parallel_mlp(self, ln_x):
        """병렬 MLP (FFN) 계산

        텐서 병렬화를 통해 MLP 계산을 수행
        """
        # 첫 번째 선형 변환 (컬럼 병렬)
        ff_output = self.ff1(ln_x)

        # GELU 활성화 함수
        ff_output = F.gelu(ff_output)

        # 두 번째 선형 변환 (행 병렬, all-reduce 포함)
        ff_output = self.ff2(ff_output)

        # 드롭아웃
        ff_output = F.dropout(ff_output, self.hidden_dropout, self.training)

        return ff_output

    def _parallel_attention(self, qkv, attention_mask=None):
        """병렬 셀프 어텐션 계산

        각 GPU는 전체 헤드의 일부만 처리함
        """
        # QKV를 분할하기 위한 인덱스 계산
        batch_size, seq_len = qkv.shape[0], qkv.shape[1]
        output_size_per_partition = qkv.shape[2] // 3  # 3등분 (Q, K, V)

        # QKV를 분할하기 위한 인덱스 계산
        q_start = 0
        q_end = output_size_per_partition
        k_start = output_size_per_partition
        k_end = 2 * output_size_per_partition
        v_start = 2 * output_size_per_partition
        v_end = 3 * output_size_per_partition

        # Q, K, V 분할
        q = qkv[:, :, q_start:q_end]
        k = qkv[:, :, k_start:k_end]
        v = qkv[:, :, v_start:v_end]

        # 각 랭크는 자신의 헤드 집합에 대해서만 어텐션 계산
        # 헤드 차원 계산
        head_dim = self.hidden_size // self.num_attention_heads
        num_heads_per_partition = self.num_attention_heads // self.tp_size

        # 텐서 reshape 및 transpose하여 멀티헤드 어텐션 형태로 변환
        # 차원 변경: [batch_size, seq_len, num_heads * head_dim] ->
        # [batch_size, num_heads, seq_len, head_dim]
        q_4d = self._reshape_to_attention_shape(
            q, batch_size, seq_len, num_heads_per_partition, head_dim
        )
        k_4d = self._reshape_to_attention_shape(
            k, batch_size, seq_len, num_heads_per_partition, head_dim
        )
        v_4d = self._reshape_to_attention_shape(
            v, batch_size, seq_len, num_heads_per_partition, head_dim
        )

        # 스케일링된 닷 프로덕트 어텐션 계산
        # q_4d: [batch_size, num_heads_per_partition, seq_len, head_dim]
        # k_4d: [batch_size, num_heads_per_partition, seq_len, head_dim]
        # 시퀀스 길이에 대한 어텐션 계산
        # 스케일링 적용
        scaling_factor = float(head_dim**-0.5)
        q_4d_scaled = q_4d * scaling_factor

        # 어텐션 스코어 계산을 위한 준비
        attn_scores_list = []

        # 배치 및 헤드별로 반복
        for b in range(batch_size):
            head_scores_list = []
            for h in range(num_heads_per_partition):
                # q_bh: [seq_len, head_dim]
                # k_bh: [seq_len, head_dim]
                q_bh = q_4d_scaled[b, h]
                k_bh = k_4d[b, h]

                # k_bh 전치: [seq_len, head_dim] -> [head_dim, seq_len]
                k_bh_t = F.transpose(k_bh, (1, 0))

                # 행렬 곱 계산: [seq_len, head_dim] x [head_dim, seq_len] = [seq_len, seq_len]
                score_bh = q_bh.matmul(k_bh_t)
                head_scores_list.append(score_bh)

            # 헤드 차원으로 결합
            batch_scores = cat(head_scores_list, dim=0)
            batch_scores = F.reshape(batch_scores, (num_heads_per_partition, seq_len, seq_len))
            attn_scores_list.append(batch_scores)

        # 배치 차원으로 결합
        attn_scores = cat(attn_scores_list, dim=0)
        attn_scores = F.reshape(
            attn_scores, (batch_size, num_heads_per_partition, seq_len, seq_len)
        )

        # 마스크 적용 (필요시)
        if attention_mask is not None:
            # 마스크 형태에 맞게 브로드캐스팅
            if attention_mask.ndim == 2:  # [seq_len, seq_len]
                attention_mask_reshaped = F.reshape(attention_mask, (1, 1, seq_len, seq_len))
            elif attention_mask.ndim == 3:  # [batch_size, seq_len, seq_len]
                attention_mask_reshaped = F.reshape(
                    attention_mask, (batch_size, 1, seq_len, seq_len)
                )
            else:
                attention_mask_reshaped = attention_mask

            attn_scores = attn_scores + attention_mask_reshaped

        # Softmax 적용 - nnF.softmax 사용
        attn_probs = nnF.softmax(attn_scores, dim=-1)

        # 어텐션 출력 계산을 위한 준비
        attn_output_list = []

        # 배치 및 헤드별로 반복
        for b in range(batch_size):
            head_output_list = []
            for h in range(num_heads_per_partition):
                # attn_probs_bh: [seq_len, seq_len]
                # v_bh: [seq_len, head_dim]
                attn_probs_bh = attn_probs[b, h]
                v_bh = v_4d[b, h]

                # 행렬 곱 계산: [seq_len, seq_len] x [seq_len, head_dim] = [seq_len, head_dim]
                output_bh = attn_probs_bh.matmul(v_bh)
                head_output_list.append(output_bh)

            # 헤드 차원으로 결합
            batch_output = cat(head_output_list, dim=0)
            batch_output = F.reshape(batch_output, (num_heads_per_partition, seq_len, head_dim))
            attn_output_list.append(batch_output)

        # 배치 차원으로 결합
        attn_output = cat(attn_output_list, dim=0)
        attn_output = F.reshape(
            attn_output, (batch_size, num_heads_per_partition, seq_len, head_dim)
        )

        # 텐서 재변환: [batch_size, num_heads_per_partition, seq_len, head_dim] ->
        # [batch_size, seq_len, num_heads_per_partition * head_dim]
        attn_output = self._reshape_from_attention_shape(
            attn_output, batch_size, seq_len, num_heads_per_partition, head_dim
        )

        # TP 랭크 간 all-gather 수행 (필요시)
        # 실제로는 각 TP 랭크가 일부 헤드에 대한 계산만 수행하고,
        # 모든 헤드의 결과를 모으기 위해 collective communication이 필요함
        if self.tp_size > 1:
            gather_needed = True  # 실제 구현에서는 이 값이 true여야 합니다

            if gather_needed:
                attn_output = all_gather(
                    attn_output, group_size=self.tp_size, group_rank=self.tp_rank
                )

        return attn_output


class ParallelTransformerBlock(Module):
    """Megatron 스타일의 병렬 변환기 블록 (여러 레이어 스택)"""

    def __init__(
        self,
        num_layers,
        hidden_size,
        num_attention_heads,
        ff_size,
        dropout=0.1,
        layer_norm_eps=1e-5,
        tp_size=1,
        tp_rank=0,
    ):
        super().__init__()
        self.num_layers = num_layers

        # 변환기 레이어 스택
        self.layers = [
            ParallelTransformerLayer(
                hidden_size, num_attention_heads, ff_size, dropout, layer_norm_eps, tp_size, tp_rank
            )
            for _ in range(num_layers)
        ]

        # 레이어 등록
        for i, layer in enumerate(self.layers):
            self.add_module(f"layer_{i}", layer)

        # 최종 정규화
        from syntorch.torch.nn.modules.normalization import LayerNorm

        self.final_layernorm = LayerNorm(hidden_size, eps=layer_norm_eps)

    def forward(self, x, attention_mask=None):
        """순전파"""
        # 각 레이어를 통과
        for layer in self.layers:
            x = layer(x, attention_mask)

        # 최종 정규화
        output = self.final_layernorm(x)

        return output
