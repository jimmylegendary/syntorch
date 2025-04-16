import syntorch.torch.functional as F
from syntorch.core.trace import SyntorchLayer, trace_manager
from syntorch.torch.module import Module, Parameter
from syntorch.torch.tensor import Tensor, cat, zeros
import syntorch.torch.nn.functional as nnF
from syntorch.parallel.comm import all_reduce, all_gather, broadcast


class ColumnParallelLinear(Module):
    """컬럼 방향으로 병렬화된 선형 레이어"""

    def __init__(
        self, input_size, output_size, bias=True, gather_output=True, tp_size=1, tp_rank=0
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        self.gather_output = gather_output

        # 병렬 처리 로직에 따라 실제 출력 크기 계산
        self.output_size_per_partition = output_size // tp_size

        # 가중치 초기화 (각 GPU는 전체 가중치의 일부만 저장)
        weight_shape = (self.output_size_per_partition, input_size)
        self.weight = Parameter(Tensor(weight_shape))
        self.weight.normal_(0, 0.02)

        if bias:
            bias_shape = (self.output_size_per_partition,)
            self.bias = Parameter(Tensor(bias_shape))
            self.bias.zero_()
        else:
            self.bias = None

    def forward(self, input_):
        """순전파"""
        # 로컬 영역에 대한 선형 변환 수행
        output = nnF.linear(input_, self.weight, self.bias)

        # 결과 수집 (gather)
        if self.gather_output and self.tp_size > 1:
            # all_gather 통신 연산 수행
            output = all_gather(output, group_size=self.tp_size, group_rank=self.tp_rank)

        return output


class RowParallelLinear(Module):
    """행 방향으로 병렬화된 선형 레이어"""

    def __init__(
        self, input_size, output_size, bias=True, input_is_parallel=False, tp_size=1, tp_rank=0
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        self.input_is_parallel = input_is_parallel

        # 병렬 처리 로직에 따라 실제 입력 크기 계산
        self.input_size_per_partition = input_size // tp_size

        # 가중치 초기화 (각 GPU는 전체 가중치의 일부만 저장)
        if input_is_parallel:
            weight_shape = (output_size, self.input_size_per_partition)
        else:
            weight_shape = (output_size, input_size)

        self.weight = Parameter(Tensor(weight_shape))
        self.weight.normal_(0, 0.02)

        if bias:
            # 바이어스는 마지막 병렬 계산 후에만 적용되므로 랭크 0에서만 초기화
            if tp_rank == 0 or not input_is_parallel:
                bias_shape = (output_size,)
                self.bias = Parameter(Tensor(bias_shape))
                self.bias.zero_()
            else:
                self.bias = None
        else:
            self.bias = None

    def forward(self, input_):
        """순전파"""
        # 입력이 이미 병렬화되어 있는지 확인
        if self.input_is_parallel:
            # 로컬 영역에 대한 선형 변환 수행
            output = nnF.linear(input_, self.weight, None)

            # 결과 전체 합산 (all-reduce)
            if self.tp_size > 1:
                output = all_reduce(
                    output, op="sum", group_size=self.tp_size, group_rank=self.tp_rank
                )

            # 바이어스 추가 (랭크 0에서만)
            if self.tp_rank == 0 and self.bias is not None:
                output = output + self.bias
                # 바이어스가 적용된 텐서를 다른 랭크에 브로드캐스트
                if self.tp_size > 1:
                    output = broadcast(
                        output, src_rank=0, group_size=self.tp_size, group_rank=self.tp_rank
                    )
        else:
            # 표준 선형 변환
            output = nnF.linear(input_, self.weight, self.bias)

        return output


class ParallelMultiheadAttention(Module):
    """병렬화된 멀티헤드 어텐션 레이어"""

    def __init__(self, embed_dim, num_heads, dropout=0.0, bias=True, tp_size=1, tp_rank=0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout

        # TP 설정
        self.tp_size = tp_size
        self.tp_rank = tp_rank

        # 헤드당 차원 계산
        self.head_dim = embed_dim // num_heads

        # TP 당 헤드 수 계산
        self.num_heads_per_partition = num_heads // tp_size

        # Q, K, V 프로젝션 (컬럼 병렬)
        self.q_proj = ColumnParallelLinear(
            embed_dim, embed_dim, bias=bias, gather_output=False, tp_size=tp_size, tp_rank=tp_rank
        )
        self.k_proj = ColumnParallelLinear(
            embed_dim, embed_dim, bias=bias, gather_output=False, tp_size=tp_size, tp_rank=tp_rank
        )
        self.v_proj = ColumnParallelLinear(
            embed_dim, embed_dim, bias=bias, gather_output=False, tp_size=tp_size, tp_rank=tp_rank
        )

        # 출력 프로젝션 (행 병렬)
        self.out_proj = RowParallelLinear(
            embed_dim,
            embed_dim,
            bias=bias,
            input_is_parallel=True,
            tp_size=tp_size,
            tp_rank=tp_rank,
        )

    def forward(self, query, key, value, attn_mask=None):
        """순전파"""
        # 배치 크기와 시퀀스 길이
        batch_size = query.shape[0]
        tgt_len = query.shape[1]
        src_len = key.shape[1]

        # 프로젝션 수행 (각 TP는 일부 헤드만 처리)
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        # 헤드 분할 - 실제 구현
        # 각 TP 랭크는 전체 헤드의 일부만 담당
        # q, k, v 형태: [batch_size, seq_len, num_heads_per_partition * head_dim]

        # 1. 헤드 차원으로 reshape
        q = self._reshape_for_multihead(q, batch_size, tgt_len)
        k = self._reshape_for_multihead(k, batch_size, src_len)
        v = self._reshape_for_multihead(v, batch_size, src_len)

        # q, k, v 형태: [batch_size, num_heads_per_partition, seq_len, head_dim]

        # 어텐션 계산
        # 스케일링
        q = q * (self.head_dim**-0.5)

        # QK^T 계산 - 실제 구현
        # q: [batch_size, num_heads_per_partition, tgt_len, head_dim]
        # k: [batch_size, num_heads_per_partition, src_len, head_dim]
        # 결과: [batch_size, num_heads_per_partition, tgt_len, src_len]
        attn_weights = self._compute_attention_scores(q, k)

        # 마스크 적용 (필요시)
        if attn_mask is not None:
            # 마스크 형태에 맞게 브로드캐스팅
            # 마스크: [1, 1, tgt_len, src_len] 또는 [batch_size, 1, tgt_len, src_len]
            if attn_mask.ndim == 2:
                attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)
            elif attn_mask.ndim == 3:
                attn_mask = attn_mask.unsqueeze(1)

            attn_weights = attn_weights + attn_mask

        # Softmax
        attn_weights = nnF.softmax(attn_weights, dim=-1)

        # Dropout
        attn_weights = nnF.dropout(attn_weights, self.dropout, self.training)

        # 가중치 합산 - 실제 구현
        # attn_weights: [batch_size, num_heads_per_partition, tgt_len, src_len]
        # v: [batch_size, num_heads_per_partition, src_len, head_dim]
        # 결과: [batch_size, num_heads_per_partition, tgt_len, head_dim]
        attn_output = self._compute_attention_output(attn_weights, v)

        # 헤드 결합 - 실제 구현
        # [batch_size, num_heads_per_partition, tgt_len, head_dim] ->
        # [batch_size, tgt_len, num_heads_per_partition * head_dim]
        attn_output = self._reshape_from_multihead(attn_output, batch_size, tgt_len)

        # 출력 프로젝션
        attn_output = self.out_proj(attn_output)

        return attn_output

    def _reshape_for_multihead(self, x, batch_size, seq_len):
        """텐서를 멀티헤드 어텐션 형태로 재구성"""
        # x: [batch_size, seq_len, num_heads_per_partition * head_dim]
        num_heads = self.num_heads_per_partition
        head_dim = self.head_dim

        # 먼저 3차원으로 재구성
        # [batch_size, seq_len, num_heads_per_partition, head_dim]
        x_reshaped = F.reshape(x, (batch_size, seq_len, num_heads, head_dim))

        # 차원 순서 변경: [batch_size, num_heads_per_partition, seq_len, head_dim]
        # 0-2, 1-3 차원 교환 필요
        x_transposed = F.transpose(x_reshaped, (0, 2, 1, 3))

        return x_transposed

    def _reshape_from_multihead(self, x, batch_size, seq_len):
        """멀티헤드 어텐션 결과를 원래 형태로 재구성"""
        # x: [batch_size, num_heads_per_partition, seq_len, head_dim]
        num_heads = self.num_heads_per_partition
        head_dim = self.head_dim

        # 차원 순서 변경: [batch_size, seq_len, num_heads_per_partition, head_dim]
        # 0-2, 1-3 차원 교환 필요
        x_transposed = F.transpose(x, (0, 2, 1, 3))

        # 마지막 두 차원 병합
        # [batch_size, seq_len, num_heads_per_partition * head_dim]
        x_merged = F.reshape(x_transposed, (batch_size, seq_len, num_heads * head_dim))

        return x_merged

    def _compute_attention_scores(self, q, k):
        """어텐션 점수 계산 (QK^T)"""
        # q: [batch_size, num_heads_per_partition, tgt_len, head_dim]
        # k: [batch_size, num_heads_per_partition, src_len, head_dim]
        # 결과: [batch_size, num_heads_per_partition, tgt_len, src_len]

        batch_size = q.shape[0]
        num_heads = q.shape[1]
        tgt_len = q.shape[2]
        src_len = k.shape[2]

        # k를 전치하여 행렬 곱 계산 (마지막 두 차원)
        scores_list = []

        # 배치 및 헤드에 대해 반복
        for b in range(batch_size):
            head_scores_list = []
            for h in range(num_heads):
                # q_bh: [tgt_len, head_dim]
                # k_bh: [src_len, head_dim]
                q_bh = q[b, h]
                k_bh = k[b, h]

                # 전치 및 행렬 곱 계산 - F.transpose 사용(차원 순서 tuple 지정)
                k_bh_transposed = F.transpose(k_bh, (1, 0))

                # 행렬 곱 계산
                score_bh = q_bh.matmul(k_bh_transposed)
                head_scores_list.append(score_bh)

            # 헤드 차원 결합
            batch_scores = cat(head_scores_list, dim=0)
            batch_scores = F.reshape(batch_scores, (num_heads, tgt_len, src_len))
            scores_list.append(batch_scores)

        # 배치 차원 결합
        scores = cat(scores_list, dim=0)
        scores = F.reshape(scores, (batch_size, num_heads, tgt_len, src_len))

        return scores

    def _compute_attention_output(self, attn_weights, v):
        """어텐션 출력 계산 (Attention(Q,K,V) = softmax(QK^T)V)"""
        # attn_weights: [batch_size, num_heads_per_partition, tgt_len, src_len]
        # v: [batch_size, num_heads_per_partition, src_len, head_dim]
        # 결과: [batch_size, num_heads_per_partition, tgt_len, head_dim]

        batch_size = attn_weights.shape[0]
        num_heads = attn_weights.shape[1]
        tgt_len = attn_weights.shape[2]
        head_dim = v.shape[3]

        output_list = []

        # 배치 및 헤드에 대해 반복
        for b in range(batch_size):
            head_output_list = []
            for h in range(num_heads):
                # attn_weights_bh: [tgt_len, src_len]
                # v_bh: [src_len, head_dim]
                attn_weights_bh = attn_weights[b, h]
                v_bh = v[b, h]

                # 행렬 곱 계산
                output_bh = attn_weights_bh.matmul(v_bh)
                head_output_list.append(output_bh)

            # 헤드 차원 결합
            batch_output = cat(head_output_list, dim=0)
            batch_output = F.reshape(batch_output, (num_heads, tgt_len, head_dim))
            output_list.append(batch_output)

        # 배치 차원 결합
        output = cat(output_list, dim=0)
        output = F.reshape(output, (batch_size, num_heads, tgt_len, head_dim))

        return output
