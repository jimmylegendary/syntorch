import json
import os
import re  # 표준 re 모듈 사용
from typing import List, Dict, Union, Optional


class GPT2Tokenizer:
    """GPT-2 토크나이저 간단 구현"""

    def __init__(self, vocab_file, merges_file):
        """
        Args:
            vocab_file: 어휘 파일 경로 (json)
            merges_file: 병합 규칙 파일 경로
        """
        if not os.path.exists(vocab_file):
            print(f"경고: 어휘 파일이 존재하지 않습니다: {vocab_file}")
            # 임시 어휘 사전 생성
            self.encoder = {f"<{i}>": i for i in range(1000)}
            self.decoder = {i: f"<{i}>" for i in range(1000)}
        else:
            with open(vocab_file, "r", encoding="utf-8") as f:
                self.encoder = json.load(f)
            self.decoder = {v: k for k, v in self.encoder.items()}

        if not os.path.exists(merges_file):
            print(f"경고: 병합 파일이 존재하지 않습니다: {merges_file}")
            self.bpe_ranks = {}
        else:
            with open(merges_file, "r", encoding="utf-8") as f:
                bpe_merges = f.read().split("\n")[1:-1]
            self.bpe_ranks = {tuple(merge.split()): i for i, merge in enumerate(bpe_merges)}

        # 특수 토큰 정의
        self.special_tokens = {"<|endoftext|>": 50256, "<|pad|>": 50257, "<|eos|>": 50258}

        # 인코더와 디코더에 특수 토큰 추가
        for token, idx in self.special_tokens.items():
            self.encoder[token] = idx
            self.decoder[idx] = token

    def encode(self, text: str) -> List[int]:
        """텍스트를 토큰 ID로 인코딩 (단순 구현)"""
        # 단어 단위로 분할하는 매우 간단한 토크나이저
        words = text.split()

        # 단어를 ID로 매핑 (단어가 사전에 없으면 문자 단위로 분할)
        result = []
        for word in words:
            if word in self.encoder:
                result.append(self.encoder[word])
            else:
                # 단어가 사전에 없으면 각 문자를 별도 토큰으로 처리
                for char in word:
                    char_id = self.encoder.get(char, 0)  # 0 = <unk>
                    result.append(char_id)

        return result

    def decode(self, token_ids: List[int]) -> str:
        """토큰 ID를 텍스트로 디코딩 (단순 구현)"""
        tokens = []
        for token_id in token_ids:
            token = self.decoder.get(token_id, "<unk>")
            tokens.append(token)

        # 특수 토큰이 아닌 경우에만 공백 추가
        text = ""
        for token in tokens:
            if token in self.special_tokens.keys():
                text += token
            else:
                # 기본 문자인 경우 그대로 추가
                text += token

        return text
