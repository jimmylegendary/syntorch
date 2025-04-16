#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
from tokenizer import GPT2Tokenizer


def main():
    """토크나이저 테스트"""
    # 설정 파일 경로 확인
    vocab_path = "configs/gpt2_vocab.json"
    merges_path = "configs/gpt2_merges.txt"

    print(f"설정 파일 확인:")
    print(f"- 현재 디렉토리: {os.getcwd()}")
    print(f"- 어휘 파일: {os.path.abspath(vocab_path)}, 존재: {os.path.exists(vocab_path)}")
    print(f"- 병합 파일: {os.path.abspath(merges_path)}, 존재: {os.path.exists(merges_path)}")

    # 토크나이저 초기화
    tokenizer = GPT2Tokenizer(vocab_path, merges_path)

    # 테스트 텍스트
    test_texts = ["Hello, my name is", "World", "Hello, World!", "12345"]

    print("\n토크나이저 테스트 결과:")
    print("=" * 50)

    for text in test_texts:
        try:
            # 인코딩
            token_ids = tokenizer.encode(text)

            # 디코딩
            decoded_text = tokenizer.decode(token_ids)

            print(f"\n입력 텍스트: '{text}'")
            print(f"토큰 ID: {token_ids}")
            print(f"디코딩된 텍스트: '{decoded_text}'")

        except Exception as e:
            print(f"\n입력 텍스트: '{text}' 처리 중 오류: {e}")

    print("\n완료!")


if __name__ == "__main__":
    main()
