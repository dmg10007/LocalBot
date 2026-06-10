"""Tests for adapters.llamacpp_client._detect_family_from_name."""
import pytest
from localbot.adapters.llamacpp_client import _detect_family_from_name, ModelFamily


@pytest.mark.parametrize("name,expected", [
    ("Meta-Llama-3.1-8B-Instruct.Q4_K_M.gguf", ModelFamily.LLAMA),
    ("llama-2-7b-chat.Q5_K_S.gguf",            ModelFamily.LLAMA),
    ("Mistral-7B-Instruct-v0.3.Q4_K_M.gguf",   ModelFamily.MISTRAL),
    ("mixtral-8x7b-instruct.Q4_0.gguf",        ModelFamily.MISTRAL),
    ("gemma-2-9b-it.Q4_K_M.gguf",              ModelFamily.GEMMA),
    ("Qwen2.5-Coder-7B-Instruct.Q4_K_M.gguf", ModelFamily.QWEN),
    ("DeepSeek-R1-Distill-Qwen-7B.Q4_K_M.gguf", ModelFamily.DEEPSEEK),
    ("Phi-3.5-mini-instruct.Q4_K_M.gguf",      ModelFamily.PHI),
    ("totally-unknown-model-v1.gguf",           ModelFamily.UNKNOWN),
    ("",                                        ModelFamily.UNKNOWN),
])
def test_family_detection(name, expected):
    assert _detect_family_from_name(name) == expected


def test_deepseek_matched_before_qwen_for_distill():
    """DeepSeek-R1-Distill-Qwen models contain 'qwen' but must map to DEEPSEEK."""
    name = "DeepSeek-R1-Distill-Qwen-14B.Q4_K_M.gguf"
    assert _detect_family_from_name(name) == ModelFamily.DEEPSEEK
