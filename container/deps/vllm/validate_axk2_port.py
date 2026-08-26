# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from vllm.model_executor.models.registry import ModelRegistry
from vllm.transformers_utils.configs.axk2 import AXK2Config


def main() -> None:
    config = AXK2Config()
    assert config.model_type == "axk2"
    assert "AXK2ForCausalLM" in ModelRegistry.get_supported_archs()

    registered = ModelRegistry.models["AXK2ForCausalLM"]
    model_cls = registered.load_model_cls()
    assert model_cls is not None
    assert model_cls.__name__ == "AXK2ForCausalLM"
    print("AXK2_PORT_WIRING_OK")


if __name__ == "__main__":
    main()
