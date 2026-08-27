# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from vllm.model_executor.models.registry import ModelRegistry
from vllm.transformers_utils.configs.axk2 import AXK2Config
from vllm.transformers_utils.configs.speculators.algos import update_dspark


def main() -> None:
    config = AXK2Config()
    assert config.model_type == "axk2"
    assert "AXK2ForCausalLM" in ModelRegistry.get_supported_archs()

    registered = ModelRegistry.models["AXK2ForCausalLM"]
    model_cls = registered.load_model_cls()
    assert model_cls is not None
    assert model_cls.__name__ == "AXK2ForCausalLM"

    # Dynamo 1.4.1's vLLM 0.26.0 base already contains the upstream DSpark
    # runtime. Verify the pieces needed by skt/A.X-K2-DSpark instead of
    # overlaying SKT's older vLLM 0.23 implementation on top of them.
    assert "Qwen3DSparkModel" in ModelRegistry.get_supported_archs()
    dspark_cls = ModelRegistry.models["Qwen3DSparkModel"].load_model_cls()
    assert dspark_cls is not None
    assert dspark_cls.__name__ == "Qwen3DSparkForCausalLM"

    converted: dict[str, object] = {}
    update_dspark(
        {
            "aux_hidden_state_layer_ids": [2, 30, 58],
            "draft_vocab_size": 32768,
            "mask_token_id": 163695,
            "markov_rank": 256,
            "markov_head_type": "vanilla",
            "block_size": 5,
            "enable_confidence_head": True,
            "confidence_head_with_markov": True,
            "sample_from_anchor": True,
        },
        converted,
    )
    assert converted["architectures"] == ["Qwen3DSparkModel"]
    assert converted["eagle_aux_hidden_state_layer_ids"] == [2, 30, 58]
    assert converted["target_layer_ids"] == [1, 29, 57]
    assert converted["draft_vocab_size"] == 32768
    assert converted["markov_rank"] == 256
    assert converted["dspark_bonus_anchor"] is False

    print("AXK2_DSPARK_PORT_WIRING_OK")


if __name__ == "__main__":
    main()
