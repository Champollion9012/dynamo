#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

image="${1:-nvcr.io/nvstaging/nim/ax-k2-nvfp4:dynamo-1.4.1-vllm0.26.0-fp8-ds-mla}"
dynamo_axk2_image="${DYNAMO_AXK2_IMAGE:-nvcr.io/nvstaging/nim/ax-k2-nvfp4:dynamo-1.4.1-vllm0.26.0-dspark@sha256:edf99463e04d405a3e58568893fe811c4e66156680b38574df559fd5b4f4398b}"

docker build \
    --progress=plain \
    --file container/Dockerfile.axk2-fp8-ds-mla \
    --build-arg "DYNAMO_AXK2_IMAGE=${dynamo_axk2_image}" \
    --tag "${image}" \
    .
