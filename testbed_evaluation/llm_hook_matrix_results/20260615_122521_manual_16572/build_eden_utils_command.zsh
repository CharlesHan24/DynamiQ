#!/usr/bin/env zsh
CONDA_ENV=llm \
CUDA_HOME=/share/apps/cuda-11.8 \
GCC_HOME=/share/apps/gcc-8.3 \
TORCH_CUDA_ARCH_LIST='8.0;8.6;8.9' \
"/cluster/project2/gcreduce_data/DynamiQ_SIGCOMM_Artifact/testbed_evaluation/build_eden_utils_llm.zsh"
