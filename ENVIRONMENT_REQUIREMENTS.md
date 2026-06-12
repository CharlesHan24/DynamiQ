# Environment Requirements

This file records the software stack used for the dynamiQ testbed artifact.
The Python package versions were checked in the active `llm` conda environment.

## Create The Conda Environment

```bash
conda create -n llm python=3.9 pip
conda activate llm
```

Install a PyTorch build that matches your CUDA/NCCL stack. The validated stack
uses CUDA 11.8 and NCCL 2.19.3:

```bash
# Example for a public CUDA 11.8 PyTorch wheel. Verify the NCCL version after
# installation; wheel-bundled NCCL versions can differ by PyTorch release.
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

pip install -r requirements.txt
```

After installation, verify:

```bash
python - <<'PY'
import sys
import torch

print("python", sys.version)
print("torch", torch.__version__)
print("torch CUDA", torch.version.cuda)
print("NCCL", torch.cuda.nccl.version())
PY
```

The expected CUDA/NCCL lines are:

```text
torch CUDA 11.8
NCCL (2, 19, 3)
```

## Tested Python Stack

Observed in the current `llm` environment:

```text
python==3.9.5
torch==2.3.0.dev20240131
torchvision==0.18.0.dev20240131
torchaudio==2.2.0.dev20240131
accelerate==1.6.0
transformers==4.50.3
datasets==3.5.0
tokenizers==0.21.1
safetensors==0.5.3
huggingface-hub==0.30.1
sentencepiece==0.1.99
numpy==1.24.3
pandas==2.2.3
pyarrow==19.0.1
scipy==1.13.1
scikit-learn==1.5.1
tqdm==4.67.1
fsspec==2023.12.2
xxhash==3.5.0
multiprocess==0.70.16
dill==0.3.8
regex==2023.12.25
requests==2.32.3
packaging==23.2
filelock==3.9.0
PyYAML==6.0.1
psutil==5.9.8
typing_extensions==4.9.0
ninja==1.13.0
setuptools==68.2.2
wheel==0.41.2
```

## Non-Python Requirements

Required:

- Linux with NVIDIA GPUs and a CUDA 11.8 compatible driver.
- Python 3.9.
- CUDA toolkit 11.8, including `nvcc`.
- NCCL 2.19.3 for distributed GPU collectives.
- GCC/G++ with C++17 support. The tested runtime PATH used GCC 9.2.0; PyTorch
  itself was built with GCC 9.3.
- `ninja` for PyTorch CUDA extension builds.
- RDMA userspace libraries and headers: `libibverbs` and `librdmacm`.
- `numactl`, used by the launch scripts for CPU/memory binding.
- `zsh` and `bash`, used by the provided launch scripts.

Recommended:

- CMake 3.24 or newer for auxiliary native builds.
- NVIDIA Nsight Systems (`nsys`) if using the optional profiling flags.
- A scheduler or launcher equivalent to the provided qsub wrappers, or adapt
  the wrappers to your local cluster scheduler.

## Version Check Commands

```bash
python --version
python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.nccl.version())'
gcc --version
g++ --version
nvcc --version
ninja --version
cmake --version
```
