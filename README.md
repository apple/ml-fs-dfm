# FS-DFM: FAST AND ACCURATE LONG TEXT GENERATION WITH FEW-STEP DIFFUSION LANGUAGE MODELS 

A PyTorch implementation of FS-DFM with custom solvers for efficient text generation and discrete sequence modeling. This software project accompanies the research paper, [FS-DFM: Fast and Accurate Long Text Generation with Few-Step Diffusion Language Models](https://arxiv.org/abs/2509.20624) .

## Overview

This repository contains:
- **Flow Matching** (`flow_matching/`): Based on [Flow Matching from Meta](https://github.com/facebookresearch/flow_matching) implementation, with our custom discrete solvers added in `discrete_solver_fsdfm.py`
- **FS-DFM** (`fs_dfm/`): Fast sampling diffusion flow matching for discrete sequences
- **Pre-training** (`pre_training/`): Transformer-based model pre-training utilities


## Comparison of Methods

| ARM | DFM | FS-DFM (Ours) |
|-----|-----|---------------|
| ![ARM](assets/gifs/arm.gif) | ![DFM](assets/gifs/dfm.gif) | ![FS-DFM](assets/gifs/fs_dfm.gif) |



## Key Features

- Custom discrete flow matching solvers (`flow_matching/solver/discrete_solver_fsdfm.py`)
- Student-teacher distillation framework
- Multiple solver options: `mixture_euler`, `mixture_euler_with_cumulative_scalar`
- Support for various source distributions (uniform, mask)
- Efficient sampling with configurable steps

## Installation

### Prerequisites
- Python 3.8+
- CUDA 11.0+ (for GPU support)
- conda or mamba package manager

### Setup Environment

```bash
# Create conda environment
conda env create -f fsdfm_environment.yml

# Activate environment
conda activate FSDFM 

# Install package in development mode
pip install -e .
```

## Usage

### Training

Train a model using the FS-DFM framework:

```bash
python fs_dfm/run_train.py \
    data.cache_dir=${CACHE_DIR:-./cache_dir}
```

Configuration can be modified in `fs_dfm/configs/config.yaml`.

### Evaluation

Evaluate a trained model:

```bash
python fs_dfm/run_eval.py \
    --work_dir "/path/to/output/artifacts" \
    --ngpus 1 \
    --perplexity_n_samples 320 \
    --eval_elbo \
    --eval_perplexity \
    --do_dynamic_step \
    --pre_trained_model_path "/path/to/checkpoint.pth"
```

### Pre-training

For pre-training transformer models:

```bash
python pre_training/run_train.py
```

## Configuration

Key configuration parameters in `fs_dfm/configs/config.yaml`:

- **Flow Settings**:
  - `source_distribution`: Choose between `uniform` or `mask`
  - `sampling_steps`: Number of sampling steps (default: 1024)
  - `student_solver`: Solver type for student model
  - `temperature`: Temperature for sampling (default: 1.0)

- **Training Settings**:
  - `optimizer`: AdamW optimizer with configurable learning rates
  - `weight_decay`: 0.03
  - `grad_clip`: 1.0
  - `n_iters`: Total training iterations

- **Evaluation Settings**:
  - `batch_size`: Evaluation batch size
  - `perplexity`: Enable perplexity evaluation
  - `sample_batch_size`: Batch size for sampling

## Project Structure

```
.
├── flow_matching/
│   └── solver/
│       └── discrete_solver_fsdfm.py    # Custom discrete flow solvers
├── fs_dfm/
│   ├── configs/                      # Configuration files
│   ├── eval.py                       # Evaluation utilities
│   ├── logic/
│   │   └── evaluate.py              # Likelihood estimation
│   └── run_train.py                 # Training script
└── pre_training/
    ├── data/
    │   └── data.py                   # Data loading utilities
    ├── model/
    │   └── transformer.py            # Transformer model components
    └── run_train.py                  # Pre-training script
```

## Key Components

### Discrete Solver (`flow_matching/solver/discrete_solver_fsdfm.py`)

The `finite_probs_to_generator` method converts probability distributions to flow generators with:
- Energy barrier for controlling transitions
- Proper normalization with step size (`dt_seg`)
- Safety clipping for numerical stability

### Model Architecture

The framework supports transformer-based architectures with:
- Configurable vocabulary size
- Dropout regularization
- Distributed training support

## Citation

If you use this code in your research, please cite:

```bibtex
@article{fsdfm2025,
  title={FS-DFM: Fast and Accurate Long Text Generation with Few-Step Diffusion Language Models},
  author={Amin Karimi Monsefi and Nikhil Bhendawade and Manuel Rafael Ciosici and Dominic Culver and Yizhe Zhang and Irina Belousova},
  year={2025}
}
```

## Acknowledgments

The flow matching implementation is based on [Flow Matching from Meta](https://github.com/facebookresearch/flow_matching), with custom discrete solvers added in `discrete_solver_fsdfm.py`.

## License

See [LICENSE](./LICENSE) file for details.


## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
