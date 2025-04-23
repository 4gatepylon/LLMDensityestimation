<p align="center">
  <img src="inference_time_experiments/static/benign-robot.png" alt="Benign Robot Logo" width="200">
</p>

# Obfuscated Activations Bypass LLM Latent-Space Defenses

[![arXiv](https://img.shields.io/badge/arXiv-2309.00236-b31b1b.svg)](https://arxiv.org/abs/2412.09565)
[![Website](https://img.shields.io/badge/Project-Page-blue)](https://obfuscated-activations.github.io/)


This repository contains the code for the paper [Obfuscated Activations Bypass LLM Latent-Space Defenses](https://arxiv.org/abs/2412.09565). The codebase is organized into two main components.

## Repository Structure

```
.
├── inference_time_experiments/  # Code for inference-time attacks (Sections 3 & 5)
└── train_time_experiments/      # Code for training-time attacks (Sections 4 & 6.2)
```

### Inference Time Experiments
The [`inference_time_experiments/`](inference_time_experiments/) directory contains code for:
- Section 3: Case Study: Detecting Harmfulness
- Section 5: Case Study: SAE Probing

This includes implementations for attacking models at inference time through suffix based attacks, with experiments on harmfulness detection and SAE probing.

### Training Time Experiments
The [`train_time_experiments/`](train_time_experiments/) directory contains code for:
- Section 4: Case Study: Backdoor Obfuscation  
- Section 6.2: Training Models to be Easier to Monitor (Obfuscated Adversarial Training)

This includes implementations for training backdoored models with both explicit and natural obfuscation strategies, as well as Obfuscated Adversarial Training (OAT).

## Getting Started

Each subdirectory contains its own README with detailed instructions for installation and running experiments. Please refer to:
- `inference_time_experiments/README.md`
- `train_time_experiments/README.md`

## Contact

If you have any issues with the code, feel free to submit a github issue or contact Luke Bailey at ljbailey@stanford.edu.

## Citation

```bibtex
@article{bailey2024obfuscated,
	title={Obfuscated Activations Bypass LLM Latent-Space Defenses},
	author={Bailey, Luke and Serrano, Alex and Sheshadri, Abhay and Seleznyov, Mikhail and Taylor, Jordan and Jenner, Erik and Hilton, Jacob and Casper, Stephen and Guestrin, Carlos and Emmons, Scott},
	journal={arXiv preprint arXiv:2412.09565},
	year={2024}
}
```
