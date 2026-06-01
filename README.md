# Open-TTT (Colocate Architecture)

Attempting to reproduce Test-Time Training (TTT) on the **Slime** framework using 8B and 20B parameter models.

## 🚧 Current Status & Known Issues
- **8B Model**: Experimental pipeline setup.
- **20B Model**: **[WIP]** Currently encountering an issue when converting and splitting the 20B HF model into Megatron `torch_dist` format. Actively debugging and fixing the weight conversion pipeline.

## Core Frameworks
- `Slime` & `Megatron-LM`: Training engine for 3D parallelism.
- `SGLang`: Rollout generation.