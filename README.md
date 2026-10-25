# Anisotropic Viscoplastic-Damage Material Model

This package contains standalone material-point implementations for explicit
and implicit anisotropic viscoplastic damage analysis, automatic
differentiation, and batched GPU execution.

| Directory | Main module | Run command |
| --- | --- | --- |
| `explicit_anisotropic` | `explicit_anisotropic_damage.py` | `python explicit_anisotropic/run_explicit_anisotropic.py --steps 8` |
| `implicit_anisotropic` | `implicit_anisotropic_damage.py` | `python implicit_anisotropic/run_implicit_anisotropic.py` |
| `automatic_differentiation` | `implicit_anisotropic_damage_ad.py` | `python automatic_differentiation/run_automatic_differentiation.py` |
| `gpu_implementation` | `batched_anisotropic_damage_gpu.py` | `python gpu_implementation/run_gpu_implementation.py` |

## Installation

```bash
python -m pip install -r requirements.txt
```

The GPU runner uses CUDA when it is available and otherwise executes on CPU.
All input data required by the included demonstrations is stored in `inputs/`.
