# Netflux (Python)

A Python adaptation of [Netflux](https://github.com/saucermanlab/Netflux), the MATLAB tool for logic-based ODE models of biological networks.

- `src/netflux/`: the new Python package
- `Netflux/`: the original MATLAB code (its example models are bundled with the package)
- `pyproject.toml`: package metadata and dependencies

## Install as a dependency

Requires Python 3.13+. In your own project:

```bash
uv add "netflux[all] @ git+https://github.com/Sauciu1/Netflux"
# or
pip install "netflux[all] @ git+https://github.com/Sauciu1/Netflux"
```

or list it in your `pyproject.toml`:

```toml
[project]
dependencies = ["netflux[all] @ git+https://github.com/Sauciu1/Netflux"]
```

`[all]` adds plotting (matplotlib) and SBML-qual import (python-libsbml); drop it for the core only.

On Linux, PyPI's default PyTorch includes several GB of CUDA libraries. To use the CPU build, add this to your `pyproject.toml` (uv), or run `pip install torch --index-url https://download.pytorch.org/whl/cpu` first (pip):

```toml
[tool.uv.sources]
torch = [{ index = "pytorch-cpu", marker = "sys_platform != 'darwin'" }]

[[tool.uv.index]]
name = "pytorch-cpu"
url = "https://download.pytorch.org/whl/cpu"
explicit = true
```
