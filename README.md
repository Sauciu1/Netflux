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

`[all]` adds plotting (matplotlib); drop it for the core only.

On Linux, PyPI's default PyTorch includes several GB of CUDA libraries. To use the CPU build, add this to your `pyproject.toml` (uv), or run `pip install torch --index-url https://download.pytorch.org/whl/cpu` first (pip):

```toml
[tool.uv.sources]
torch = [{ index = "pytorch-cpu", marker = "sys_platform != 'darwin'" }]

[[tool.uv.index]]
name = "pytorch-cpu"
url = "https://download.pytorch.org/whl/cpu"
explicit = true
```

## Use as a module

```python
import netflux

model = netflux.load_example("exampleNet")      # bundled example, or netflux.read_model("my_model.xlsx")

# One-shot simulation from t = 0
result = netflux.simulate(model, t_end=10, dt=0.1, reactions={"r1": {"weight": 1}})
result.final()                    # {"A": 0.99995, ...} at the last time point
result["E"]                       # trajectory of E (torch tensor, one value per time point)
result.to_dataframe()             # pandas DataFrame: index t, one column per species

# Step-by-step experiments, as in the Netflux GUI: each run continues from the last state
sim = netflux.Simulator(model)
sim.set_reaction("r1", weight=1)  # stimulate A
sim.run(10)
sim.set_reaction("r1", weight=0)  # washout
sim.set_species("C", tau=2)       # also ymax=..., or y0=... to set the current activity
result = sim.run(10)              # cumulative result, t = 0..20
result.plot(["A", "C", "E"])      # matplotlib, needs [plot]
result.write_data("data.txt")     # output/data.txt, the GUI's "Export Data" table
sim.reset()                       # back to t = 0; sim.reset_parameters() restores the model's values

# Models can also be built in code
toy = netflux.Model(
    "toy",
    species=[netflux.Species("A"), netflux.Species("B", tau=2)],
    reactions=[netflux.Reaction.from_rule("r1", "=> A"), netflux.Reaction.from_rule("r2", "A & !B => B", n=2)],
)
toy.diagnose()                    # list of problems, e.g. unknown species in a rule
```

- `run()` and `simulate()` accept `method`, `rtol` and `atol` for the torchdiffeq solver (default `dopri5`, `rtol=1e-6`, `atol=1e-8`).
- The parameters `sim.weight`, `sim.n`, `sim.ec50`, `sim.ymax` and `sim.tau` are float64 tensors. They can be given leading batch dimensions to run many parameter sets at once, or `requires_grad` for gradient-based fitting.
- Files are written to `./output/` unless the path is absolute. Change this with `netflux.set_output_dir("results")` or the `NETFLUX_OUTPUT_DIR` environment variable.

## Model files

Models can be Excel (`.xlsx`/`.xls`, the original Netflux layout) or **SBML-qual** (`.xml`), the SBML standard for logical models.

In SBML-qual each species is a `qualitativeSpecies`. Each species produced by reactions gets one `transition`, and every Netflux reaction is one of its `functionTerm`s: the inputs of a term are ANDed, and several terms are ORed. For example, reaction `r5: C & !D => E` becomes:

```xml
<qual:functionTerm qual:resultLevel="1">
  <annotation><netflux:reaction id="r5" order="4" weight="1" n="1.4" ec50="0.5"/></annotation>
  <math> C == 1 and D == 0 </math>  <!-- written as MathML -->
</qual:functionTerm>
```

Netflux's continuous parameters (`y0`, `ymax`, `tau`, `weight`, `n`, `ec50`) are stored in `<annotation>` elements, so a round trip loses nothing. Other tools such as GINsim, bioLQM, MaBoSS and Cell Collective ignore the annotations and still read the network logic. SBML-qual files from those tools can be read too: their logic is converted into Netflux reactions with default parameters.

SBML reading and writing go through python-libsbml.
- **Unreadable files:** an invalid file raises `ModelError` listing the problems with line numbers, and so does logic Netflux can't represent.
- **Approximations:** non-fatal issues, such as multi-valued species being treated as Boolean, go into `model.warnings`.
- **Written files:** these are validated before they are saved.

```python
import netflux

model = netflux.read_model("model.xml")      # format chosen by suffix
model = netflux.from_sbml(sbml_text)         # SBML generated in code
netflux.write_model(model, "model.xml")      # output/model.xml
sbml_text = netflux.to_sbml(model)
```

`read_model` reads `.xlsx`, `.xls`, `.xml`/`.sbml` (SBML-qual) and `.sif`. `write_model` writes `.xlsx`, `.xml`/`.sbml`, `.sif` (plus `_nodeAttributes.txt`), `.xgmml` (Cytoscape) and `.py` (standalone PyTorch code). The CLI does the same: `netflux convert exampleNet -o exampleNet.xml`.
