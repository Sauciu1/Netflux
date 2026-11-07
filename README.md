# Netflux

Netflux is a user-friendly software for developing dynamic computational models of biological networks. Models are created in Excel format and then simulated using the Netflux graphical interface. No computer programming is required. Netflux is written in MATLAB, but binary versions are available for Windows and MacOS for those that do not have MATLAB installed.

Netflux can export network structures and simulation results for visualization in Cytoscape or other software packages. Netflux can also export MATLAB or Python code for more advanced network simulations. Netflux can import network structures based on SIF or SBML-QUAL formats.

Netflux simulates networks using a modeling approach called logic-based differential equations. These methods are described in detail here:
Kraeutler MJ, Soltis AR, Saucerman JJ. Modeling cardiac β-adrenergic signaling with normalized-Hill differential equations: comparison with a biochemical model. BMC Syst Biol. 2010 Nov 18;4:157.
https://bmcsystbiol.biomedcentral.com/articles/10.1186/1752-0509-4-157 (open access)

Netflux has been used to develop large-scale computational models of signaling networks for cardiomyocyte hypertrophy, fibroblasts, and mechano-signaling. Netflux has also been used in educational settings from high school (simulations of social networks) through graduate school levels.

## Installation
Click on the button labeled "Clone or download" and then "Download ZIP". This ZIP file contains the MATLAB version of Netflux as well as binary versions for Windows or MacOS users that do not have MATLAB installed.

Users with MATLAB:
Extract the “Netflux” folder from within the ZIP file and move it to your desired location. Run Netflux.m in MATLAB.

Windows users that do not have MATLAB:
Open the “Netflux Compiled” folder and run “Netflux Installer Windows.exe”. This installer will automatically download and install the MATLAB Compiler Runtime, which will take some time as that file is about 800 MB. Netflux will be installed to the C:\Program Files\Netflux folder. Note installation may require administrator privileges on your computer. An exception to Windows firewall may also be needed, but we are checking on this.

MacOS users with Apple Silicon chips (M1 or M2) that do not have MATLAB:
Open the “Netflux Compiled” folder, extract “Netflux_Installer_MacOS”, and install by running ./Netflux_Installer_macOS.app at the terminal. This installer will automatically download and install the MATLAB Compiler Runtime, which will take some time as that file is about 800 MB. Netflux will be installed to the Applications\Netflux folder. Run Netflux.app from the terminal, in admin mode if necessary. MacOS with older Intel chips are not currently supported. Note Jan 2025: users have been encountering an error when running Netflux.app. If this is not fixed, you will need to use the MATLAB version (see above).

For detailed instructions on installing without MATLAB, see:
https://www.mathworks.com/help/compiler/install-deployed-application.html

## Documentation
"Netflux Documentation.docx" is in the Netflux directory.

## Video Tutorials
Video tutorials are available on Youtube at:
https://www.youtube.com/watch?v=tU2gU8d5hvQ&list=PLTbDE7ZQfKy-ijd5ZrCQJ7eB-NSCQKIBi

## Python package (`src/netflux`)

A pure-Python port of the Netflux core (no MATLAB): Excel model loading, simulation with PyTorch and [torchdiffeq](https://github.com/rtqichen/torchdiffeq), data export, Cytoscape SIF/XGMML export, SIF and SBML-qual import, and Python ODE code export. The GUI is not ported.

### Install
```bash
uv sync --all-extras     # from this repository; extras: [plot] matplotlib, [sbml] python-libsbml
pip install ".[all]"     # or with pip
```
With plain pip on Linux, install the CPU build of torch first: `pip install torch --index-url https://download.pytorch.org/whl/cpu`.

### Python
```python
import netflux

model = netflux.load_example("exampleNet")   # or netflux.read_model("model.xlsx")
sim = netflux.Simulator(model)
sim.set_reaction("r1", weight=1)             # stimulate A
sim.run(10)
sim.set_reaction("r1", weight=0)             # washout; continues from t = 10
result = sim.run(10)

result.plot(["A", "C", "E"])                 # needs [plot]
result.write_data()                          # output/exampleNet_data.txt ("Export Data" format)
netflux.write_sif(model)                     # output/exampleNet.sif + _nodeAttributes.txt
netflux.write_xgmml(model, reference="layout.xgmml")
netflux.export_python_ode(model)             # output/NetfluxODE.py, _params.py, _run.py
```
Parameters (`sim.weight`, `n`, `ec50`, `ymax`, `tau`) are float64 tensors and may be batched or `requires_grad`.

### Command line
```bash
netflux examples
netflux simulate exampleNet --set r1.weight=1 --t-end 10 -o data.txt --plot fig.png
netflux export-sif exampleNet
netflux export-xgmml exampleNet --style
netflux export-python exampleNet
netflux convert-sif net.sif
netflux convert-sbml model.xml
```
`MODEL` is a file path or a bundled example name.

### Output folder
Everything is written to `./output/`, which git ignores. Relative paths go inside it, so `-o data.txt` writes `output/data.txt`; absolute paths are used as given. To change it, use `netflux.set_output_dir(path)` or `netflux.using_output_dir(path)` in Python, the `NETFLUX_OUTPUT_DIR` environment variable, or the CLI's `--output-dir`.

### Differences from MATLAB
- Rules also accept the Netflux2 forms `->`, `AND` and `NOT x`. `!AB` no longer negates a species `A`.
- Problems in a model file are collected in `Model.warnings` instead of stopping the load.
- The default solver is `dopri5`. torchdiffeq's `bosh3` (ode23's scheme) interpolates poorly between steps.
- Activity is clamped at 0 inside the Hill function, where MATLAB goes complex.
- `EC50^n >= 0.5` raises a `NetfluxWarning`.
- In SBML-qual, a transition with several inputs gives one reaction per input. Converted models are saved as `.xlsx`.

Tests (`uv run pytest`) check the SIF export byte-for-byte against the MATLAB files in `Autogenerated models/` and the ODEs against `ODE.m`.
