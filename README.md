### Experiments
The experiments are saved as tiff stacks. Each stack needs a json metadata file that can be created with `createMetadata.py`. The stack can then be processed with `experiments_process.py`. These can then be fitted with `experiments_fit.py`.

### Libraries
The fitting functions are located in `diffusiophoresis_fitting.py` and can be used to fit a set of diffusiophoresis curves.
The image processing functions are in `diffusiophoresis_processing.py` and are used to extract the diffusiophoresis curves.

### Simulations
Simulations are done in COMSOL and extracted as a grid data. The `comsol_parser.py` file parses the COMSOL files and translate them into numpy data. These can then be fitted with `simulations_fit.py`.

### Figures
The figures are created with the `figure_*.py` files.
