# Ionic Conductivity and Diffusion Analysis

This repository contains a Python script for analyzing the diffusion properties and ionic conductivity of materials using molecular dynamics simulation data. The script utilizes the `pymatgen` library to perform various analyses, including diffusivity calculations and probability density analysis. Additionally, it generates plots and summary tables to visualize and document the results. Script accepts .extxyz files by default but can be modified for other formats if they are recognized by `pymatgen` and `ase`. Please refer to the links below for further information about calculations:

https://materialsvirtuallab.github.io/pymatgen-analysis-diffusion/pymatgen.analysis.diffusion.analyzer.html#pymatgen.analysis.diffusion.analyzer.DiffusionAnalyzer

https://github.com/materialsvirtuallab/pymatgen-analysis-diffusion

https://github.com/materialsvirtuallab

https://github.com/materialsproject/pymatgen
## Features

- **Diffusivity Calculation**: Computes the diffusivity of ions from molecular dynamics trajectories using the `DiffusionAnalyzer` from `pymatgen`. 
- **Ionic Conductivity Calculation**: Calculates the ionic conductivity from the diffusivity data.
- **Probability Density Analysis**: Analyzes the probability density of ion positions over time to visualize ion pathways.
- **Arrhenius Plot**: Fits the diffusivity data to an Arrhenius equation to extract activation energy and pre-exponential factor.
- **Plotting and Exporting Data**: Generates and saves plots for diffusivity, conductivity and mean squared displacement (MSD). Exports MSD data to CSV files.
- **Summarization**: Creates a summary table of key properties and writes it to an output text file.

## Dependencies

- Python 3.7 or later
- `matplotlib`
- `pymatgen`
- `pymatgen-analysis-diffusion`
- `ase`
- `numpy`
- `scipy`
- `tabulate`

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/okibrisli/conductivity_pymatgen.git
   
2. **Install required Python packages**:
   ```bash
   pip install pymatgen
   pip install ase
   etc.
## Usage
1. **Modify input files**:
   ```bash
   files = [
     '/gpfs/fs1/home/o/ovoznyy/orhan/scratch/xtb_md/concatenate_extxyz/L4_800_final.extxyz',
     '/gpfs/fs1/home/o/ovoznyy/orhan/scratch/xtb_md/concatenate_extxyz/M4_1000_final.extxyz',
     '/gpfs/fs1/home/o/ovoznyy/orhan/scratch/xtb_md/concatenate_extxyz/N4_1200_final.extxyz',
     '/gpfs/fs1/home/o/ovoznyy/orhan/scratch/xtb_md/concatenate_extxyz/O4_1400_final.extxyz',
     '/gpfs/fs1/home/o/ovoznyy/orhan/scratch/xtb_md/concatenate_extxyz/P4_1600_final.extxyz',]

2. **Modify temperatures according to input files**:
   ```bash
   temperatures = [800, 1000, 1200, 1400, 1600] 
3. **Modify miscellaneous parameters**:
   ```bash
   diffusing_species = 'Li'
   time_step = 0.001
   smoothed = False
   steps_to_ignore = 15000
   avg_nsteps = 1000
   step_skip = 100
4. **Modify results directory**:
   ```bash
   job_id = os.environ.get('SLURM_JOB_ID')
   output_dir = f"__{job_id}__"
   if not os.path.exists(output_dir):
       os.makedirs(output_dir)
5. **Run the script**:
   ```bash
   python condcutivity.py

## Outputs

- **Text File**: `output.txt` containing the extrapolated diffusivity and conductivity, diffusion summary, and Arrhenius values.
- **CSV Files**: MSD data files for each temperature.
- **Plots**:
  - `diffusivity_plot_analyzer.png`
  - `conductivity_plot_analyzer.png`
  - MSD plots for each temperature (e.g., `msd_plot_800K.png`)
  - Arrhenius plot (`arrhenius_plot.png`)
- **Probability Density Analysis Output Files**: (e.g., `CHGCAR_800K.vasp`)
   