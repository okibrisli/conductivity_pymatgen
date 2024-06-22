# Ionic Conductivity and Diffusion Analysis

This repository contains a Python script for analyzing the diffusion properties and ionic conductivity of materials using molecular dynamics simulation data. The script utilizes the `pymatgen` library to perform various analyses, including diffusivity calculations, probability density analysis, and van Hove function analysis. Additionally, it generates plots and summary tables to visualize and document the results.

## Features

- **Diffusivity Calculation**: Computes the diffusivity of ions from molecular dynamics trajectories using the `DiffusionAnalyzer` from `pymatgen`.
- **Ionic Conductivity Calculation**: Calculates the ionic conductivity from the diffusivity data.
- **Probability Density Analysis**: Analyzes the probability density of ion positions over time to visualize ion pathways.
- **Van Hove Function Analysis**: Computes and plots the self-part and distinct-part of the van Hove correlation function.
- **Arrhenius Plot**: Fits the diffusivity data to an Arrhenius equation to extract activation energy and pre-exponential factor.
- **Plotting and Exporting Data**: Generates and saves plots for diffusivity, conductivity, mean squared displacement (MSD), and van Hove functions. Exports MSD data to CSV files.
- **Summarization**: Creates a summary table of key properties and writes it to an output text file.

## Dependencies

- Python 3.7 or later
- `matplotlib`
- `pymatgen`
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
2. **Run the script**:
   ```bash
   python conductivity.py