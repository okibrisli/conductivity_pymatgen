import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pymatgen.analysis.diffusion.analyzer import DiffusionAnalyzer, get_extrapolated_conductivity, \
    get_extrapolated_diffusivity, get_conversion_factor, fit_arrhenius
from pymatgen.analysis.diffusion.aimd.pathway import ProbabilityDensityAnalysis
from pymatgen.analysis.diffusion.aimd.van_hove import VanHoveAnalysis
from pymatgen.io.ase import AseAtomsAdaptor
from ase.io import read
import os
import numpy as np
import csv
from scipy import stats
from tabulate import tabulate


# Constants
e = 1.60217662e-19  # Charge of an electron in coulombs
k_B = 1.38064852e-23  # Boltzmann constant in J/K

# Get the SLURM job ID
job_id = os.environ.get('SLURM_JOB_ID')

# Create a directory named with the job ID in the working directory
output_dir = f"__{job_id}__"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# List of extxyz files
files = [
     '/gpfs/fs1/home/o/ovoznyy/orhan/scratch/xtb_md/concatenate_extxyz/L4_800_final.extxyz',
     '/gpfs/fs1/home/o/ovoznyy/orhan/scratch/xtb_md/concatenate_extxyz/M4_1000_final.extxyz',
     '/gpfs/fs1/home/o/ovoznyy/orhan/scratch/xtb_md/concatenate_extxyz/N4_1200_final.extxyz',
     '/gpfs/fs1/home/o/ovoznyy/orhan/scratch/xtb_md/concatenate_extxyz/O4_1400_final.extxyz',
     '/gpfs/fs1/home/o/ovoznyy/orhan/scratch/xtb_md/concatenate_extxyz/P4_1600_final.extxyz',
]

# Manually define temperatures corresponding to each file
temperatures = [800, 1000, 1200, 1400, 1600]  # Adjust respectively with extxyz files


ase_to_pmg = AseAtomsAdaptor()


def structures_traj(file, steps_to_ignore=False):
    structure_list = read(file, index=':', format='extxyz')
    structure_list = list(structure_list)  # Convert to list if not already
    if steps_to_ignore:
        structure_list = structure_list[steps_to_ignore:]
    pmg_structures = [ase_to_pmg.get_structure(s) for s in structure_list]
    return pmg_structures


# Specify the diffusing species in the simulation
diffusing_species = 'Li'

# Time step between frames in the trajectory (in picoseconds)
time_step = 0.001

# Whether to smooth the calculated mean squared displacements (MSD)
smoothed = False

# Number of initial steps to ignore in the trajectory to allow equilibration
steps_to_ignore = 5000

# Number of steps to average over for calculating diffusivity
avg_nsteps = 5000

# Step interval to skip when calculating MSD (i.e., use every 'step_skip' step)
step_skip = 1000

diff_analyzer = []
diffusivities = []
msd_diffusivities = []
conductivities_manual = []
conductivities_analyzer = []
conductivities_analyzer_S = [value / 1000 for value in conductivities_analyzer]
conversion_factors = []
summaries = []


# Function to calculate diffusivity from MSD
def get_diffusivity_from_msd(msd, dt, smoothed='max'):
    msd = np.array(msd)
    dt = np.array(dt)
    slope, intercept, r_value, p_value, std_err = stats.linregress(dt, msd)
    diffusivity = slope / 6  # Assuming 3D diffusion (factor of 2d where d=3)
    return diffusivity, std_err


# Loop through each file and calculate diffusivity
for i in range(len(files)):
    structures = structures_traj(files[i], steps_to_ignore)
    analyzer = DiffusionAnalyzer.from_structures(structures,
                                                 specie=diffusing_species,
                                                 temperature=temperatures[i],
                                                 time_step=time_step,
                                                 smoothed=smoothed,
                                                 step_skip=step_skip,
                                                 avg_nsteps=avg_nsteps)
    diff_analyzer.append(analyzer)

    # Calculate diffusivity using DiffusionAnalyzer's diffusivity attribute
    diffusivity_analyzer = analyzer.diffusivity
    diffusivities.append(diffusivity_analyzer)

    conductivity_analyzer = analyzer.conductivity
    conductivities_analyzer.append(conductivity_analyzer)

    # Calculate conductivity manually using analyzer's diffusivity
    structure = analyzer.structure
    num_ions = sum([site.species[diffusing_species] for site in structure if diffusing_species in site.species])
    volume = structure.volume
    N = num_ions / volume  # Number of ions per unit volume
    conductivity_manual = (diffusivity_analyzer * e ** 2 * N) / (k_B * analyzer.temperature)
    conductivities_manual.append(conductivity_manual)

    # Export MSD data to CSV
    msd_filename = os.path.join(output_dir, f"msd_data_{temperatures[i]}K.csv")
    analyzer.export_msdt(msd_filename)

    conversion_factor = get_conversion_factor(structure, diffusing_species, temperatures[i])
    conversion_factors.append(conversion_factor)

    # Get summary dictionary
    summary = analyzer.get_summary_dict(include_msd_t=False, include_mscd_t=False)
    summaries.append(summary)

    # Calculate diffusivity from MSD data
    if os.path.exists(msd_filename):
        time = []
        msd = []
        with open(msd_filename, 'r') as file:
            reader = csv.reader(file)
            next(reader)  # Skip header
            for row in reader:
                time.append(float(row[0]))
                msd.append(float(row[1]))

        diffusivity_msd, std_err = get_diffusivity_from_msd(msd, time)
        msd_diffusivities.append(diffusivity_msd)
    else:
        msd_diffusivities.append(None)

# Calculate Arrhenius values using all temperatures and diffusivities
arrhenius_values = fit_arrhenius(temperatures, diffusivities)

# Calculate extrapolated diffusivity and conductivity
if len(files) > 1:
    structure = read(files[0], index=0, format='extxyz')
    structure = ase_to_pmg.get_structure(structure)
    extrapolated_diffusivity = get_extrapolated_diffusivity(temperatures, diffusivities, new_temp=300)
    extrapolated_conductivity = get_extrapolated_conductivity(temperatures, diffusivities, new_temp=300,
                                                              structure=structure, species='Li')
else:
    extrapolated_diffusivity = None
    extrapolated_conductivity = None

# Write results to a file
output_file = os.path.join(output_dir, "output.txt")
with open(output_file, "w") as f:

    if extrapolated_diffusivity and extrapolated_conductivity:
        f.write(f"Extrapolated Diffusivity at 300K: {extrapolated_diffusivity:.3e} cm^2/s\n")
        f.write(f"Extrapolated Conductivity at 300K: {extrapolated_conductivity/1000:.3e} S/cm\n")
    else:
        f.write("\nOnly one file provided, extrapolated diffusivity and conductivity not calculated.\n")

    f.write("\n\nDiffusion Summary:\n")
    for key in summaries[0].keys():
        f.write(f"{key}:\n")
        for temp, summary in zip(temperatures, summaries):
            f.write(f"  {temp}K: {summary[key]}\n")

    # Arrhenius values
    f.write("\n\nArrhenius Values from Analyzer:\n")
    f.write(f"Activation Energy: {arrhenius_values[0]:.3e} eV\n")
    f.write(f"Pre-exponential Factor: {arrhenius_values[1]:.3e} cm^2/s\n")
    if arrhenius_values[2] is not None:
        f.write(f"Standard Error: {arrhenius_values[2]:.3e} eV\n")
    else:
        f.write("Standard Error: Not available\n")


print(f"Results written to {output_file}")

# Plot Diffusivity
plt.figure()
plt.plot(temperatures, diffusivities, 'o-', label='Analyzer Diffusivity')
plt.xlabel('Temperature (K)')
plt.ylabel('Diffusivity (cm^2/s)')
plt.title('Diffusivity vs Temperature')

# Set the y-axis to exponential format
ax = plt.gca()
ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
ax.yaxis.get_major_formatter().set_scientific(True)
ax.yaxis.get_major_formatter().set_powerlimits((0, 0))
plt.legend()
plt.savefig(os.path.join(output_dir, 'diffusivity_plot_analyzer.png'))

# Plot Conductivity
plt.figure()
plt.plot(temperatures, conductivities_analyzer_S, 'o-')
plt.xlabel('Temperature (K)')
plt.ylabel('Conductivity (S/cm)')
plt.title('Conductivity vs Temperature')
plt.savefig(os.path.join(output_dir, 'conductivity_plot_analyzer.png'))

# Plot MSD from CSV files and calculate diffusivity from MSD
for temp in temperatures:
    msd_filename = os.path.join(output_dir, f"msd_data_{temp}K.csv")
    if os.path.exists(msd_filename):
        time = []
        msd = []
        msd_a = []
        msd_b = []
        msd_c = []
        mscd = []
        with open(msd_filename, 'r') as file:
            reader = csv.reader(file)
            next(reader)  # Skip header
            for row in reader:
                time.append(float(row[0]))
                msd.append(float(row[1]))
                msd_a.append(float(row[2]))
                msd_b.append(float(row[3]))
                msd_c.append(float(row[4]))
                mscd.append(float(row[5]))

        plt.figure()
        plt.plot(time, msd, label='MSD', linewidth=0.3)  # Thinner lines
        plt.plot(time, msd_a, label='MSD_a', linewidth=0.3)  # Thinner lines
        plt.plot(time, msd_b, label='MSD_b', linewidth=0.3)  # Thinner lines
        plt.plot(time, msd_c, label='MSD_c', linewidth=0.3)  # Thinner lines
        plt.plot(time, mscd, label='MSCD', linewidth=0.3)  # Thinner lines
        plt.xlabel('Time (ps)')
        plt.ylabel('MSD (Å^2)')
        plt.title(f'MSD vs Time at {temp}K')
        plt.legend()
        plt.savefig(os.path.join(output_dir, f'msd_plot_{temp}K.png'))
        plt.show()
    else:
        print(f"MSD file {msd_filename} not found.")

# Calculate custom Arrhenius values
def custom_arrhenius(temperatures, diffusivities):
    inv_T = 1000 / np.array(temperatures)  # 1/T in 1/K
    ln_D = np.log(diffusivities)
    slope, intercept, r_value, p_value, std_err = stats.linregress(inv_T, ln_D)
    Ea = -slope * 8.617333262145e-5 * 1000  # eV/K (Boltzmann constant in eV/K) multiplied by 1000 to correct units
    c = np.exp(intercept)
    return inv_T, ln_D, Ea, c, std_err


inv_T, ln_D, Ea, c, std_err = custom_arrhenius(temperatures, diffusivities)

# Write custom Arrhenius values to the output file
with open(output_file, "a") as f:
    f.write("\n\nCustom Arrhenius Values:\n")
    f.write(f"Activation Energy: {Ea:.3e} eV\n")
    f.write(f"Pre-exponential Factor: {c:.3e} cm^2/s\n")
    if std_err is not None:
        f.write(f"Standard Error: {std_err:.3e} eV\n")
    else:
        f.write("Standard Error: Not available\n")


# Function to create an Arrhenius plot
def plot_arrhenius(temperatures, diffusivities, output_dir):
    inv_T_p = 1000 / np.array(temperatures)  # 1/T in 1/K
    ln_D_p = np.log(diffusivities)
    slope_p, intercept_p, r_value_p, p_value_p, std_err_p = stats.linregress(inv_T_p, ln_D_p)
    Ea_p = -slope_p * 8.617333262145e-5  # eV/K (Boltzmann constant in eV/K) multiplied by 1000 to correct units
    c_p = np.exp(intercept_p)

    fig, ax1 = plt.subplots()

    # Primary axis (bottom) showing 1000/T
    ax1.plot(inv_T_p, ln_D_p, 'o', label='Data')
    ax1.plot(inv_T_p, slope_p * inv_T_p + intercept_p, '-', label=f'Fit: Ea={Ea_p:.3e} eV')
    ax1.set_xlabel('1000/T (1/K)')
    ax1.set_ylabel('ln(Diffusivity) (ln(cm^2/s))')
    ax1.set_title('Arrhenius Plot')
    ax1.legend()


    # Adding text to the plot with adjusted positions
    ax1.text(0.003, -22, f'Extrapolated Conductivity: {extrapolated_conductivity / 1000:.3e} S/cm', fontsize=12,
             color='red')
    ax1.text(0.003, -24, f'Extrapolated Diffusivity: {extrapolated_diffusivity:.3e} cm^2/s', fontsize=12, color='blue')

    # Secondary axis (top) showing Temperature in K
    def inv_T_to_T(inv_T):
        return 1000 / inv_T

    def T_to_inv_T(T):
        return 1000 / T

    ax2 = ax1.secondary_xaxis('top', functions=(inv_T_to_T, T_to_inv_T))
    ax2.set_xlabel('Temperature (K)')

    plt.savefig(os.path.join(output_dir, 'arrhenius_plot.png'))
    plt.show()

    return Ea_p, c_p, std_err_p


# Plot Arrhenius
Ea_p, c_p, std_err_p = plot_arrhenius(temperatures, diffusivities, output_dir)


# Function to write final summary table to output file
def write_table_to_output(file_path, temperatures, diffusivities, msd_diffusivities, conductivities_analyzer_S, conductivities_manual, conversion_factors):
    with open(file_path, "a") as f:
        f.write("\n\nSummary Table:\n")
        headers = ["Property"] + [f"{temp}K" for temp in temperatures]
        rows = [
            ["diffusivities_analyzer (cm^2/s)"] + diffusivities,
            ["conductivities_analyzer (S/cm)"] + conductivities_analyzer_S,
            ["msd_diffusivities (cm^2/s)"] + msd_diffusivities,
            ["conductivities_manual"] + conductivities_manual,
            ["conversion_factors"] + conversion_factors
        ]

        table = tabulate(rows, headers, tablefmt="grid")
        f.write(table + "\n")


write_table_to_output(output_file, temperatures, diffusivities, msd_diffusivities, conductivities_analyzer_S, conductivities_manual, conversion_factors)

print(f"Results written to {output_file}")

# Perform Probability Density Analysis for each temperature
for i, analyzer in enumerate(diff_analyzer):
    structure = analyzer.structure
    trajectories = [s.frac_coords for s in analyzer.get_drift_corrected_structures()]
    pda = ProbabilityDensityAnalysis(structure, trajectories, species="Li")
    output_filename = os.path.join(output_dir, f"CHGCAR_{temperatures[i]}K.vasp")
    pda.to_chgcar(output_filename)
    print(f"Probability Density Analysis for {temperatures[i]}K written to {output_filename}")

# Perform Van Hove Analysis for each temperature
for i, analyzer in enumerate(diff_analyzer):
    van_hove = VanHoveAnalysis(analyzer, avg_nsteps=avg_nsteps, step_skip=step_skip, species=['Li'])

    # Save 1D plot of the distinct part of the van Hove function
    vh_1d_distinct_plot_file = os.path.join(output_dir, f"van_hove_distinct_{temperatures[i]}K.png")
    van_hove.get_1d_plot(mode='distinct', times=[0.0, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0])
    plt.savefig(vh_1d_distinct_plot_file)
    plt.close()
    print(f"Van Hove distinct plot for {temperatures[i]}K saved to {vh_1d_distinct_plot_file}")

    # Save 1D plot of the self part of the van Hove function
    vh_1d_self_plot_file = os.path.join(output_dir, f"van_hove_self_{temperatures[i]}K.png")
    van_hove.get_1d_plot(mode='self', times=[0.0, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0])
    plt.savefig(vh_1d_self_plot_file)
    plt.close()
    print(f"Van Hove self plot for {temperatures[i]}K saved to {vh_1d_self_plot_file}")

    # Save 3D plot of the distinct part of the van Hove function
    vh_3d_distinct_plot_file = os.path.join(output_dir, f"van_hove_3d_distinct_{temperatures[i]}K.png")
    van_hove.get_3d_plot(mode='distinct')
    plt.savefig(vh_3d_distinct_plot_file)
    plt.close()
    print(f"Van Hove 3D distinct plot for {temperatures[i]}K saved to {vh_3d_distinct_plot_file}")

    # Save 3D plot of the self part of the van Hove function
    vh_3d_self_plot_file = os.path.join(output_dir, f"van_hove_3d_self_{temperatures[i]}K.png")
    van_hove.get_3d_plot(mode='self')
    plt.savefig(vh_3d_self_plot_file)
    plt.close()
    print(f"Van Hove 3D self plot for {temperatures[i]}K saved to {vh_3d_self_plot_file}")
