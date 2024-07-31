import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pymatgen.analysis.diffusion.analyzer import DiffusionAnalyzer, get_extrapolated_conductivity, \
    get_extrapolated_diffusivity, get_conversion_factor, fit_arrhenius, get_arrhenius_plot
from pymatgen.analysis.diffusion.aimd.pathway import ProbabilityDensityAnalysis
from pymatgen.analysis.diffusion.aimd.van_hove import VanHoveAnalysis
from pymatgen.analysis.diffusion.aimd.rdf import RadialDistributionFunction
from pymatgen.io.ase import AseAtomsAdaptor
from ase.io import read
# from ase.rdf import RadialDistributionFunction
import os
import numpy as np
import csv
from scipy import stats
from tabulate import tabulate

# Constants
# e = 1.60217662e-19  # Charge of an electron in coulombs
# k_B = 1.38064852e-23  # Boltzmann constant in J/K



# List of extxyz files
# files = [
#       '/gpfs/fs1/home/o/ovoznyy/orhan/scratch/xtb_md/concatenate_extxyz/xtb_final_6_trimmed/L6_800_final.extxyz',
#       '/gpfs/fs1/home/o/ovoznyy/orhan/scratch/xtb_md/concatenate_extxyz/xtb_final_6_trimmed/M6_1000_final.extxyz',
#       '/gpfs/fs1/home/o/ovoznyy/orhan/scratch/xtb_md/concatenate_extxyz/xtb_final_6_trimmed/N6_1200_final.extxyz',
#       '/gpfs/fs1/home/o/ovoznyy/orhan/scratch/xtb_md/concatenate_extxyz/xtb_final_6_trimmed/O6_1400_final.extxyz',
#       '/gpfs/fs1/home/o/ovoznyy/orhan/scratch/xtb_md/concatenate_extxyz/xtb_final_6_trimmed/P6_1600_final.extxyz',
#  ]

# files = [
#     '/gpfs/fs1/home/o/ovoznyy/orhan/scratch/xtb_md/concatenate_extxyz/xdatcar_1_trimmed_to_match_extxyz/XDATCAR_800K',
#     '/gpfs/fs1/home/o/ovoznyy/orhan/scratch/xtb_md/concatenate_extxyz/xdatcar_1_trimmed_to_match_extxyz/XDATCAR_1000K',
#     '/gpfs/fs1/home/o/ovoznyy/orhan/scratch/xtb_md/concatenate_extxyz/xdatcar_1_trimmed_to_match_extxyz/XDATCAR_1200K',
#     '/gpfs/fs1/home/o/ovoznyy/orhan/scratch/xtb_md/concatenate_extxyz/xdatcar_1_trimmed_to_match_extxyz/XDATCAR_1400K',
#     '/gpfs/fs1/home/o/ovoznyy/orhan/scratch/xtb_md/concatenate_extxyz/xdatcar_1_trimmed_to_match_extxyz/XDATCAR_1600K',
# ]

files = [
    '/gpfs/fs1/home/o/ovoznyy/orhan/scratch/trajectories/vasp_long_conced/trimmed_2_final/XDATCAR_600K_2_final',
    '/gpfs/fs1/home/o/ovoznyy/orhan/scratch/trajectories/vasp_long_conced/trimmed_2_final/XDATCAR_800K_2_final',
    '/gpfs/fs1/home/o/ovoznyy/orhan/scratch/trajectories/vasp_long_conced/trimmed_2_final/XDATCAR_1200K_2_final',
    '/gpfs/fs1/home/o/ovoznyy/orhan/scratch/trajectories/vasp_long_conced/trimmed_2_final/XDATCAR_1400K_2_final',
    '/gpfs/fs1/home/o/ovoznyy/orhan/scratch/trajectories/vasp_long_conced/trimmed_2_final/XDATCAR_1600K_2_final',
    '/gpfs/fs1/home/o/ovoznyy/orhan/scratch/trajectories/vasp_long_conced/trimmed_2_final/XDATCAR_1800K_2_final',
    '/gpfs/fs1/home/o/ovoznyy/orhan/scratch/trajectories/vasp_long_conced/trimmed_2_final/XDATCAR_2000K_2_final',
]


temperatures = [600, 800, 1200, 1400, 1600, 1800, 2000]


# Function to get the file type
def get_file_type(file_path):
    if file_path.endswith('.extxyz'):
        return 'extxyz'
    elif 'XDATCAR' in file_path:
        return 'XDATCAR'
    else:
        return 'unknown'


# Get the file type of the first item (assuming all items have the same file type)
input_filetype = get_file_type(files[0])

# Manually define temperatures corresponding to each file
# temperatures = [800, 1000, 1200, 1400, 1600]  # Adjust respectively with input files

# Specify the diffusing species in the simulation
diffusing_species = 'Li'

# Time step between frames in the trajectory (in picoseconds)
time_step = 0.001

# Whether to smooth the calculated mean squared displacements (MSD)
smoothed = False

# Number of initial steps to ignore in the trajectory to allow equilibration
steps_to_ignore = 100

# Number of steps to average over for calculating diffusivity
avg_nsteps = 1000

# Step interval to skip when calculating MSD (i.e., use every 'step_skip' step)
step_skip = 15000

# Number of radial grid points
# This determines the resolution of the radial distribution function.
# A higher value results in finer resolution.
ngrid = 101

# Maximum radius for radial distribution
# This sets the upper limit for the distance over which the radial distribution function is calculated.
# Units are typically in angstroms (Å).
rmax = 10.0

# Standard deviation for Gaussian smearing
# This parameter controls the smoothing applied to the radial distribution function.
# Smaller values result in less smoothing, while larger values provide more smoothing.
sigma = 0.1

# Range of translational vector elements associated with the supercell This defines how many adjacent cells along
# each axis are considered when calculating the distinct part of the van Hove function. A value of 1 includes the
# nearest neighbor cells.
cell_range = 1

tag = f'ign{steps_to_ignore}_skip{step_skip}_avg{avg_nsteps}_{input_filetype}'

# Create a directory named with the job ID in the working directory
job_id = os.environ.get('SLURM_JOB_ID')
output_dir = f"{job_id}_{tag}"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


reference_species = None  # Add reference species if needed
indices = None  # Add specific indices if needed

diff_analyzer = []
diffusivities = []
msd_diffusivities = []
conductivities_manual = []
conductivities_analyzer = []
conversion_factors = []
summaries = []

ase_to_pmg = AseAtomsAdaptor()


def structures_traj(file, steps_to_ignore=False):
    input_filetype = get_file_type(file)
    if input_filetype == 'extxyz':
        structure_list = read(file, index=':', format='extxyz')
    elif input_filetype == 'XDATCAR':
        structure_list = read(file, index=':', format='vasp-xdatcar')
    else:
        raise ValueError(f"Unsupported file type: {input_filetype}")
    if steps_to_ignore:
        structure_list = structure_list[steps_to_ignore:]
    pmg_structures = [ase_to_pmg.get_structure(s) for s in structure_list]
    return pmg_structures


# Function to calculate diffusivity from MSD
def get_diffusivity_from_msd(msd, dt, smoothed='max'):
    msd = np.array(msd)
    dt = np.array(dt)
    slope, intercept, r_value, p_value, std_err = stats.linregress(dt, msd)
    diffusivity = slope / 6  # Assuming 3D diffusion (factor of 2d where d=3)
    return diffusivity, std_err


# Calculate custom Arrhenius values
def custom_arrhenius(temperatures, diffusivities):
    inv_T = 1000 / np.array(temperatures)  # 1/T in 1/K
    ln_D = np.log(diffusivities)
    slope, intercept, r_value, p_value, std_err = stats.linregress(inv_T, ln_D)
    Ea = -slope * 8.617333262145e-5 * 1000000  # eV/K (Boltzmann constant in eV/K) multiplied by 1000 to correct units
    c = np.exp(intercept)
    return inv_T, ln_D, Ea, c, std_err


# Function to write final summary table to output file
def write_table_to_output(file_path, temperatures, diffusivities, msd_diffusivities, conductivities_analyzer_S, conductivities_manual, conversion_factors):
    with open(file_path, "a") as f:
        f.write("\n\nSummary Table:\n")
        headers = ["Property"] + [f"{temp}K" for temp in temperatures]
        rows = [
            ["diffusivities (cm^2/s)"] + diffusivities,
            ["conductivities (S/cm)"] + conductivities_analyzer_S,
            ["conductivities (mS/cm)"] + conductivities_analyzer,
            # ["msd_diffusivities (cm^2/s)"] + msd_diffusivities,
            # ["conductivities_manual"] + conductivities_manual,
            # ["conversion_factors"] + conversion_factors
        ]

        table = tabulate(rows, headers, tablefmt="grid")
        f.write(table + "\n")


# Function to create an Arrhenius plot
def plot_arrhenius(temperatures, diffusivities, output_dir):
    inv_T_p = 1000 / np.array(temperatures)  # 1/T in 1/K
    ln_D_p = np.log(diffusivities)
    slope_p, intercept_p, r_value_p, p_value_p, std_err_p = stats.linregress(inv_T_p, ln_D_p)
    Ea_p = -slope_p * 8.617333262145e-5 * 1000000
    c_p = np.exp(intercept_p)

    fig, ax1 = plt.subplots()

    plt.title(f'Arrhenius_{tag}')

    # Primary axis (bottom) showing 1000/T
    ax1.plot(inv_T_p, ln_D_p, 'o', label='Data')
    ax1.plot(inv_T_p, slope_p * inv_T_p + intercept_p, '-', label=f'Fit')
    ax1.set_xlabel('1000/T (1/K)')
    ax1.set_ylabel('ln(Diffusivity) (ln(cm^2/s))')
    ax1.legend()

    # Define the range for extrapolation including 300 K
    temperatures_with_300K = np.append(temperatures, 300)
    inv_T_extrapolated_range = 1000 / np.array(temperatures_with_300K)
    ln_D_extrapolated_range = slope_p * inv_T_extrapolated_range + intercept_p

    # Plot extrapolated line
    ax1.plot(inv_T_extrapolated_range, ln_D_extrapolated_range, 'r--', color='red')

    # Extrapolate to 300 K
    T_extrapolated = np.array([300])
    inv_T_extrapolated = 1000 / T_extrapolated
    ln_D_extrapolated = slope_p * inv_T_extrapolated + intercept_p

    # Plot extrapolated data
    ax1.plot(inv_T_extrapolated, ln_D_extrapolated, 'rx', color='blue')

    # Adding annotations for extrapolated conductivity and diffusivity in the bottom left corner
    ax1.annotate(
        f'Conductivity 300K: {extrapolated_conductivity / 1000:.3e} S/cm\nDiffusivity 300K: {extrapolated_diffusivity:.3e} cm^2/s\nEa={Ea_p:.3f} meV\nln(Diff) 300K: {np.log(extrapolated_diffusivity):.3f} cm^2/s',
        xy=(0.05, 0.05), xycoords='axes fraction', fontsize=10, color='black', ha='left')

    # Secondary axis (top) showing Temperature in K
    def inv_T_to_T(inv_T):
        return 1000 / inv_T

    def T_to_inv_T(T):
        return 1000 / T

    ax2 = ax1.secondary_xaxis('top', functions=(inv_T_to_T, T_to_inv_T))
    # ax2.set_xlabel('Temperature (K)')
    ax2.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax2.xaxis.set_tick_params(rotation=90, labelsize=8)

    plt.savefig(os.path.join(output_dir, 'arrhenius_plot_custom.png'))
    plt.show()

    return Ea_p, c_p, std_err_p


# Function to extract indices of a specific species
def get_species_indices(structures, species):
    indices = []
    for i, site in enumerate(structures[0]):
        if site.species_string == species:
            indices.append(i)
    return indices


# Function to plot RDF
def plot_rdf(structures, label, ngrid=101, rmax=10.0, sigma=0.1, cell_range=1, output_dir='.', temperature=None):
    # Extract indices for Li atoms
    li_indices = get_species_indices(structures, 'Li')

    # Set both indices and reference_indices to Li atom indices
    indices = li_indices
    reference_indices = li_indices

    rdf = RadialDistributionFunction(structures, indices, reference_indices, ngrid=ngrid, rmax=rmax, sigma=sigma,
                                     cell_range=cell_range)
    rdf_plot = rdf.get_rdf_plot(label=label, xlim=(0.0, rmax), ylim=(-0.005, 3.0), loc_peak=True)
    plt.title(f'rdf_{temperatures[i]}K_{tag}')
    rdf_plot.figure.tight_layout()
    rdf_filename = os.path.join(output_dir, f'rdf_{temperature}K.png')
    rdf_plot.figure.savefig(rdf_filename)
    plt.close(rdf_plot.figure)


# Loop through each file and calculate diffusivity and more
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

    conductivities_analyzer_S = [value / 1000 for value in conductivities_analyzer]

    # structure = analyzer.structure

    # # Calculate conductivity manually using analyzer's diffusivity
    # num_ions = sum([site.species[diffusing_species] for site in structure if diffusing_species in site.species])
    # volume = structure.volume
    # N = num_ions / volume  # Number of ions per unit volume
    # conductivity_manual = 10e24 * (diffusivity_analyzer * 1.60217662e-19 ** 2 * N) / (1.38064852e-23 * analyzer.temperature)
    # conductivities_manual.append(conductivity_manual)

    # Export MSD data to CSV
    msd_filename = os.path.join(output_dir, f"msd_data_{temperatures[i]}K.csv")
    analyzer.export_msdt(msd_filename)

    # # Get conversion factors
    # conversion_factor = get_conversion_factor(structures, diffusing_species, temperatures[i])
    # conversion_factors.append(conversion_factor)

    # Get summary dictionary
    summary = analyzer.get_summary_dict(include_msd_t=False, include_mscd_t=False)
    summaries.append(summary)

    # Get msd plot from analyzer
    msd_plot_anal = analyzer.get_msd_plot(mode='species')
    plt.title(f'MSD_species_{temperatures[i]}K_{tag}')
    msd_plot_anal.figure.tight_layout()
    msd_plot_anal.figure.savefig(os.path.join(output_dir, f'msd_species_{temperatures[i]}K.png'))
    plt.close(msd_plot_anal.figure)

    msd_plot_anal = analyzer.get_msd_plot(mode='sites')
    plt.title(f'MSD_sites_{temperatures[i]}K_{tag}')
    msd_plot_anal.figure.tight_layout()
    msd_plot_anal.figure.savefig(os.path.join(output_dir, f'msd_sites_{temperatures[i]}K.png'))
    plt.close(msd_plot_anal.figure)

    msd_plot_anal = analyzer.get_msd_plot(mode='direction')
    plt.title(f'MSD_direction_{temperatures[i]}K_{tag}')
    msd_plot_anal.figure.tight_layout()
    msd_plot_anal.figure.savefig(os.path.join(output_dir, f'msd_direction_{temperatures[i]}K.png'))
    plt.close(msd_plot_anal.figure)

    msd_plot_anal = analyzer.get_msd_plot(mode='mscd')
    plt.title(f'MSCD_{temperatures[i]}K_{tag}')
    msd_plot_anal.figure.tight_layout()
    msd_plot_anal.figure.savefig(os.path.join(output_dir, f'msd_mscd_{temperatures[i]}K.png'))
    plt.close(msd_plot_anal.figure)

    framework_rms_plot = analyzer.get_framework_rms_plot(granularity=200, matching_s=None)
    plt.title(f'framework_rms_{temperatures[i]}K_{tag}')
    framework_rms_plot.figure.tight_layout()
    framework_rms_plot.figure.savefig(os.path.join(output_dir, f'framework_rms_{temperatures[i]}K.png'))
    plt.close(framework_rms_plot.figure)

    # # Calculate diffusivity from MSD data
    # if os.path.exists(msd_filename):
    #     time = []
    #     msd = []
    #     with open(msd_filename, 'r') as file:
    #         reader = csv.reader(file)
    #         next(reader)  # Skip header
    #         for row in reader:
    #             time.append(float(row[0]))
    #             msd.append(float(row[1]))
    #
    #     diffusivity_msd, std_err = get_diffusivity_from_msd(msd, time)
    #     msd_diffusivities.append(diffusivity_msd)
    # else:
    #     msd_diffusivities.append(None)


# Calculate Arrhenius values using all temperatures and diffusivities from analyzer
arrhenius_values = fit_arrhenius(temperatures, diffusivities)

# Calculate custom Arrhenius values
inv_T, ln_D, Ea, c, std_err = custom_arrhenius(temperatures, diffusivities)

# Plot analyzer Arrhenius
arr_plot = get_arrhenius_plot(temperatures, diffusivities)
plt.title(f'Arrhenius_{tag}')
arr_plot.figure.tight_layout()
arr_plot.figure.savefig(os.path.join(output_dir, 'arrhenius_plot.png'))
plt.close(arr_plot.figure)

# Calculate extrapolated diffusivity and conductivity
if len(files) > 1:
    if input_filetype == 'extxyz':
        structure = read(files[0], index=0, format='extxyz')
    elif input_filetype == 'XDATCAR':
        structure = read(files[0], index=0, format='vasp-xdatcar')
    else:
        raise ValueError(f"Unsupported file type: {input_filetype}")

    structure = ase_to_pmg.get_structure(structure)
    extrapolated_diffusivity = get_extrapolated_diffusivity(temperatures, diffusivities, new_temp=300)
    extrapolated_conductivity = get_extrapolated_conductivity(temperatures, diffusivities, new_temp=300,
                                                              structure=structure, species='Li')
else:
    extrapolated_diffusivity = None
    extrapolated_conductivity = None


# Plot custom Arrhenius
Ea_p, c_p, std_err_p = plot_arrhenius(temperatures, diffusivities, output_dir)


# Write results to a file
output_file = os.path.join(output_dir, "output.txt")
with open(output_file, "w") as f:

    if extrapolated_diffusivity and extrapolated_conductivity:
        f.write(f"Extrapolated Diffusivity at 300K: {extrapolated_diffusivity:.3e} cm^2/s\n")
        f.write(f"Extrapolated Conductivity at 300K: {extrapolated_conductivity/1000:.3e} S/cm\n")
        f.write(f"Extrapolated Conductivity at 300K: {extrapolated_conductivity:.3e} mS/cm\n")
    else:
        f.write("\nOnly one file provided, extrapolated diffusivity and conductivity not calculated.\n")

    f.write(f"avg_nsteps = {avg_nsteps}\n")
    f.write(f"steps_to_ignore = {steps_to_ignore}\n")
    f.write(f"step_skip = {step_skip}\n")
    f.write(f"input_filetype = {input_filetype}\n")

    # Print input files
    f.write("\n\nInput files:\n")
    for file in files:
        f.write(f"{file}\n")

    # Arrhenius values from analyzer
    f.write("\n\nArrhenius Values from Analyzer:\n")
    f.write(f"Activation Energy: {arrhenius_values[0] * 1000:.3f} meV\n")
    f.write(f"Pre-exponential Factor: {arrhenius_values[1]:.3e} cm^2/s\n")
    if arrhenius_values[2] is not None:
        f.write(f"Standard Error: {arrhenius_values[2]:.3f} meV\n")
    else:
        f.write("Standard Error: Not available\n")

    # Custom Arrhenius values
    f.write("\n\nCustom Arrhenius Values:\n")
    f.write(f"Activation Energy: {Ea:.3f} meV\n")
    f.write(f"Pre-exponential Factor: {c:.3e} cm^2/s\n")
    if std_err is not None:
        f.write(f"Standard Error: {std_err:.3f} meV\n")
    else:
        f.write("Standard Error: Not available\n")

    # Diffusion summary
    f.write("\n\nDiffusion Summary:\n")
    for key in summaries[0].keys():
        f.write(f"{key}:\n")
        for temp, summary in zip(temperatures, summaries):
            f.write(f"  {temp}K: {summary[key]}\n")


# Plot Diffusivity
plt.figure()
plt.plot(temperatures, diffusivities, 'o-', label='Diffusivities')
plt.xlabel('Temperature (K)')
plt.ylabel('Diffusivity (cm^2/s)')
plt.title(f'Diffusivity_{tag}')
ax = plt.gca()
ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
ax.yaxis.get_major_formatter().set_scientific(True)
ax.yaxis.get_major_formatter().set_powerlimits((0, 0))
plt.legend()
plt.savefig(os.path.join(output_dir, 'diffusivities.png'))
plt.close()

# Plot Conductivity
plt.figure()
plt.plot(temperatures, conductivities_analyzer_S, 'o-')
plt.xlabel('Temperature (K)')
plt.ylabel('Conductivity (S/cm)')
plt.title(f'Conductivity_{tag}')
ax = plt.gca()
ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
ax.yaxis.get_major_formatter().set_scientific(True)
ax.yaxis.get_major_formatter().set_powerlimits((0, 0))
plt.savefig(os.path.join(output_dir, 'conductivities.png'))
plt.close()

# # Plot MSD from CSV files and calculate diffusivity from MSD
# for temp in temperatures:
#     msd_filename = os.path.join(output_dir, f"msd_data_{temp}K.csv")
#     if os.path.exists(msd_filename):
#         time = []
#         msd = []
#         msd_a = []
#         msd_b = []
#         msd_c = []
#         mscd = []
#         with open(msd_filename, 'r') as file:
#             reader = csv.reader(file)
#             next(reader)  # Skip header
#             for row in reader:
#                 time.append(float(row[0]))
#                 msd.append(float(row[1]))
#                 msd_a.append(float(row[2]))
#                 msd_b.append(float(row[3]))
#                 msd_c.append(float(row[4]))
#                 mscd.append(float(row[5]))
#
#         plt.figure()
#         plt.plot(time, msd, label='MSD', linewidth=0.2)
#         plt.plot(time, msd_a, label='MSD_a', linewidth=0.2)
#         plt.plot(time, msd_b, label='MSD_b', linewidth=0.2)
#         plt.plot(time, msd_c, label='MSD_c', linewidth=0.2)
#         plt.plot(time, mscd, label='MSCD', linewidth=0.2)
#         plt.xlabel('Time (ps)')
#         plt.ylabel('MSD (Å^2)')
#         plt.title(f'MSD vs Time at {temp}K')
#         plt.legend()
#         plt.savefig(os.path.join(output_dir, f'msd_plot_{temp}K.png'))
#         plt.show()
#     else:
#         print(f"MSD file {msd_filename} not found.")

# Call summary table function print in output file
write_table_to_output(output_file, temperatures, diffusivities, msd_diffusivities, conductivities_analyzer_S, conductivities_manual, conversion_factors)

# Get RDF plot
for i, file in enumerate(files):
    structures = structures_traj(file, steps_to_ignore)
    plot_rdf(structures, label=f'{temperatures[i]}K', ngrid=ngrid, rmax=rmax, sigma=sigma, cell_range=cell_range, output_dir=output_dir, temperature=temperatures[i])

# Perform Probability Density Analysis for each temperature
for i, analyzer in enumerate(diff_analyzer):
    structure = analyzer.structure
    trajectories = [s.frac_coords for s in analyzer.get_drift_corrected_structures()]
    pda = ProbabilityDensityAnalysis(structure, trajectories, species="Li")
    output_filename = os.path.join(output_dir, f"CHGCAR_{temperatures[i]}K.vasp")
    pda.to_chgcar(output_filename)

# Perform Van Hove Analysis for each temperature
for i, analyzer in enumerate(diff_analyzer):
    try:
        # Set the indices for Li atoms only
        li_indices = get_species_indices([analyzer.structure], 'Li')

        van_hove = VanHoveAnalysis(
            analyzer,
            avg_nsteps=avg_nsteps,
            ngrid=ngrid,
            rmax=rmax,
            step_skip=step_skip,
            sigma=sigma,
            cell_range=cell_range,
            species=diffusing_species,
            reference_species=diffusing_species,  # Reference species is also Li for Li-Li pairs
            indices=li_indices,  # Indices for Li atoms
        )

        # Save 3d plot distinct
        van_hove_3d_plot = van_hove.get_3d_plot(mode='distinct')
        plt.title(f'van_hove_distinct_{temperatures[i]}K_{tag}')
        van_hove_3d_plot_filename = os.path.join(output_dir, f'van_hove_3d_distinct_{temperatures[i]}.png')
        van_hove_3d_plot.figure.tight_layout()
        van_hove_3d_plot.figure.savefig(van_hove_3d_plot_filename)
        plt.close(van_hove_3d_plot.figure)

        # Save 3d plot self
        van_hove_3d_plot = van_hove.get_3d_plot(mode='self')
        plt.title(f'van_hove_self_{temperatures[i]}K_{tag}')
        van_hove_3d_plot_filename = os.path.join(output_dir, f'van_hove_3d_self_{temperatures[i]}.png')
        van_hove_3d_plot.figure.tight_layout()
        van_hove_3d_plot.figure.savefig(van_hove_3d_plot_filename)
        plt.close(van_hove_3d_plot.figure)

        # Save 1d plot distinct
        van_hove_1d_plot = van_hove.get_1d_plot(mode='distinct', times=[10, 50, 100])
        plt.title(f'van_hove_distinct_{temperatures[i]}K_{tag}')
        van_hove_1d_plot_filename = os.path.join(output_dir, f'van_hove_1d_distinct_{temperatures[i]}.png')
        van_hove_1d_plot.figure.tight_layout()
        van_hove_1d_plot.figure.savefig(van_hove_1d_plot_filename)
        plt.close(van_hove_1d_plot.figure)

        # Save 1d plot self
        van_hove_1d_plot = van_hove.get_1d_plot(mode='self', times=[10, 50, 100])
        plt.title(f'van_hove_self_{temperatures[i]}K_{tag}')
        van_hove_1d_plot_filename = os.path.join(output_dir, f'van_hove_1d_self_{temperatures[i]}.png')
        van_hove_1d_plot.figure.tight_layout()
        van_hove_1d_plot.figure.savefig(van_hove_1d_plot_filename)
        plt.close(van_hove_1d_plot.figure)

    except RuntimeError as e:
        print(f"RuntimeError for analyzer {i} at {temperatures[i]}K: {e}")
    except Exception as e:
        print(f"Unexpected error for analyzer {i} at {temperatures[i]}K: {e}")

