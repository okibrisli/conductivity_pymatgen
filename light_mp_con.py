import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pymatgen.analysis.diffusion.analyzer import DiffusionAnalyzer, get_extrapolated_conductivity, \
    get_extrapolated_diffusivity, fit_arrhenius
from pymatgen.io.ase import AseAtomsAdaptor
from ase.io import read
import os
import numpy as np
import csv
from scipy import stats
from tabulate import tabulate
import concurrent.futures
import psutil
import traceback

time_step = 0.001
steps_to_ignore = 15000
step_skip = 100
avg_nsteps = 1000
smoothed = True

# files = [
#     '/gpfs/fs1/home/o/ovoznyy/orhan/scratch/trajectories/A_xTB/L6_800_final.extxyz',
#     '/gpfs/fs1/home/o/ovoznyy/orhan/scratch/trajectories/A_xTB/M6_1000_final.extxyz',
#     '/gpfs/fs1/home/o/ovoznyy/orhan/scratch/trajectories/A_xTB/N6_1200_final.extxyz',
#     '/gpfs/fs1/home/o/ovoznyy/orhan/scratch/trajectories/A_xTB/O6_1400_final.extxyz',
#     '/gpfs/fs1/home/o/ovoznyy/orhan/scratch/trajectories/A_xTB/P6_1600_final.extxyz',
# ]
# temperatures = [800, 1000, 1200, 1400, 1600]


# files = [
#     '/gpfs/fs1/home/o/ovoznyy/orhan/scratch/trajectories/A_VASP_214761steps/XDATCAR_800K',
#     '/gpfs/fs1/home/o/ovoznyy/orhan/scratch/trajectories/A_VASP_214761steps/XDATCAR_1000K',
#     '/gpfs/fs1/home/o/ovoznyy/orhan/scratch/trajectories/A_VASP_214761steps/XDATCAR_1200K',
#     '/gpfs/fs1/home/o/ovoznyy/orhan/scratch/trajectories/A_VASP_214761steps/XDATCAR_1400K',
#     '/gpfs/fs1/home/o/ovoznyy/orhan/scratch/trajectories/A_VASP_214761steps/XDATCAR_1600K',
# ]
# temperatures = [800, 1000, 1200, 1400, 1600]

files = [
    '/gpfs/fs1/home/o/ovoznyy/orhan/scratch/trajectories/vasp_long_conced/trimmed_2_final/XDATCAR_600K_2_final',
    '/gpfs/fs1/home/o/ovoznyy/orhan/scratch/trajectories/vasp_long_conced/trimmed_2_final/XDATCAR_800K_2_final',
    '/gpfs/fs1/home/o/ovoznyy/orhan/scratch/trajectories/vasp_long_conced/trimmed_2_final/XDATCAR_1200K_2_final',
    '/gpfs/fs1/home/o/ovoznyy/orhan/scratch/trajectories/vasp_long_conced/trimmed_2_final/XDATCAR_1400K_2_final',
    '/gpfs/fs1/home/o/ovoznyy/orhan/scratch/trajectories/vasp_long_conced/trimmed_2_final/XDATCAR_1600K_2_final',
    '/gpfs/fs1/home/o/ovoznyy/orhan/scratch/trajectories/vasp_long_conced/trimmed_2_final/XDATCAR_1800K_2_final',
    '/gpfs/fs1/home/o/ovoznyy/orhan/scratch/trajectories/vasp_long_conced/trimmed_2_final/XDATCAR_2000K_2_final',
]
temperatures = [
                600,
                800,
                1200,
                1400,
                1600,
                1800,
                2000
]


def get_file_type(file_path):
    if file_path.endswith('.extxyz'):
        return 'extxyz'
    elif 'XDATCAR' in file_path:
        return 'XDATCAR'
    else:
        return 'unknown'


input_filetype = get_file_type(files[0])
diffusing_species = 'Li'
ngrid = 101
rmax = 10.0
sigma = 0.1
cell_range = 1
tag = f'ign{steps_to_ignore}_skip{step_skip}_avg{avg_nsteps}_{input_filetype}'
job_id = os.environ.get('SLURM_JOB_ID')
output_dir = f"{job_id}_{tag}"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

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

def custom_arrhenius(temperatures, diffusivities):
    inv_T = 1000 / np.array(temperatures)
    ln_D = np.log(diffusivities)
    slope, intercept, r_value, p_value, std_err = stats.linregress(inv_T, ln_D)
    Ea = -slope * 8.617333262145e-5 * 1000000
    c = np.exp(intercept)
    return inv_T, ln_D, Ea, c, std_err

def write_table_to_output(file_path, temperatures, diffusivities, msd_diffusivities, conductivities_analyzer_S,
                          conductivities_manual, conversion_factors):
    with open(file_path, "a") as f:
        f.write("\n\nSummary Table:\n")
        headers = ["Property"] + [f"{temp}K" for temp in temperatures]
        rows = [
            ["diffusivities (cm^2/s)"] + diffusivities,
            ["conductivities (S/cm)"] + conductivities_analyzer_S,
            ["conductivities (mS/cm)"] + conductivities_analyzer,
        ]
        table = tabulate(rows, headers, tablefmt="grid")
        f.write(table + "\n")

def plot_arrhenius(temperatures, diffusivities, output_dir):
    inv_T_p = 1000 / np.array(temperatures)
    ln_D_p = np.log(diffusivities)
    slope_p, intercept_p, r_value_p, p_value_p, std_err_p = stats.linregress(inv_T_p, ln_D_p)
    Ea_p = -slope_p * 8.617333262145e-5 * 1000000
    c_p = np.exp(intercept_p)

    fig, ax1 = plt.subplots()

    plt.title(f'Arrhenius_{tag}')
    ax1.plot(inv_T_p, ln_D_p, 'o', label='Data')
    ax1.plot(inv_T_p, slope_p * inv_T_p + intercept_p, '-', label=f'Fit')
    ax1.set_xlabel('1000/T (1/K)')
    ax1.set_ylabel('ln(Diffusivity) (ln(cm^2/s))')
    ax1.legend()

    temperatures_with_300K = np.append(temperatures, 300)
    inv_T_extrapolated_range = 1000 / np.array(temperatures_with_300K)
    ln_D_extrapolated_range = slope_p * inv_T_extrapolated_range + intercept_p

    ax1.plot(inv_T_extrapolated_range, ln_D_extrapolated_range, 'r--', color='red')

    T_extrapolated = np.array([300])
    inv_T_extrapolated = 1000 / T_extrapolated
    ln_D_extrapolated = slope_p * inv_T_extrapolated + intercept_p

    ax1.plot(inv_T_extrapolated, ln_D_extrapolated, 'rx', color='blue')
    ax1.annotate(
        f'Conductivity 300K: {extrapolated_conductivity / 1000:.3e} S/cm\nDiffusivity 300K: {extrapolated_diffusivity:.3e} cm^2/s\nEa={Ea_p:.3f} meV\nln(Diff) 300K: {np.log(extrapolated_diffusivity):.3f} cm^2/s',
        xy=(0.05, 0.05), xycoords='axes fraction', fontsize=10, color='black', ha='left')

    def inv_T_to_T(inv_T):
        return 1000 / inv_T

    def T_to_inv_T(T):
        return 1000 / T

    ax2 = ax1.secondary_xaxis('top', functions=(inv_T_to_T, T_to_inv_T))
    # ax2.set_xlabel('Temperature (K)')
    ax2.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax2.xaxis.set_tick_params(rotation=90, labelsize=8)
    fig.tight_layout()

    plt.savefig(os.path.join(output_dir, 'arrhenius_plot_custom.png'))
    plt.show()

    return Ea_p, c_p, std_err_p

def log_memory_usage(tag):
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    print(f"[{tag}] Memory Usage: {memory_info.rss / 1e6} MB")

def process_file(index):
    log_memory_usage(f"Start process_file {index}")
    file = files[index]
    temperature = temperatures[index]
    structures = structures_traj(file, steps_to_ignore)
    summary = {}

    try:
        analyzer = DiffusionAnalyzer.from_structures(structures,
                                                     specie=diffusing_species,
                                                     temperature=temperature,
                                                     time_step=time_step,
                                                     smoothed=smoothed,
                                                     step_skip=step_skip,
                                                     avg_nsteps=avg_nsteps)
        diffusivity_analyzer = analyzer.diffusivity
        conductivity_analyzer = analyzer.conductivity
        msd_filename = os.path.join(output_dir, f"msd_data_{temperature}K.csv")
        analyzer.export_msdt(msd_filename)

        msd_plot_anal = analyzer.get_msd_plot(mode='direction')
        plt.title(f'MSD_direction_{temperature}K_{tag}')
        msd_plot_anal.figure.tight_layout()
        msd_plot_anal.figure.savefig(os.path.join(output_dir, f'msd_direction_{temperature}K.png'))
        plt.close(msd_plot_anal.figure)

        summary = analyzer.get_summary_dict(include_msd_t=False, include_mscd_t=False)

        log_memory_usage(f"End process_file {index}")
        return {
            'diffusivity': diffusivity_analyzer,
            'conductivity': conductivity_analyzer,
            'conductivities_analyzer_S': conductivity_analyzer / 1000,
            'summary': summary
        }
    except Exception as e:
        print(f"Error processing file {file}: {e}")
        traceback.print_exc()
        return None

with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:  # Adjust max_workers if needed
    results = list(executor.map(process_file, range(len(files))))

# Process the collected results
diffusivities = []
conductivities_analyzer = []
conductivities_analyzer_S = []
summaries = []

for result in results:
    if result is not None:
        diffusivities.append(result['diffusivity'])
        conductivities_analyzer.append(result['conductivity'])
        conductivities_analyzer_S.append(result['conductivities_analyzer_S'])
        summaries.append(result['summary'])

arrhenius_values = fit_arrhenius(temperatures, diffusivities)
inv_T, ln_D, Ea, c, std_err = custom_arrhenius(temperatures, diffusivities)

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

Ea_p, c_p, std_err_p = plot_arrhenius(temperatures, diffusivities, output_dir)

output_file = os.path.join(output_dir, "output.txt")
with open(output_file, "w") as f:
    if extrapolated_diffusivity and extrapolated_conductivity:
        f.write(f"Extrapolated Diffusivity at 300K: {extrapolated_diffusivity:.3e} cm^2/s\n")
        f.write(f"Extrapolated Conductivity at 300K: {extrapolated_conductivity / 1000:.3e} S/cm\n")
        f.write(f"Extrapolated Conductivity at 300K: {extrapolated_conductivity:.3e} mS/cm\n")
    else:
        f.write("\nOnly one file provided, extrapolated diffusivity and conductivity not calculated.\n")

    f.write(f"avg_nsteps = {avg_nsteps}\n")
    f.write(f"steps_to_ignore = {steps_to_ignore}\n")
    f.write(f"step_skip = {step_skip}\n")
    f.write(f"input_filetype = {input_filetype}\n")

    f.write("\n\nInput files:\n")
    for file in files:
        f.write(f"{file}\n")

    f.write("\n\nArrhenius Values from Analyzer:\n")
    f.write(f"Activation Energy: {arrhenius_values[0] * 1000:.3f} meV\n")
    f.write(f"Pre-exponential Factor: {arrhenius_values[1]:.3e} cm^2/s\n")
    if arrhenius_values[2] is not None:
        f.write(f"Standard Error: {arrhenius_values[2]:.3f} meV\n")
    else:
        f.write("Standard Error: Not available\n")

    f.write("\n\nCustom Arrhenius Values:\n")
    f.write(f"Activation Energy: {Ea:.3f} meV\n")
    f.write(f"Pre-exponential Factor: {c:.3e} cm^2/s\n")
    if std_err is not None:
        f.write(f"Standard Error: {std_err:.3f} meV\n")
    else:
        f.write("Standard Error: Not available\n")

    f.write("\n\nDiffusion Summary:\n")
    for key in summaries[0].keys():
        f.write(f"{key}:\n")
        for temp, summary in zip(temperatures, summaries):
            f.write(f"  {temp}K: {summary[key]}\n")

plt.figure()
plt.plot(temperatures, diffusivities, 'o-')
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

write_table_to_output(output_file, temperatures, diffusivities, [], conductivities_analyzer_S, [], [])