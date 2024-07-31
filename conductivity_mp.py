import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pymatgen.analysis.diffusion.analyzer import (DiffusionAnalyzer, get_extrapolated_conductivity,
                                                  get_extrapolated_diffusivity, get_conversion_factor, fit_arrhenius,
                                                  get_arrhenius_plot)
from pymatgen.analysis.diffusion.aimd.pathway import ProbabilityDensityAnalysis
from pymatgen.analysis.diffusion.aimd.van_hove import VanHoveAnalysis
from pymatgen.analysis.diffusion.aimd.rdf import RadialDistributionFunction
from pymatgen.io.ase import AseAtomsAdaptor
from ase.io import read
import os
import numpy as np
import csv
from scipy import stats
from tabulate import tabulate
import concurrent.futures

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

def get_file_type(file_path):
    if file_path.endswith('.extxyz'):
        return 'extxyz'
    elif 'XDATCAR' in file_path:
        return 'XDATCAR'
    else:
        return 'unknown'


input_filetype = get_file_type(files[0])
# temperatures = [600, 1000, 1200, 1400, 1600, 1800]
diffusing_species = 'Li'
time_step = 0.001
smoothed = False
steps_to_ignore = 15000
avg_nsteps = 1000
step_skip = 100
ngrid = 101
rmax = 10.0
sigma = 0.1
cell_range = 1

tag = f'ign{steps_to_ignore}_skip{step_skip}_avg{avg_nsteps}_{input_filetype}'

job_id = os.environ.get('SLURM_JOB_ID')
output_dir = f"{job_id}_{tag}"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


reference_species = None
indices = None

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


def get_diffusivity_from_msd(msd, dt, smoothed='max'):
    msd = np.array(msd)
    dt = np.array(dt)
    slope, intercept, r_value, p_value, std_err = stats.linregress(dt, msd)
    diffusivity = slope / 6
    return diffusivity, std_err


def custom_arrhenius(temperatures, diffusivities):
    inv_T = 1000 / np.array(temperatures)
    ln_D = np.log(diffusivities)
    slope, intercept, r_value, p_value, std_err = stats.linregress(inv_T, ln_D)
    Ea = -slope * 8.617333262145e-5 * 1000000
    c = np.exp(intercept)
    return inv_T, ln_D, Ea, c, std_err


def write_table_to_output(file_path, temperatures, diffusivities, msd_diffusivities, conductivities_analyzer_S, conductivities_manual, conversion_factors):
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
    ax2.set_xlabel('Temperature (K)')
    ax2.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax2.xaxis.set_tick_params(rotation=90, labelsize=8)

    plt.savefig(os.path.join(output_dir, 'arrhenius_plot_custom.png'))
    plt.show()

    return Ea_p, c_p, std_err_p


def get_species_indices(structures, species):
    indices = []
    for i, site in enumerate(structures[0]):
        if site.species_string == species:
            indices.append(i)
    return indices


def plot_rdf(structures, label, ngrid=101, rmax=10.0, sigma=0.1, cell_range=1, output_dir='.', temperature=None):
    li_indices = get_species_indices(structures, 'Li')
    indices = li_indices
    reference_indices = li_indices

    rdf = RadialDistributionFunction(structures, indices, reference_indices, ngrid=ngrid, rmax=rmax, sigma=sigma,
                                     cell_range=cell_range)
    rdf_plot = rdf.get_rdf_plot(label=label, xlim=(0.0, rmax), ylim=(-0.005, 3.0), loc_peak=True)
    plt.title(f'rdf_{temperature}K_{tag}')
    rdf_plot.figure.tight_layout()
    rdf_filename = os.path.join(output_dir, f'rdf_{temperature}K.png')
    rdf_plot.figure.savefig(rdf_filename)
    plt.close(rdf_plot.figure)


def process_file(index):
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
        diff_analyzer.append(analyzer)
        diffusivity_analyzer = analyzer.diffusivity
        diffusivities.append(diffusivity_analyzer)
        conductivity_analyzer = analyzer.conductivity
        conductivities_analyzer.append(conductivity_analyzer)
        conductivities_analyzer_S = [value / 1000 for value in conductivities_analyzer]
        msd_filename = os.path.join(output_dir, f"msd_data_{temperature}K.csv")
        analyzer.export_msdt(msd_filename)

        summary = analyzer.get_summary_dict(include_msd_t=False, include_mscd_t=False)
        summaries.append(summary)

        msd_plot_anal = analyzer.get_msd_plot(mode='species')
        plt.title(f'MSD_species_{temperature}K_{tag}')
        msd_plot_anal.figure.tight_layout()
        msd_plot_anal.figure.savefig(os.path.join(output_dir, f'msd_species_{temperature}K.png'))
        plt.close(msd_plot_anal.figure)

        msd_plot_anal = analyzer.get_msd_plot(mode='sites')
        plt.title(f'MSD_sites_{temperature}K_{tag}')
        msd_plot_anal.figure.tight_layout()
        msd_plot_anal.figure.savefig(os.path.join(output_dir, f'msd_sites_{temperature}K.png'))
        plt.close(msd_plot_anal.figure)

        msd_plot_anal = analyzer.get_msd_plot(mode='direction')
        plt.title(f'MSD_direction_{temperature}K_{tag}')
        msd_plot_anal.figure.tight_layout()
        msd_plot_anal.figure.savefig(os.path.join(output_dir, f'msd_direction_{temperature}K.png'))
        plt.close(msd_plot_anal.figure)

        msd_plot_anal = analyzer.get_msd_plot(mode='mscd')
        plt.title(f'MSCD_{temperature}K_{tag}')
        msd_plot_anal.figure.tight_layout()
        msd_plot_anal.figure.savefig(os.path.join(output_dir, f'msd_mscd_{temperature}K.png'))
        plt.close(msd_plot_anal.figure)

        framework_rms_plot = analyzer.get_framework_rms_plot(granularity=200, matching_s=None)
        plt.title(f'framework_rms_{temperature}K_{tag}')
        framework_rms_plot.figure.tight_layout()
        framework_rms_plot.figure.savefig(os.path.join(output_dir, f'framework_rms_{temperature}K.png'))
        plt.close(framework_rms_plot.figure)

        plot_rdf(structures, label=f'{temperature}K', ngrid=ngrid, rmax=rmax, sigma=sigma, cell_range=cell_range,
                 output_dir=output_dir, temperature=temperature)

        structure = analyzer.structure
        trajectories = [s.frac_coords for s in analyzer.get_drift_corrected_structures()]
        pda = ProbabilityDensityAnalysis(structure, trajectories, species="Li")
        output_filename = os.path.join(output_dir, f"CHGCAR_{temperature}K.vasp")
        pda.to_chgcar(output_filename)

        try:
            li_indices = get_species_indices([analyzer.structure], 'Li')
            van_hove = VanHoveAnalysis(analyzer, avg_nsteps=avg_nsteps, ngrid=ngrid, rmax=rmax, step_skip=step_skip,
                                       sigma=sigma, cell_range=cell_range, species=diffusing_species,
                                       reference_species=diffusing_species, indices=li_indices)
            van_hove_3d_plot = van_hove.get_3d_plot(mode='distinct')
            plt.title(f'van_hove_distinct_{temperature}K_{tag}')
            van_hove_3d_plot_filename = os.path.join(output_dir, f'van_hove_3d_distinct_{temperature}.png')
            van_hove_3d_plot.figure.tight_layout()
            van_hove_3d_plot.figure.savefig(van_hove_3d_plot_filename)
            plt.close(van_hove_3d_plot.figure)

            van_hove_3d_plot = van_hove.get_3d_plot(mode='self')
            plt.title(f'van_hove_self_{temperature}K_{tag}')
            van_hove_3d_plot_filename = os.path.join(output_dir, f'van_hove_3d_self_{temperature}.png')
            van_hove_3d_plot.figure.tight_layout()
            van_hove_3d_plot.figure.savefig(van_hove_3d_plot_filename)
            plt.close(van_hove_3d_plot.figure)

            van_hove_1d_plot = van_hove.get_1d_plot(mode='distinct', times=[10, 50, 100])
            plt.title(f'van_hove_distinct_{temperature}K_{tag}')
            van_hove_1d_plot_filename = os.path.join(output_dir, f'van_hove_1d_distinct_{temperature}.png')
            van_hove_1d_plot.figure.tight_layout()
            van_hove_1d_plot.figure.savefig(van_hove_1d_plot_filename)
            plt.close(van_hove_1d_plot.figure)

            van_hove_1d_plot = van_hove.get_1d_plot(mode='self', times=[10, 50, 100])
            plt.title(f'van_hove_self_{temperature}K_{tag}')
            van_hove_1d_plot_filename = os.path.join(output_dir, f'van_hove_1d_self_{temperature}.png')
            van_hove_1d_plot.figure.tight_layout()
            van_hove_1d_plot.figure.savefig(van_hove_1d_plot_filename)
            plt.close(van_hove_1d_plot.figure)
        except RuntimeError as e:
            print(f"RuntimeError for analyzer {index} at {temperature}K: {e}")
        except Exception as e:
            print(f"Unexpected error for analyzer {index} at {temperature}K: {e}")

    except Exception as e:
        print(f"Unexpected error for analyzer {index} at {temperature}K: {e}")

    return {
        'diffusivity': diffusivity_analyzer,
        'conductivity': conductivity_analyzer,
        'conductivities_analyzer_S': conductivities_analyzer_S,
        'summary': summary,
        'temperature': temperature
    }


with concurrent.futures.ProcessPoolExecutor(max_workers=5) as executor:
    results = list(executor.map(process_file, range(len(files))))

# Process the collected results
conductivities_analyzer_S = []
for result in results:
    diffusivities.append(result['diffusivity'])
    conductivities_analyzer.append(result['conductivity'])
    conductivities_analyzer_S.append(result['conductivities_analyzer_S'])
    summaries.append(result['summary'])

arrhenius_values = fit_arrhenius(temperatures, diffusivities)
inv_T, ln_D, Ea, c, std_err = custom_arrhenius(temperatures, diffusivities)

arr_plot = get_arrhenius_plot(temperatures, diffusivities)
plt.title(f'Arrhenius_{tag}')
arr_plot.figure.tight_layout()
arr_plot.figure.savefig(os.path.join(output_dir, 'arrhenius_plot.png'))
plt.close(arr_plot.figure)

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

write_table_to_output(output_file, temperatures, diffusivities, msd_diffusivities, conductivities_analyzer_S,
                      conductivities_manual, conversion_factors)
