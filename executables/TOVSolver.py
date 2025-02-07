# Standard Libraries ------------
import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
import h5py
# -------------------------------
### Module for using RePrimAnd
import pyreprimand as pyr
# -------------------------------
# Non-standard Libraries
import bilby
# -------------------------------

### Unit conversion/Constants
gcm3_to_dynecm2=8.9875e20 
MeV_to_gcm3 = 1.7827e12 
dynecm2_to_MeVFermi = 1.6022e33
gcm3_to_fm4 = 3.5178e14
sat_dens = 2.8*(np.power(10.0,14.)) ### CGS
c = bilby.core.utils.speed_of_light

## Here we append M-R information to the EoS pressure density posterior samples

### EoS samples
# eos_samples = h5py.File("/home/sunny.ng/TOVSol/MMGP_full_covariance.h5", "r+")
# num_eos = len(eos_samples["eos"])
# eos_to_be_used = np.arange(num_eos)
# ### Enumerating Pressure-Density Relations for iteration
# micro_data = {eos_num : np.array(eos_samples['eos'][eos_id]) for eos_num, eos_id in enumerate(eos_samples['eos'])}

def monotonicity_check(press, edens, bary_dens):
    press_inc = np.diff(press)
    while any(press_inc<=0):
        zero_indices = np.where(press_inc <= 0)[0]
        edens = np.delete(edens,zero_indices)
        bary_dens = np.delete(bary_dens,zero_indices)
        press = np.delete(press,zero_indices)
        press_inc = np.diff(press)
    mono_eos = {"pressurec2":press,"energy_densityc2":edens,"baryon_density":bary_dens}
    return mono_eos

### Need to convert everything to geometric units!
def cgs_density_to_geometric(rho):
    return (rho/2.8e14 * .00045)

def eos_instance(pressure, rho, eps, eps_0):
    rho_min = rho[1] ### Avoiding floating point error, value mismatch, 
    rho_max = rho[-2] ### currently rho[0] raises a "not within range error" 
    rng = pyr.range(rho_min, rho_max)
    eos = pyr.make_eos_barotr_spline(rho = np.array(rho), press = np.array(pressure), csnd = np.sqrt(np.array(np.gradient(pressure, eps))),
                                     temp = [], efrac = [], eps_0 = eps_0, n_poly = (3.), ### n_poly = 3 --> adiabatic index of 4/3
                                     rg_rho = rng, ### Need to change units from cgs to SI
                                     units = pyr.units(1.,1.,1.), 
                                     pts_per_mag = 200)
    
    return eos
    
def TOVSolve(pressure, rho, eps, eps_0, min_central_dens = sat_dens, max_central_dens = 10*sat_dens):
    
    ### returns an eos_barotrop object (expects geometric units)
    eos = eos_instance(pressure, rho, eps, eps_0)
    ### find central density corresponding to maximum mass (in geometric units) based on supplied EoS
    max_central_density = pyr.find_rhoc_tov_max_mass(eos, rhobr0 = cgs_density_to_geometric(min_central_dens),  ### arbitrary minimum density
                               rhobr1 = cgs_density_to_geometric(max_central_dens), ### arbitrary max density for large enough finding range
                               nbits = 28, acc = 1e-8, max_steps = 30) ### numerical defaults given by RePrimAnd
    
    ### Create a density range from arbitary minimum to first* maximum mass central density
    central_densities = np.linspace(cgs_density_to_geometric(min_central_dens), max_central_density, 100)
    
    ### TOV Solving
    tidal = np.array([pyr.get_tov_properties(eos, rhoc).deformability.lambda_tidal for rhoc in central_densities])
    masses = np.array([pyr.get_tov_properties(eos, rhoc).grav_mass for rhoc in central_densities])
    ### 1.477 is for converting radii from geometric units to km
    radii = [((pyr.get_tov_properties(eos, rhoc).circ_radius)*1.477) for rhoc in central_densities] 
    
    ### Now make a structured array with tidal and masses
    tidal_masses = np.zeros((1,len(central_densities)),
                             dtype={'names':("Lambda", "M", "R"), ### name of the groups in the structured array
                          'formats':("f8", "f8", "f8")}) ### assumed all floats
    
    tidal_masses["Lambda"] = tidal
    tidal_masses["M"] = masses
    tidal_masses["R"] = radii
    
    return tidal_masses
    
def get_seq(eos, acc = pyr.star_acc_simple()):
    tov_branch = pyr.make_tov_branch_stable(eos, acc)
    return tov_branch

if __name__ == "__main__":
    
    ### EoS samples ###
    # eos_samples = h5py.File("/home/sunny.ng/TOVSol/MMGP_full_covariance.h5", "r+")
    eos_samples = h5py.File("/home/sunny.ng/TOVSol/lyla_test_again.h5", "r+")
    
    ### Create range for iterating through
    num_eos = len(eos_samples["eos"])
    eos_to_be_used = np.arange(num_eos)
    ### Enumerating Pressure-Density Relations for iteration
    micro_data = {eos_num : np.array(eos_samples['eos'][eos_id]) for eos_num, eos_id in enumerate(eos_samples['eos'])}
    
    try:
        ns = eos_samples.create_group("ns")
    except:
        raise ("ns group already exists.")
    
    for eqn in range(len(eos_to_be_used)):
        try:
            macro_draw = TOVSolve(pressure = cgs_density_to_geometric(micro_data[eqn]["pressurec2"]),
                     rho = cgs_density_to_geometric(micro_data[eqn]["baryon_density"]),
                     eps = cgs_density_to_geometric(micro_data[eqn]["energy_densityc2"]),
                     eps_0 = cgs_density_to_geometric(micro_data[eqn]["energy_densityc2"][0]))
            
            ns[f"eos_{eqn:06d}"] = macro_draw[0]
            print(f"Macro draw: eos_{eqn:06d} - generated.")
        except:
            print(f"TOV Solving for EoS {eqn} did not work. Attempting correction...")
            corr_eos = monotonicity_check(micro_data[eqn]["pressurec2"],
                                          micro_data[eqn]["energy_densityc2"],
                                          micro_data[eqn]["baryon_density"])
            ### Need to re-interpolate over densities to have equally spaced increments for sound speed calculation
            rng = np.geomspace(corr_eos["energy_densityc2"][0],corr_eos["energy_densityc2"][-1], len(corr_eos["energy_densityc2"])) 
            corr_press = np.interp(rng, corr_eos["energy_densityc2"], corr_eos["pressurec2"])
            corr_eos["pressurec2"] = corr_press
            corr_eos["energy_densityc2"] = rng
            macro_draw = TOVSolve(pressure = cgs_density_to_geometric(corr_eos["pressurec2"]),
                     rho = cgs_density_to_geometric(corr_eos["baryon_density"]),
                     eps = cgs_density_to_geometric(corr_eos["energy_densityc2"]),
                     eps_0 = cgs_density_to_geometric(corr_eos["energy_densityc2"][0]))

            ns[f"eos_{eqn:06d}"] = macro_draw[0]
            print(f"Macro draw: eos_{eqn:06d} - generated.")
            continue
    
    eos_samples.close()
    
    
