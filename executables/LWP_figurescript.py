### Standard Libraries ------------------------------------------------
import numpy as np
import pandas as pd
import os
import bilby.gw.conversion as conversion
import matplotlib.pyplot as plt
import seaborn as sns
import h5py
import bilby
### -------------------------------------------------------------------

### Constants
gcm3_to_dynecm2=8.9875e20 
MeV_to_gcm3 = 1.7827e12 
dynecm2_to_MeVFermi = 1.6022e33
gcm3_to_fm4 = 3.5178e14
sat_dens = 2.8*(np.power(10.0,14.))
twice_sat_dens = 2*sat_dens
six_sat_dens = 6*sat_dens
c = bilby.core.utils.speed_of_light


astro_prefix= "PECE-ET_low_spin_01.csv"
event = "nXG (CE-ET) Event 1"

### Loading EoS prior samples
samples_file =  "full_covariance_example.h5"
samples  = h5py.File(samples_file)
### Generating a ranged array of the EoS's and their respective indices
# eos_to_be_used = np.arange(10000)
eos_to_be_used = np.arange(5206)
### Enumerating Mass-Radius EoS Relation values
macro_data = {eos_num : np.array(samples['ns'][eos_id]) for eos_num, eos_id in enumerate(samples['eos'])}
### Enumerating Pressure-Density Relations for likelihood weighing
macro_data2 = {eos_num : np.array(samples['eos'][eos_id]) for eos_num, eos_id in enumerate(samples['eos'])}

### Calling on weights generated from LWP
lwp_run = True
verbose = True
gw_event = "nXG"
type_event = "01_MM+GP"

if lwp_run == True: ### To load weights from a condor-submitted LWP job
    weights = pd.read_csv(f"{gw_event}_{type_event}_lwp/result/{gw_event}_{type_event}_lwp_eos.csv")
elif lwp_run != True: ### To load weights from LWP notebook analysis
    weights = pd.read_csv(f"{astro_prefix}_eos.csv") 
    if verbose == True:
        print(f"Loading weights from: {astro_prefix}")
max_logweight = max(weights['logmargweight'])


### Only need draw EoS's from posterior distribution once:
# Resampling EoS's with weighted values according to lwp likelihoods
ex_weights = np.exp(weights["logmargweight"])
# Normalize values to have probabilities add up to 1
ex_weights = ex_weights/(sum(ex_weights))
# Resample EoS's according to weights --> gives posterior distribution of EoS's
weight_eos = np.random.choice(eos_to_be_used, size=20000, replace=True, p=ex_weights)


### Building the posterior contour by obtaining upper and lower sigma values for pressure at each density interval:
plot_densities = np.geomspace(np.power(10.0,13.0), np.power(10.0, 16.0), 10000)
percentiles = [5,95] # Currently set to generate a 90% C.L Contour

### ONLY FOR MM+GP EOS SET (PRELIMINARY) ####################################
fixed_weights = weights[weights.isna().any(axis=1)]
weights.iloc[3750]["logmargweight"] = -np.inf
#############################################################################

### Interpolation along sigma values to generate posterior contour
# --------------------------------------------------------------------
pevals = []
w_pevals = []

### Building contour for the EoS prior set
for eos in range(len(eos_to_be_used)):
    eos_densities = macro_data2[eos]["energy_densityc2"]
    eos_pressures = macro_data2[eos]["pressurec2"]*gcm3_to_dynecm2
    pevals.append(np.interp(plot_densities, eos_densities, eos_pressures))
pevals = np.array(pevals)

### Building contour for the gravitationally informed, EoS Set
for eos in weight_eos:
    eos_densities = macro_data2[eos]["energy_densityc2"]
    eos_pressures = macro_data2[eos]['pressurec2']*gcm3_to_dynecm2
    w_pevals.append(np.interp(plot_densities, eos_densities, eos_pressures))
w_pevals = np.array(w_pevals)

### Obtaining quantiles 
p_sigmas = np.zeros((len(plot_densities),len(percentiles)))
p_w_sigmas = np.zeros((len(plot_densities),len(percentiles)))

for i in range(len(plot_densities)):
    p_sigmas[i]=np.percentile(np.array(pevals[:,i]),percentiles)
    p_w_sigmas[i]=np.percentile(np.array(w_pevals[:,i]),percentiles)

# ---------------------------------------------------------------------

### Actual Plotting Portion
plt.loglog()

plt.plot(plot_densities,p_sigmas[:,0], color = "k", alpha = .8, label = "Prior") 
plt.plot(plot_densities,p_sigmas[:,1], color = "k", alpha = .8)
plt.plot(plot_densities,p_w_sigmas[:,0], color = "#558BD4", alpha = .9, label = f"{event} Informed") 
plt.plot(plot_densities,p_w_sigmas[:,1], color = "#558BD4", alpha = .9)
plt.fill_between(plot_densities,p_sigmas[:,0], p_sigmas[:,1], alpha = 0.15, color = "k")
plt.fill_between(plot_densities,p_w_sigmas[:,0], p_w_sigmas[:,1], alpha = 0.15, color = "#558BD4")

plt.xlabel(r"$\varepsilon \ [g/cm^3]$", math_fontfamily = "cm", fontsize = 15.)
plt.ylabel(r"$P \ [dyne/cm^2]$", math_fontfamily="cm", fontsize = 15.)

xmin = np.power(10.0,13.4)
xmax = np.power(10.0,15.5)
ymin = np.power(10.0,32.)
ymax = np.power(10.0,37.0)
plt.xlim(xmin,xmax)
plt.ylim(ymin,ymax)

### Annotating intervals of once, twice, and six times nuclear saturation density
plt.axvline(x = sat_dens, color = "k", linewidth = .8)
plt.axvline(x = twice_sat_dens, color = "k", linewidth = .8)
plt.axvline(x = six_sat_dens, color = "k", linewidth = .8)

plt.annotate(r"$\rho_{nuc}$",xy = ((sat_dens), (np.power(10.,36.5))),
             rotation = 90.0, math_fontfamily = "cm", fontsize = 13.)
plt.annotate(r"$2\rho_{nuc}$",xy = ((2*sat_dens)*(np.power(10.0,.012)), (np.power(10.,36.5))),
             rotation = 90.0, math_fontfamily = "cm", fontsize = 12.)
plt.annotate(r"$6\rho_{nuc}$",xy = ((6*sat_dens)*(np.power(10.0,.012)),(np.power(10.,36.5))),
             rotation = 90.0, math_fontfamily = "cm", fontsize = 12.)

plt.grid(alpha = 0.2)
plt.legend(fontsize = 8.)
plt.gcf().set_dpi(500)

plt.show()