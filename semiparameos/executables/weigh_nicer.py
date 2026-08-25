"""
Demonstrating how to weigh mass-radius samples based on EoSs in standard h5 format. 
"""
import temperance as tmpy 
import temperance.core.result as result
from temperance.core.result import EoSPosterior
from temperance.sampling.eos_prior import EoSPriorH5
import temperance.weighing.weigh_by_density_estimate as weigh_by_density_estimate

import pandas as pd
import numpy as np
import h5py
import tqdm
import os.path

class FlatMassPrior:
    def __init__(self, m_min, m_max, seed=None):
        self.m_min = m_min
        self.m_max = m_max
        self.rng = np.random.default_rng(seed)

    def sample(self, size):
        return self.rng.uniform(self.m_min, self.m_max, size)
    
def marginalize_over_mr_samples(mr_samples, num_marginalization_samples=50, nicer_tag="j0740"):
    marginalizable_samples = mr_samples.copy()[["eos", "Mmax", f"logweight_{nicer_tag}"]]
    marginalizable_samples.loc[f"weight_{nicer_tag}"] = np.exp(marginalizable_samples[f"logweight_{nicer_tag}"])
    marginalized_weights = marginalizable_samples.groupby("eos").mean()
    marginalized_weights.reset_index(inplace=True)
    return marginalized_weights

def collate_weights(likelihood_directory,
                    number_of_lists,
                    nicer_tag, label,
                    outpath = ".",
                    save_to_csv = True):
    
    # Initial dataframe with first set of 10 EoS's
    recur_nicer_likelihoods = pd.read_csv(f"{likelihood_directory}/{label}-0_{nicer_tag}_eos.csv")
    for idx in range(number_of_lists):
        likelihoods_to_add = pd.read_csv(f"{likelihood_directory}/{label}-{idx}_{nicer_tag}_eos.csv")
        if idx == 0:
            continue
        else:
            recur_nicer_likelihoods = pd.concat([recur_nicer_likelihoods, likelihoods_to_add],
                                                  ignore_index = True)
    final_nicer_likelihoods = recur_nicer_likelihoods.copy()
    if save_to_csv:
        final_nicer_likelihoods.to_csv(f"{outpath}/{label}-collated_eos_post.csv", index = False)
    return final_nicer_likelihoods

if __name__ == "__main__":
    
    # out directory for overall executable
    main_outdir = "/home/sunny.ng/test_exec/test_output/"
    outdir = main_outdir
    
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    
    nicer_tag = "j0437"
    posterior_tag = "semiparametric"
    
    # EoS h5 path + astro_weighted dataframe holding Mmax for this EoS set
    eos_h5_path = "/home/sunny.ng/semiparameos/generated_eoss/MMGP_set_prior.h5"
    eos_set_astro_post = pd.read_csv("/home/sunny.ng/semiparameos/astro_likelihoods/MMGP_set_prior_posterior.csv")
    eos_mmax = eos_set_astro_post["Mmax"]
    num_eos_samples = 75000
    
    # replace with actual nicer samples, downsample
    # fake_nicer_samples = pd.DataFrame({"M":1.4 + .2 * np.random.rand(1000), "R":12.5 + 1.5 * np.random.rand(1000)})
    
    # replace Mmax with the correct array of Mmaxs
    eos_posterior = EoSPosterior(samples = pd.DataFrame({"eos":np.arange(num_eos_samples),
                                                              "Mmax":np.array(eos_mmax[:num_eos_samples])}), # replaced
                                                              label=posterior_tag)
    def nicer_samples_for_eos_posterior_subset(eos_posterior):
        # Read in EoS h5 file for prior object
        eos_prior_h5 = EoSPriorH5(h5_path=eos_h5_path, eos_subgroup_path="eos",
                                       macro_subgroup_path="ns", index_to_key=lambda x : f"eos_{int(x):06}")
        # Get M-R samples from h5 prior
        mr_samples = weigh_by_density_estimate.generate_mr_samples(eos_posterior, eos_prior_h5,
                                                                   FlatMassPrior, num_samples_per_eos=300,
                                                                   mass_prior_kwargs={"m_min":1.2, "m_max":1.6})

        ######################################################################
        ### imported from original NICER analysis script before h5 adaptation
        ######################################################################

        sample_weights = weigh_by_density_estimate.weigh_mr_samples(mr_samples[["M", "R"]], nicer_data, 
                                                                     prior_column=result.WeightColumn("log(prior)",
                                                                     is_log=True))
        mr_samples[f"logweight_{nicer_tag}"] = sample_weights
        mr_samples.to_csv(f"{outdir}/{eos_posterior.label}_{nicer_tag}_post.csv", index=False)
        eos_post = marginalize_over_mr_samples(mr_samples, nicer_tag=nicer_tag)

        eos_post.to_csv(f"{outdir}/{eos_posterior.label}_{nicer_tag}_eos.csv",
                            index=False)
    ######################################################################
    # nicer_samples_for_eos_posterior_subset(eos_posterior)
    
    ### We can do this if we want weights to be seperated into a whole bunch of split up csv's
    eos_lists = np.array_split(np.arange(len(eos_posterior.samples["eos"])), 1000)
    for i, eos_list in enumerate(tqdm.tqdm(eos_lists)):
        eos_post_local = EoSPosterior(eos_posterior.samples.iloc[eos_list],
                                              label=f"{posterior_tag}-{i}")
        nicer_samples_for_eos_posterior_subset(eos_post_local)
    print("Collating samples into single .csv...")
    collate_weights(f"{outdir}", len(eos_lists),
                    nicer_tag = nicer_tag,
                    label = posterior_tag,
                    outpath = "/home/sunny.ng/semiparameos/astro_likelihoods/nicer_likelihoods")
