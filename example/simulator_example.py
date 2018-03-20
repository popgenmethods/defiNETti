from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import msprime

# Data shape
HEIGHT = 198  # number of haplotypes, CEU = 198, YRI = 216
WIDTH = 24   # number of seg sites, should also be divisible by 4
NUM_CLASSES = 2

# Simulation parameters
NE = 1.0e4
MU = 1.1e-8
R  = 1.25e-8
theta = 0.00044
rho = .0005
hap_len = 52000 #heuristic to get enough SNPs
hot_spot_len = 2000
lower_intensity = 10
upper_intensity = 100
GEN_IN_YRS = 29


def parse_hapmap_empirical_prior(files):
    print("Parsing empirical prior...")
    assert(len(files) == 1)
    print(files)
    mat = np.loadtxt(files[0], skiprows = 1, usecols=(1,2))
    print(mat.shape)
    mat[:,1] = mat[:,1]*(1.e-8)
    mat = mat[mat[:,1] != 0.0, :] # remove 0s
    weights = mat[1:,0] - mat[:-1,0]
    prior_rates = mat[:-1,1]
    prob = weights / np.sum(weights)
    print("Done parsing prior")

    return prior_rates, prob


def draw_background_rate_from_prior(prior_rates, prob):
    return np.random.choice(prior_rates, p=prob)

def simulate_data(prior_rates, weight_prob):
    hot_spot = np.random.randint(0,2)

    # Use flat empirical maps
    background = draw_background_rate_from_prior(prior_rates, weight_prob)
    if hot_spot >= 1:
        heat = np.random.uniform(lower_intensity,upper_intensity)*max(R/background,1.)
        reco_map = msprime.RecombinationMap([0,(hap_len - hot_spot_len)/2,
                                         (hap_len + hot_spot_len)/2, hap_len],[background,background*heat,background,0])
    else:
        reco_map = msprime.RecombinationMap([0,hap_len],[background,0])


    # Simulate
    tree_sequence = msprime.simulate(sample_size=HEIGHT, Ne = NE, recombination_map=reco_map,
                                    mutation_rate=MU)
    seg_sites = tree_sequence.get_num_mutations()

    # Make sure that we simulated enough seg sites for this example. Should not resimulate very often
    if seg_sites < WIDTH:
        print("Resimulating0...",file=sys.stderr)
        print(seg_sites)
        return simulate_data()


    # center around the hotspot
    mut_pos = [mut[0] for mut in tree_sequence.mutations()] # chromosome positions
    assert(len(mut_pos) == seg_sites)
    before_hs = [x < 0.5*hap_len for x in mut_pos].count(True)
    after_hs  = [x >= 0.5*hap_len for x in mut_pos].count(True)
    hots = [x >= (hap_len-hot_spot_len)*0.5 and x < (hap_len+hot_spot_len)*0.5 for x in mut_pos].count(True)
    assert(before_hs + after_hs == seg_sites)


    # get distances and image
    if hots >= WIDTH:
        print("Hot Length: ", hots)
        print("Resimulating1.5...",file=sys.stderr)
        return simulate_data()
    # Check if there are enough SNPs left and right of the hotspot
    if before_hs < WIDTH//2:
        print("Resimulating2...",file=sys.stderr)
        print(before_hs)
        return simulate_data()
    if after_hs <= WIDTH//2:
        print("Resimulating3...",file=sys.stderr)
        print(after_hs)
        return simulate_data()
    distances = np.array([mut_pos[i+1]-mut_pos[i] for i in range(before_hs-WIDTH//2,before_hs+WIDTH//2)],dtype=float)[:-1] * 4*NE*MU
    distances = np.vstack([np.copy(distances) for k in range(HEIGHT)])
    image = np.array([np.copy(variant.genotypes) if variant.genotypes.sum() < HEIGHT/2 else np.copy(1-variant.genotypes) for variant in tree_sequence.variants()]).transpose()[:,before_hs-WIDTH//2:before_hs+WIDTH//2].reshape([HEIGHT,WIDTH])



    assert(image.shape[1] == distances.shape[1] + 1)
    assert(image.shape[0] == distances.shape[0])
    combined = np.empty((image.shape[0], image.shape[1], 2), dtype=distances.dtype)
    combined[:,:,0] = image
    combined[:,:-1,1] = distances
    combined[:,-1,1] = 0.0
    
    label = np.zeros(2)
    label[hot_spot] = 1
    return (combined,label)

# prior_rates, weight_prob = parse_hapmap_empirical_prior(['genetic_map_GRCh37_chr1_truncated.txt'])
