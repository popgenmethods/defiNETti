import defiNETti
import simulator_example

prior_rates, weight_prob = simulator_example.parse_hapmap_empirical_prior(['genetic_map_GRCh37_chr1_truncated.txt'])
defiNETti.train((198,24,2), (2,), lambda: simulator_example.simulate_data(prior_rates, weight_prob), 
                phi_net = [('conv', 5, 32),('conv', 5, 64)], 
                h_net = [("fc",128),("fc",128),("softmax",)],
                training_threads=5,
                sim_threads=5, 
                save_path='./example_weights')