import defiNETti
import simulator_example

# Train the network --  Use small num_batches for example purposes.
prior_rates, weight_prob = simulator_example.parse_hapmap_empirical_prior(['genetic_map_GRCh37_chr1_truncated.txt'])
defiNETti.train((198,24,2), (2,), lambda: simulator_example.simulate_data(prior_rates, weight_prob), 
                phi_net = [('conv', 5, 32),('conv', 5, 64)], 
                h_net = [("fc",128),("fc",128),("softmax",)],
                num_batches = 2000,
                training_threads=5,
                sim_threads=5, 
                save_path='./example_weights')


# Test the network
print("Testing the network...")
data = []
for i in range(500):
    data_i, _ = simulator_example.simulate_data(prior_rates, weight_prob) # Typically you should feed in your real data here instead of simulated.
    data.append(data_i)
output = defiNETti.test(data, model_path='./example_weights', threads = 5)