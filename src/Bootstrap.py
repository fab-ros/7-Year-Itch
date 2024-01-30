import numpy as np
from tqdm import tqdm


''' Method to perform a bootstrap of num_sim simulations on the data '''
def bootstrap(num_sim, batch_size, data, data_probs):
    buckets = len(data)

    means = np.zeros(num_sim)
    most_probables = np.zeros(num_sim)
    medians = np.zeros(num_sim)
    estimates = np.zeros(shape=(num_sim, buckets))

    np.random.seed(2024)

    for simulation in tqdm(np.arange(num_sim)):
        # simulate N lifetimes
        samples = np.random.choice(np.array(data), size=batch_size, p=data_probs)

        p_estimated = [np.sum(samples == i) / batch_size for i in np.array(data)]
        mean_estimated = np.sum(np.array(data) * p_estimated)
        median_estimated = data[np.argmax(np.cumsum(p_estimated) >= 0.5)]
        most_likely_estimated = data[np.argmax(p_estimated)]

        # store the estimates
        means[simulation] = mean_estimated
        most_probables[simulation] = most_likely_estimated
        medians[simulation] = median_estimated
        estimates[simulation, :] = p_estimated

    return means, medians, most_probables, estimates
