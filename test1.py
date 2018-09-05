
import numpy as np

total_continuous_dim = 2
order = 0

continuous_sample_points = np.linspace(-1.0, 1.0, 20)
#a specific noise factor will be varied with 10 steps.

num_points, steps = 10, len(continuous_sample_points)
# each step has points with randomly-sampled other noise factor


continuous_noise = []
for _ in range(num_points):
    cur_sample = np.random.normal(size=[1, total_continuous_dim])
    continuous_noise.extend([cur_sample]*steps)
continuous_noise = np.concatenate(continuous_noise)

varying_factor = np.tile(continuous_sample_points, num_points)
continuous_noise[:, order] = varying_factor 
continuous_noise = np.float32(continuous_noise)

print(continuous_noise)