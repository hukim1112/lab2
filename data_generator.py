import numpy as np

def noisy_single_sinewave():
	x = np.float32(np.random.uniform(-1, 1, [1, 1000])[0])
	y = np.float32(np.sin(x*np.pi)+np.random.normal(0, 0.1, 1000))
	real_data = np.array([[i, j] for i, j in zip(x, y)])

	return real_data

def single_sinewave():
	x = np.float32(np.random.uniform(-1, 1, [1, 1000])[0])
	y = np.float32(np.sin(x*np.pi))
	real_data = np.array([[i, j] for i, j in zip(x, y)])

	return real_data