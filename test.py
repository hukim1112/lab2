
import tensorflow as tf
ds = tf.contrib.distributions
from tensorflow.python.ops.losses import losses
import numpy as np
#test code

def get_noise(batch_size, total_continuous_noise_dims):
  """Get unstructured and structured noise for InfoGAN.

  Args:
    batch_size: The number of noise vectors to generate.
    categorical_dim: The number of categories in the categorical noise.
    structured_continuous_dim: The number of dimensions of the uniform
      continuous noise.
    total_continuous_noise_dims: The number of continuous noise dimensions. This
      number includes the structured and unstructured noise.

  Returns:
    A 2-tuple of structured and unstructured noise. First element is the
    unstructured noise, and the second is a 2-tuple of
    (categorical structured noise, continuous structured noise).
  """
  # Get unstructurd noise.
  noise = (tf.random_uniform(
      [batch_size, total_continuous_noise_dims]) - 0.5 )*4
  # noise = tf.random_normal([batch_size, total_continuous_noise_dims])

  return noise

#######################parameters
total_continuous_dim = 2
order = 0




continuous_sample_points = np.linspace(-2.0, 2.0, 20)
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

print(continuous_noise.max(), continuous_noise.min())

with tf.Session() as sess:
  noise = get_noise(32, total_continuous_dim)
  x = sess.run(noise)
  print(x.max(), x.min())