
import tensorflow as tf
ds = tf.contrib.distributions
from tensorflow.python.ops.losses import losses
#test code

def get_noise(batch_size, total_continuous_noise_dims, a):
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
  noise = 4*a
  # noise = tf.random_normal([batch_size, total_continuous_noise_dims])

  return noise


a = 0.1


print(get_noise(10, 20, a))