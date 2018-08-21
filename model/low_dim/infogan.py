import tensorflow as tf
slim = tf.contrib.slim
tfgan = tf.contrib.gan
layers = tf.contrib.layers
ds = tf.contrib.distributions
from tensorflow.python.ops import variable_scope
from tensorflow.python.framework import ops
import numpy as np
import visualizations, losses_fn
import os
import cv2
from datasets.reader import mnist as mnist_reader
from matplotlib import pyplot as plt
import matplotlib.cm as cm
from tensorflow.python.ops.losses import losses
leaky_relu = lambda net: tf.nn.leaky_relu(net, alpha=0.01)

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
  noise = tf.random_normal(
      [batch_size, total_continuous_noise_dims])

  return noise


class Infogan():
    def __init__(self, data):

        self.graph = tf.Graph()
        self.sess = tf.Session()
        gpu_options = tf.GPUOptions(allow_growth=True)
        self.sess = tf.Session(graph = self.graph, config=tf.ConfigProto(gpu_options=gpu_options))

        self.generator = generator
        self.discriminator = discriminator
        self.data = data

        # data
        self.cat_dim = self.data.cat_dim
        self.code_con_dim = self.data.code_con_dim
        self.total_con_dim = self.data.total_con_dim
        self.channel = self.data.channel
        self.dataset_path = self.data.path
        self.dataset_name = self.data.name
        self.split_name = self.data.split_name
        self.batch_size = self.data.batch_size
        self.real_data = self.data.real_data
        print(self.batch_size)
        with self.graph.as_default():
                # x = np.float32(np.random.uniform(-1, 1, [1, 1000])[0])
                # y = np.float32(np.sin(x*np.pi) + np.random.normal(0, 0.2, [1000]))
                # data = [[i, j] for i, j in zip(x, y)]
                self.real_data = self.data.data             
                self.gen_input_noise = get_noise(self.batch_size, self.total_con_dim)

                with variable_scope.variable_scope('generator') as self.gen_scope:
                    self.gen_data = self.generator(self.gen_input_noise) #real/fake loss
                with variable_scope.variable_scope('discriminator') as self.dis_scope:
                    self.dis_gen_data, self.Q_net = self.discriminator(self.gen_data) #real/fake loss + I(c' ; X_{data}) loss

                with variable_scope.variable_scope(self.dis_scope.name, reuse = True):
                    self.real_data = ops.convert_to_tensor(self.real_data)
                    self.dis_real_data , _= self.discriminator(self.real_data) #real/fake loss
            
                #loss
                self.dis_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.dis_scope.name)
                self.gen_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.gen_scope.name)

                self.D_loss = losses_fn.wasserstein_discriminator_loss(self.dis_real_data, self.dis_gen_data)
                self.G_loss = losses_fn.wasserstein_generator_loss(self.dis_gen_data)
                self.wasserstein_gradient_penalty_loss = losses_fn.wasserstein_gradient_penalty(self, self.real_data, self.gen_data)
                self.mutual_information_loss = mutual_information_penalty(self.gen_input_noise[0], self.Q_net)
                tf.summary.scalar('D_loss', self.D_loss + self.wasserstein_gradient_penalty_loss)
                tf.summary.scalar('G_loss', self.G_loss)
                self.merged = tf.summary.merge_all()

                self.global_step = tf.Variable(0, name='global_step', trainable=False)
                #solver
                self.D_solver = tf.train.AdamOptimizer(0.001, beta1=0.5).minimize(self.D_loss+self.wasserstein_gradient_penalty_loss, var_list=self.dis_var, global_step=self.global_step)
                self.G_solver = tf.train.AdamOptimizer(0.0001, beta1=0.5).minimize(self.G_loss, var_list=self.gen_var)
                self.mutual_information_solver = tf.train.AdamOptimizer(0.0001, beta1=0.5).minimize(self.mutual_information_loss, var_list=self.gen_var + self.dis_var)
                self.saver = tf.train.Saver()
                self.initializer = tf.global_variables_initializer()
    def train(self, result_dir, ckpt_dir, log_dir, training_iteration = 1000000, G_update_num=1, D_update_num=1, Q_update_num =1):
        with self.graph.as_default():
            path_to_latest_ckpt = tf.train.latest_checkpoint(checkpoint_dir=ckpt_dir)
            if path_to_latest_ckpt == None:
            	print('scratch from random distribution')
            	self.sess.run(self.initializer)
            else:
                self.saver.restore(self.sess, path_to_latest_ckpt)
                print('restored')
            self.train_writer = tf.summary.FileWriter(log_dir, self.sess.graph)
            for i in range(training_iteration):
                for _ in range(D_update_num):
                    self.sess.run(self.D_solver)
                for _ in range(G_update_num):
                    self.sess.run(self.G_solver)
                for _ in range(Q_update_num):
                    self.sess.run(self.mutual_information_solver)
                merge, global_step = self.sess.run([self.merged, self.global_step])
                self.train_writer.add_summary(merge, global_step)
                if ((i % 500) == 0):
                    fig = plt.figure()
                    ax1 = fig.add_subplot(111)
                    gen_data_test = self.sess.run(self.gen_data)
                    print(self.gen_data.shape)
                    ax1.scatter(gen_data_test[:, 0], gen_data_test[:, 1], s = 10, c ='b', marker="s", label='first')
                    fig.savefig(os.path.join(result_dir, str(i)+'.png'), dpi=fig.dpi)
                    plt.close(fig)
            varying_noise_continuous_ndim_without_category(self, 0, self.total_con_dim, result_dir)

                # if ((i % 500) == 0 ):
                # 	self.saver.save(self.sess, os.path.join(ckpt_dir, 'model'), global_step=self.global_step)


def varying_noise_continuous_ndim_without_category(self, order, total_continuous_dim, result_path):
    """Create noise showing impact of categorical noise in InfoGAN.

    Categorical noise is constant across columns. Other noise is constant across
    rows.

    Args:
    self : model class itself.
    order : integer. it points out the order of varying continuous code's factor from -1 to 1
    categorical_dim : The number of object to appear in dataset.
    code_continuous_dim : The number of factors to be disentangled in input representation for generating
    total_continuous_dim : The number of continuous factors in input representation for generating
    iteration : global step number
    result_path : path to save the result
    """
    row_num = 5
    continuous_sample_points = np.linspace(-1.0, 1.0, 10)

    rows, cols = row_num, len(continuous_sample_points)

    continuous_noise = []
    for _ in range(rows):
        cur_sample = np.random.normal(size=[1, total_continuous_dim])
        continuous_noise.extend([cur_sample]*cols)
    continuous_noise = np.concatenate(continuous_noise)

    varying_factor = np.tile(continuous_sample_points, rows)
    continuous_noise[:, order] = varying_factor 
    continuous_noise = np.float32(continuous_noise)
    
    display_images = []
    with variable_scope.variable_scope(self.gen_scope.name, reuse = True):
        data = self.generator(continuous_noise)

    colors = cm.rainbow(np.linspace(0, 1, len(continuous_sample_points)))
    gen_data_test = self.sess.run(self.gen_data)
    for i in range(row_num):
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        for j in range(len(continuous_sample_points)):
            ax1.scatter(gen_data_test[len(continuous_sample_points)*i+j, 0], gen_data_test[len(continuous_sample_points)*i+j, 1], s = 10, c =colors[j], marker="s")
        fig.savefig(os.path.join(result_path, str(i)+'_variation.png'), dpi=fig.dpi)
        plt.close(fig)

def mutual_information_penalty(
    structured_generator_inputs,
    predicted_distributions,
    weights=1.0,
    scope=None,
    add_summaries=False):
  """Returns a penalty on the mutual information in an InfoGAN model.

  This loss comes from an InfoGAN paper https://arxiv.org/abs/1606.03657.

  Args:
    structured_generator_inputs: A list of Tensors representing the random noise
      that must  have high mutual information with the generator output. List
      length should match `predicted_distributions`.
    predicted_distributions: A list of tf.Distributions. Predicted by the
      recognizer, and used to evaluate the likelihood of the structured noise.
      List length should match `structured_generator_inputs`.
    weights: Optional `Tensor` whose rank is either 0, or the same dimensions as
      `structured_generator_inputs`.
    scope: The scope for the operations performed in computing the loss.
    loss_collection: collection to which this loss will be added.
    reduction: A `tf.losses.Reduction` to apply to loss.
    add_summaries: Whether or not to add summaries for the loss.

  Returns:
    A scalar Tensor representing the mutual information loss.
  """
  #print('cat shape', log_prob_cat.shape)  
  q_cont = predicted_distributions
  sigma_cont = tf.ones_like(q_cont)
  q_cont = ds.Normal(loc=q_cont, scale=sigma_cont)
  log_prob_con = tf.reduce_mean(q_cont.log_prob(structured_generator_inputs), axis = 0)

  loss = -1 * losses.compute_weighted_loss(log_prob_con, weights, scope)

  return loss

def generator(gen_input_noise, weight_decay=2.5e-5):
    """InfoGAN discriminator network on MNIST digits.
    
    Based on a paper https://arxiv.org/abs/1606.03657 and their code
    https://github.com/openai/InfoGAN.
    
    Returns:
        A generated image in the range [-1, 1].
    """

    with slim.arg_scope(
        [layers.fully_connected, layers.conv2d_transpose],
        activation_fn=leaky_relu, normalizer_fn=layers.batch_norm,
        weights_regularizer=layers.l2_regularizer(weight_decay)):
        net = layers.fully_connected(gen_input_noise, 64)
        net = layers.fully_connected(net, 32)
        # Make sure that generator output is in the same range as `inputs`
        # ie [-1, 1].
        net = layers.fully_connected(net, 2, normalizer_fn=None, activation_fn=tf.tanh)
        return net

def discriminator(img, weight_decay=2.5e-5):
    """InfoGAN discriminator network on MNIST digits.
    
    Based on a paper https://arxiv.org/abs/1606.03657 and their code
    https://github.com/openai/InfoGAN.
    
    Args:
        img: Real or generated MNIST digits. Should be in the range [-1, 1].
        unused_conditioning: The TFGAN API can help with conditional GANs, which
            would require extra `condition` information to both the generator and the
            discriminator. Since this example is not conditional, we do not use this
            argument.
        weight_decay: The L2 weight decay.
        categorical_dim: Dimensions of the incompressible categorical noise.
        continuous_dim: Dimensions of the incompressible continuous noise.
    
    Returns:
        Logits for the probability that the image is real, and a list of posterior
        distributions for each of the noise vectors.
    """
    with slim.arg_scope(
        [layers.conv2d, layers.fully_connected],
        activation_fn=leaky_relu, normalizer_fn=None,
        weights_regularizer=layers.l2_regularizer(weight_decay),
        biases_regularizer=layers.l2_regularizer(weight_decay)):
        net = layers.fully_connected(img, 32, normalizer_fn=layers.batch_norm)
        net = layers.fully_connected(net, 64, normalizer_fn=layers.batch_norm)
        logits_real = layers.fully_connected(net, 1, activation_fn=None)
        encoder = layers.fully_connected(net, 64, normalizer_fn=layers.batch_norm)
        q_cont = layers.fully_connected(encoder, 1, activation_fn=None)
        return logits_real, q_cont
