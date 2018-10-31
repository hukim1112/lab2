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
leaky_relu = lambda net: tf.nn.leaky_relu(net, alpha=0.01)

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
  noise = (tf.random_uniform(
      [batch_size, total_continuous_noise_dims]) - 0.5 )*4*a
  # noise = tf.random_normal([batch_size, total_continuous_noise_dims])

  return noise


class Gan():
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

        self.anything = self.data.anything


        print(self.batch_size)
        with self.graph.as_default():         
                self.gen_input_noise = get_noise(self.batch_size, self.total_con_dim, self.data.anything)

                with variable_scope.variable_scope('generator') as self.gen_scope:
                    self.gen_data = self.generator(self.gen_input_noise) #real/fake loss
                with variable_scope.variable_scope('discriminator') as self.dis_scope:
                    self.dis_gen_data = self.discriminator(self.gen_data) #real/fake loss + I(c' ; X_{data}) loss
                with variable_scope.variable_scope(self.dis_scope.name, reuse = True):
                    self.real_data = ops.convert_to_tensor(self.real_data)
                    self.dis_real_data = self.discriminator(self.real_data) #real/fake loss

                
                #loss
                self.dis_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.dis_scope.name)
                self.gen_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.gen_scope.name)

                self.D_loss = losses_fn.wasserstein_discriminator_loss(self.dis_real_data, self.dis_gen_data)
                self.G_loss = losses_fn.wasserstein_generator_loss(self.dis_gen_data)
                self.wasserstein_gradient_penalty_loss = losses_fn.wasserstein_gradient_penalty(self, self.real_data, self.gen_data)
                tf.summary.scalar('D_loss', self.D_loss + self.wasserstein_gradient_penalty_loss)
                tf.summary.scalar('G_loss', self.G_loss)
                self.merged = tf.summary.merge_all()

                self.global_step = tf.Variable(0, name='global_step', trainable=False)
                #solver
                self.D_solver = tf.train.AdamOptimizer(0.001, beta1=0.5).minimize(self.D_loss+self.wasserstein_gradient_penalty_loss, var_list=self.dis_var, global_step=self.global_step)
                self.G_solver = tf.train.AdamOptimizer(0.0001, beta1=0.5).minimize(self.G_loss, var_list=self.gen_var)
    
                self.saver = tf.train.Saver()
                self.initializer = tf.global_variables_initializer()
    def train(self, result_dir, ckpt_dir, log_dir, training_iteration = 1000000, G_update_num=1, D_update_num=1):
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
                merge, global_step = self.sess.run([self.merged, self.global_step])
                self.train_writer.add_summary(merge, global_step)
                
                if ((i % 1000) == 0):

                    [gen_data_test, _input] = self.sess.run([self.gen_data, self.gen_input_noise])

                    fig = plt.figure(figsize=(8, 12), dpi = 80)
                    ax1 = fig.add_subplot(311)
                    ax1.scatter(gen_data_test[:, 0], gen_data_test[:, 1], s = 10, c ='b', marker="s", label='first')
                    for j, factor in enumerate(_input):
                        if j % 50 == 0:
                            ax1.annotate(str(round(factor[0], 2)), (gen_data_test[j, 0], gen_data_test[j, 1]), color=(0, 0, 0))

                    ax2 = fig.add_subplot(312)
                    ax2.scatter(gen_data_test[:, 0], gen_data_test[:, 1], s = 10, c ='b', marker="s", label='first')
                    varying_noise_continuous_ndim_without_category(self, ax2, global_step, 0, self.total_con_dim, result_dir)



                    ax3 = fig.add_subplot(313)
                    varying_noise_continuous_ndim_without_category(self, ax3, global_step, 0, self.total_con_dim, result_dir)

                    fig.savefig(os.path.join(result_dir, str(i)+'.png'), dpi=fig.dpi)
                    plt.close(fig)
                    self.saver.save(self.sess, os.path.join(ckpt_dir, 'model'), global_step=self.global_step)
                if ((i % 5000) ==0):
                    print(i)

    # def test(self, result_dir):
    #     fig = plt.figure()
    #     ax1 = fig.add_subplot(111)

    #     new_data = self.sess.run(self.gen_data)
    #     ax1.scatter(new_data[:, 0], new_data[:, 1], s = 10, c ='b', marker="s", label='first')
    #     fig.savefig(os.path.join(result_dir, 'test.png'), dpi=fig.dpi)

    # def test_tmp(self, ckpt_dir, result_dir):
    #     with self.graph.as_default():
    #         path_to_latest_ckpt = tf.train.latest_checkpoint(checkpoint_dir=ckpt_dir)
    #         if path_to_latest_ckpt == None:
    #             print('There is no trained weight files...')
    #             return
    #         else:
    #             self.saver.restore(self.sess, path_to_latest_ckpt)
    #             print('restored')
    #             images = self.sess.run(self.gen_data)
    #             print('shape check of result : ', images.shape)

    #         fig = plt.figure()
    #         ax1 = fig.add_subplot(111)
    #         with variable_scope.variable_scope(self.dis_scope.name, reuse = True):
    #             test_tmp_data = self.generator(tf.random_uniform([1000, 1], 0., 1)) #real/fake loss
    #             new_data = self.sess.run(test_tmp_data)
    #         ax1.scatter(new_data[:, 0], new_data[:, 1], s = 10, c ='b', marker="s", label='first')
    #         fig.savefig(os.path.join(result_dir, 'test1.png'), dpi=fig.dpi)

def varying_noise_continuous_ndim_without_category(self, figure, iteration, order, total_continuous_dim, result_path):
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

    continuous_sample_points = np.linspace(-2.0, 2.0, 20)
    #a specific noise factor will be varied with 10 steps.

    num_points, steps = 1, len(continuous_sample_points)
    # each step has points with randomly-sampled other noise factor


    continuous_noise = []
    for _ in range(num_points):
        cur_sample = np.random.normal(size=[1, total_continuous_dim])
        continuous_noise.extend([cur_sample]*steps)
    continuous_noise = np.concatenate(continuous_noise)

    varying_factor = np.tile(continuous_sample_points, num_points)
    continuous_noise[:, order] = varying_factor 
    continuous_noise = np.float32(continuous_noise)
    
    display_images = []
    with variable_scope.variable_scope(self.gen_scope.name, reuse = True):
        varying_data = self.generator(continuous_noise)

    #colors = cm.rainbow(np.linspace(0, 1, len(continuous_sample_points)))
    colors = [ ( 1/(i%steps + 1), 0, (i%steps + 1)/steps, 1) for i in range( continuous_noise.shape[0] )] #red to green

    scales = [ (1.1**(i%steps + 1))*10  for i in range( continuous_noise.shape[0] )]

    gen_data_test = self.sess.run(varying_data)
    ax1 = figure
    ax1.scatter(gen_data_test[:, 0], gen_data_test[:, 1], s=scales, c=(0, 0, 0))

    for i, factor in enumerate(continuous_noise[:, order]):
        ax1.annotate(str(round(factor, 2)), (gen_data_test[i, 0], gen_data_test[i, 1]), color=colors[i])

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
        return logits_real
