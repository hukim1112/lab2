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

leaky_relu = lambda net: tf.nn.leaky_relu(net, alpha=0.01)

def get_infogan_noise(batch_size, categorical_dim, code_continuous_dim,
                      total_continuous_noise_dims):
  """Get unstructured and structured noise for InfoGAN.

  Args:
    batch_size: The number of noise vectors to generate.
    categorical_dim: The number of categories in the categorical noise.
    code_continuous_dim: The number of dimensions of the uniform
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
      [batch_size, total_continuous_noise_dims - code_continuous_dim])

  # Get categorical noise Tensor.
  categorical_dist = ds.Categorical(logits=tf.zeros([categorical_dim]))
  categorical_code = categorical_dist.sample([batch_size])
  categorical_code = tf.one_hot(categorical_code, categorical_dim)

  # Get continuous noise Tensor.
  continuous_dist = ds.Uniform(-tf.ones([code_continuous_dim]),
                               tf.ones([code_continuous_dim]))
  continuous_code = continuous_dist.sample([batch_size])

  return noise, [categorical_code, continuous_code]

class Megan():
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
        self.visual_prior_path = self.data.visual_prior_path
        with self.graph.as_default():
            with slim.queues.QueueRunners(self.sess):
                self.dataset, self.real_data, self.labels = load_batch(self.dataset_path, self.dataset_name, self.split_name, self.batch_size)

                visual_prior = {'category' : list(range(10)), 'rotation' : ['min', 'max'], 'width' : ['min', 'max']}
                self.visual_prior_images = {}
                for key in visual_prior.keys():
                    self.visual_prior_images[key] = {}

                    for attribute in visual_prior[key]:
                        self.visual_prior_images[key][attribute] = []
                        path = os.path.join(self.visual_prior_path, key, str(attribute))
                        for img_file in os.listdir(path):
                            sample = cv2.imread(os.path.join(path, img_file))
                            sample = cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY)
                            sample = (tf.to_float(sample) - 128.0) / 128.0
                            sample = tf.reshape(sample, (28, 28, 1))
                            self.visual_prior_images[key][attribute].append(sample)
                        self.visual_prior_images[key][attribute] = ops.convert_to_tensor(self.visual_prior_images[key][attribute])

                self.variation_key = [key_name for key_name in self.visual_prior_images.keys() if key_name!='category']
                self.variation_key.sort()



                tf.train.start_queue_runners(self.sess)
                self.gen_input_noise, self.gen_input_code = get_infogan_noise(self.batch_size, self.cat_dim, self.code_con_dim, self.total_con_dim)

                with variable_scope.variable_scope('generator') as self.gen_scope:
                    self.gen_data = self.generator(self.gen_input_noise, self.gen_input_code) #real/fake loss
                
                with variable_scope.variable_scope('discriminator') as self.dis_scope:
                    self.dis_gen_data, self.Q_net = self.discriminator(self.gen_data, self.cat_dim, self.code_con_dim) #real/fake loss + I(c' ; X_{data}) loss
                with variable_scope.variable_scope(self.dis_scope.name, reuse = True):
                    self.real_data = ops.convert_to_tensor(self.real_data)
                    self.dis_real_data, _ = self.discriminator(self.real_data, self.cat_dim, self.code_con_dim) #real/fake loss 

                #loss
                self.dis_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.dis_scope.name)
                self.gen_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.gen_scope.name)

                self.D_loss = losses_fn.wasserstein_discriminator_loss(self.dis_real_data, self.dis_gen_data)
                self.G_loss = losses_fn.wasserstein_generator_loss(self.dis_gen_data)
                self.wasserstein_gradient_penalty_loss = losses_fn.wasserstein_gradient_penalty_infogan(self, self.real_data, self.gen_data)
                self.mutual_information_loss = losses_fn.mutual_information_penalty(self.gen_input_code, self.Q_net)
                self.visual_prior_penalty = losses_fn.visual_prior_penalty(self, self.visual_prior_images)

                tf.summary.scalar('D_loss', self.D_loss + self.wasserstein_gradient_penalty_loss)
                tf.summary.scalar('G_loss', self.G_loss)
                tf.summary.scalar('Mutual_information_loss', self.mutual_information_loss)
                tf.summary.scalar('visual_prior_loss', self.visual_prior_penalty)
                # tf.summary.scalar('log_prob_cat', self.log_prob_cat)
                # tf.summary.scalar('log_prob_con', self.log_prob_con)
                self.merged = tf.summary.merge_all()

                self.global_step = tf.Variable(0, name='global_step', trainable=False)
                
                #solver
                self.D_solver = tf.train.AdamOptimizer(0.001, beta1=0.5).minimize(self.D_loss+self.wasserstein_gradient_penalty_loss, var_list=self.dis_var, global_step=self.global_step)
                self.G_solver = tf.train.AdamOptimizer(0.0001, beta1=0.5).minimize(self.G_loss, var_list=self.gen_var)
                self.mutual_information_solver = tf.train.AdamOptimizer(0.0001, beta1=0.5).minimize(self.mutual_information_loss, var_list=self.gen_var + self.dis_var)
                self.visual_prior_solver = tf.train.AdamOptimizer(0.001, beta1=0.5).minimize(self.visual_prior_penalty, var_list=self.dis_var)
                self.saver = tf.train.Saver()
                self.initializer = tf.global_variables_initializer()
    def train(self, result_dir, ckpt_dir, log_dir, training_iteration = 1000000, G_update_num=1, D_update_num=1, Q_update_num=1):
        with self.graph.as_default():
            path_to_latest_ckpt = tf.train.latest_checkpoint(checkpoint_dir=ckpt_dir)
            if path_to_latest_ckpt == None:
                print('scratch from random distribution')
                self.sess.run(self.initializer)
            else:
                self.saver.restore(self.sess, path_to_latest_ckpt)
                print('restore')
            self.train_writer = tf.summary.FileWriter(log_dir, self.sess.graph)
            for i in range(training_iteration):
                for _ in range(D_update_num):
                    self.sess.run(self.D_solver)
                for _ in range(G_update_num):
                    self.sess.run(self.G_solver)
                for _ in range(Q_update_num):
                    self.sess.run(self.mutual_information_solver)
                for _ in range(1):
                    self.sess.run(self.visual_prior_solver)
                merge, global_step = self.sess.run([self.merged, self.global_step])
                self.train_writer.add_summary(merge, global_step)
                if ((i % 1000) == 0):
                    for j in range(self.code_con_dim):
                        visualizations.varying_noise_continuous_ndim(self, j, self.cat_dim, self.code_con_dim, self.total_con_dim, global_step, result_dir, name=self.variation_key[j])
                    visualizations.varying_categorical_noise(self, self.cat_dim, self.code_con_dim, self.total_con_dim, global_step, result_dir)
                if ((i % 1000) == 0 ):
                    self.saver.save(self.sess, os.path.join(ckpt_dir, 'model'), global_step=self.global_step)
    def evaluate_with_random_sample(self, result_dir, ckpt_dir):
        with self.graph.as_default():
            path_to_latest_ckpt = tf.train.latest_checkpoint(checkpoint_dir=ckpt_dir)
            if path_to_latest_ckpt == None:
                print('There is no trained weight files...')
                return
            else:
                self.saver.restore(self.sess, path_to_latest_ckpt)
                print('restored')
                images = self.sess.run(self.gen_data)
                print('shape check of result : ', images.shape)

                for i in range(len(images)):
                    cv2.imwrite(os.path.join(result_dir, str(i)+'.jpg'), images[i])
    def test(self):
        print(ops.convert_to_tensor(self.visual_prior_images['category'][0]).shape)
        print(self.real_data.shape)

def load_batch(dataset_path, dataset_name, split_name, batch_size=128):

    #Data pipeline
    dataset = mnist_reader.get_split(split_name, dataset_path)
    print(dataset_name)
    print(split_name)
    data_provider = slim.dataset_data_provider.DatasetDataProvider(
                    dataset, common_queue_capacity=4*batch_size, common_queue_min=batch_size)    
    [image, label] = data_provider.get(['image', 'label'])
    image = (tf.to_float(image) - 128.0) / 128.0 # convert 0~255 scale into -1~1 scale
    images, labels = tf.train.batch(
              [image, label],
              batch_size=batch_size,
              num_threads=4,
              capacity=2 * batch_size)
    print('batch image size :', images.shape)
    return dataset, images, labels


def generator(gen_input_noise, gen_input_code, weight_decay=2.5e-5):
    """InfoGAN discriminator network on MNIST digits.
    
    Based on a paper https://arxiv.org/abs/1606.03657 and their code
    https://github.com/openai/InfoGAN.
    
    Returns:
        A generated image in the range [-1, 1].
    """
    gen_input_code = tf.concat([gen_input_code[0], gen_input_code[1]], axis = 1)
    all_noise = tf.concat([gen_input_noise, gen_input_code], axis=1)
    with slim.arg_scope(
        [layers.fully_connected, layers.conv2d_transpose],
        activation_fn=leaky_relu, normalizer_fn=layers.batch_norm,
        weights_regularizer=layers.l2_regularizer(weight_decay)):
        net = layers.fully_connected(all_noise, 1024)
        net = layers.fully_connected(net, 7 * 7 * 128)
        net = tf.reshape(net, [-1, 7, 7, 128])
        net = layers.conv2d_transpose(net, 64, [4, 4], stride=2)
        net = layers.conv2d_transpose(net, 32, [4, 4], stride=2)
        # Make sure that generator output is in the same range as `inputs`
        # ie [-1, 1].
        net = layers.conv2d(net, 1, 4, normalizer_fn=None, activation_fn=tf.tanh)   
        return net


def discriminator(img, categorical_dim=10, continuous_dim=2, weight_decay=2.5e-5):
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
        net = layers.conv2d(img, 64, [4, 4], stride=2)
        net = layers.conv2d(net, 128, [4, 4], stride=2)
        net = layers.flatten(net)
        net = layers.fully_connected(net, 1024, normalizer_fn=layers.batch_norm)
    
        logits_real = layers.fully_connected(net, 1, activation_fn=None)

        # Recognition network for latent variables has an additional layer
        encoder = layers.fully_connected(net, 128, normalizer_fn=layers.batch_norm)

        # Compute logits for each category of categorical latent.
        q_cat = layers.fully_connected(
            encoder, categorical_dim, activation_fn=None)
        #q_cat = ds.Categorical(q_cat)

        # Compute mean for Gaussian posterior of continuous latents.
        q_cont = layers.fully_connected(
            encoder, continuous_dim, activation_fn=None)
        #sigma_cont = tf.ones_like(q_cont)
        #q_cont = ds.Normal(loc=q_cont, scale=sigma_cont)

        return logits_real, [q_cat, q_cont]

