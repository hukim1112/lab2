from matplotlib import pyplot as plt
import tensorflow as tf
slim = tf.contrib.slim
import numpy as np
import time
import cv2
import os
from tensorflow.python.ops import variable_scope
tfgan = tf.contrib.gan
leaky_relu = lambda net: tf.nn.leaky_relu(net, alpha=0.01)

def varying_categorical_noise(self, categorical_dim,
    code_continuous_dim, total_continuous_dim, iteration, result_path):
    """Create noise showing impact of categorical noise in InfoGAN.

    Categorical noise is constant across columns. Other noise is constant across
    rows.

    Args:
    self : model class itself.
    categorical_dim : The number of object to appear in dataset.
    code_continuous_dim : The number of factors to be disentangled in input representation for generating
    total_continuous_dim : The number of continuous factors in input representation for generating
    iteration : global step number
    result_path : path to save the result
    """
    row_num = 10
    categorical_sample_points = np.array(range(categorical_dim))
    continuous_sample_points = np.linspace(-1.0, 1.0, 10)

    rows, cols = row_num, len(categorical_sample_points)

    # Take random draws for non-categorical noise, making sure they are constant
    # across columns.
    continuous_noise = []
    for _ in range(rows):
        cur_sample = np.random.normal(size=[1, total_continuous_dim - code_continuous_dim])
        continuous_noise.extend([cur_sample] * cols)
    continuous_noise = np.concatenate(continuous_noise)
    # continuous_noise is nxm. n : rows x cols, m : total_continuous_dim - code_continuous_dim. Each of cols number rows is the same. 

    # Increase categorical noise from left to right, making sure they are constant
    # across rows.
    categorical_code = np.tile(np.eye(categorical_dim)[categorical_sample_points], (rows, 1))
      
    # Take random draws for non-categorical noise, making sure they are constant
    # across columns.
    continuous_code = []
    for _ in range(rows):
        cur_sample = np.random.choice(
            continuous_sample_points, size=[1, code_continuous_dim])
        continuous_code.extend([cur_sample] * cols)
    continuous_code = np.concatenate(continuous_code)

    display_images = []
    with variable_scope.variable_scope(self.gen_scope.name, reuse = True):
        display_images = self.generator(np.float32(continuous_noise), [np.float32(categorical_code), np.float32(continuous_code)])

    display_img = tfgan.eval.image_reshaper(tf.concat(display_images, 0), num_cols=cols)
    results = np.squeeze(self.sess.run(display_img))
    results = results*128 + 128
    cv2.imwrite(os.path.join(result_path , str(iteration)+'_categorization.png'), results.astype(np.uint8))
    print(str(iteration)+'th result saved')



def varying_noise_continuous_ndim(self, order, categorical_dim,
    code_continuous_dim, total_continuous_dim, iteration, result_path, name=None):
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
    row_num = 10
    categorical_sample_points = np.array(range(categorical_dim))
    continuous_sample_points = np.linspace(-1.0, 1.0, 10)

    rows, cols = row_num, len(continuous_sample_points)


    # Take random draws for non-categorical noise, making sure they are constant
    # across columns.
    continuous_noise = []
    for _ in range(rows):
        cur_sample = np.random.normal(size=[1, total_continuous_dim - code_continuous_dim])
        continuous_noise.extend([cur_sample] * cols)
    continuous_noise = np.concatenate(continuous_noise)
    # continuous_noise is nxm. n : rows x cols, m : total_continuous_dim - code_continuous_dim. Each of cols number rows is the same. 


    categorical_code = []
    for _ in range(rows):
        cur_sample = np.random.choice(categorical_sample_points) #random sampling from categorical sample points
        categorical_code.extend( [np.eye(categorical_dim)[np.tile(cur_sample, cols)]] ) #repeat sample by # of columns
    categorical_code = np.concatenate(categorical_code) 
    #concatenate categorical codes for each row

    continuous_code = []
    for _ in range(rows):
        cur_sample = np.random.normal(size=[1, code_continuous_dim])
        continuous_code.extend([cur_sample]*cols)
    continuous_code = np.concatenate(continuous_code)

    varying_factor = np.tile(continuous_sample_points, rows)
    continuous_code[:, order] = varying_factor 

    display_images = []
    with variable_scope.variable_scope(self.gen_scope.name, reuse = True):
        display_images = self.generator(np.float32(continuous_noise), [np.float32(categorical_code), np.float32(continuous_code)])

    display_img = tfgan.eval.image_reshaper(tf.concat(display_images, 0), num_cols=cols)
    results = np.squeeze(self.sess.run(display_img))
    results = results*128 + 128
    if name==None:
        cv2.imwrite(os.path.join(result_path , str(iteration)+'_continuous'+str(order)+'.png'), results.astype(np.uint8))
        print(str(iteration)+'th result saved')
    else:
        cv2.imwrite(os.path.join(result_path , str(iteration)+'_continuous_'+ name+'.png'), results.astype(np.uint8))
        print(str(iteration)+'th result saved')



def varying_noise_continuous_ndim_without_category(self, order, total_continuous_dim, iteration, result_path):
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
    row_num = 10
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
        display_images = self.generator(continuous_noise)

    print('display_images shape : ', display_images.shape)

    display_img = tfgan.eval.image_reshaper(tf.concat(display_images, 0), num_cols=cols)
    results = np.squeeze(self.sess.run(display_img))
    
    print('results shape : ', results.shape)
    results = results*128 + 128
    cv2.imwrite(os.path.join(result_path , str(iteration)+'_continuous'+str(order)+'.png'), results.astype(np.uint8))
    print(str(iteration)+'th result saved')