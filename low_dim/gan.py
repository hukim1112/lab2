from model import data
from model.low_dim import gan
import os
import tensorflow as tf
from matplotlib import pyplot as plt
cat_dim = 10
code_con_dim = 2
total_con_dim = 10
channel = 1
path = '/home/dan/prj/datasets/mnist'
name = 'mnist'
split_name = 'train'
batch_size = 1000
#iteration test
result_dir = os.path.join('/media/dan/DATA/hukim/Research/srnet/low_dim', 'test1', 'result')
ckpt_dir =  os.path.join('/media/dan/DATA/hukim/Research/srnet/low_dim', 'test1', 'weight')
log_dir =  os.path.join('/media/dan/DATA/hukim/Research/srnet/low_dim', 'test1', 'weight')
if not os.path.isdir(result_dir):
	os.makedirs(result_dir)
if not os.path.isdir(ckpt_dir):
	os.makedirs(ckpt_dir)
if not os.path.isdir(log_dir):
	os.makedirs(log_dir)
gan_data = data.Data(cat_dim, code_con_dim, total_con_dim, channel, path, name, split_name, batch_size)
gan_model = gan.Gan(gan_data)
gan_model.train(result_dir, ckpt_dir, log_dir, training_iteration = 10, G_update_num=5, D_update_num=1, Q_update_num=2)
#a = gan_model.test()

del gan_data
del gan_model
