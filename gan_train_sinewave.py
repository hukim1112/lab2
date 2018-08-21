from model import data
from model.low_dim import gan
import os
import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np
cat_dim = 10
code_con_dim = 1
total_con_dim = 1
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






x = np.float32(np.random.uniform(-1, 1, [1, 1000])[0])
y = np.float32(np.sin(x*np.pi)+np.random.normal(0, 0.1, 1000))
real_data = np.array([[i, j] for i, j in zip(x, y)])

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.scatter(real_data[:, 0], real_data[:, 1], s = 10, c ='b', marker="s", label='first')
fig.savefig(os.path.join(result_dir, 'original.png'), dpi=fig.dpi)


gan_data = data.Data(cat_dim, code_con_dim, total_con_dim, channel, path, name, split_name, batch_size)
gan_data.real_data = real_data
gan_model = gan.Gan(gan_data)
gan_model.train(result_dir, ckpt_dir, log_dir, training_iteration = 20000, G_update_num=5, D_update_num=1)
a = gan_model.test_tmp(result_dir)

del gan_data
del gan_model
