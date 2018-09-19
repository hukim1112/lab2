from model import data
from model.low_dim import infogan
import os
import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np
import data_generator as dg
cat_dim = 10
code_con_dim = 1
total_con_dim = 1
channel = 1
path = '/home/dan/prj/datasets/mnist'
name = 'mnist'
split_name = 'train'
batch_size = 1000
#iteration test
save_path = '/home/dan/prj/lab2/results/infogan'



# test_name = 'test11'
# result_dir = os.path.join(save_path, test_name, 'result')
# ckpt_dir =  os.path.join(save_path, test_name, 'weight')
# log_dir =  os.path.join(save_path, test_name, 'weight')
# if not os.path.isdir(result_dir):
# 	os.makedirs(result_dir)
# if not os.path.isdir(ckpt_dir):
# 	os.makedirs(ckpt_dir)
# if not os.path.isdir(log_dir):
# 	os.makedirs(log_dir)



# real_data = dg.noisy_single_sinewave(0.08)
# fig = plt.figure()
# ax1 = fig.add_subplot(111)
# ax1.scatter(real_data[:, 0], real_data[:, 1], s = 10, c ='b', marker="s", label='first')
# fig.savefig(os.path.join(result_dir, 'original.png'), dpi=fig.dpi)


# gan_data = data.Data(cat_dim, code_con_dim, total_con_dim, channel, path, name, split_name, batch_size)
# gan_data.real_data = real_data
# gan_model = infogan.Infogan(gan_data)
# gan_model.train(result_dir, ckpt_dir, log_dir, training_iteration = 50000, G_update_num=5, D_update_num=1, Q_update_num=2)

# del gan_data
# del gan_model


# test_name = 'test12'
# result_dir = os.path.join(save_path, test_name, 'result')
# ckpt_dir =  os.path.join(save_path, test_name, 'weight')
# log_dir =  os.path.join(save_path, test_name, 'weight')
# if not os.path.isdir(result_dir):
# 	os.makedirs(result_dir)
# if not os.path.isdir(ckpt_dir):
# 	os.makedirs(ckpt_dir)
# if not os.path.isdir(log_dir):
# 	os.makedirs(log_dir)



# real_data = dg.noisy_single_sinewave(0.05)
# fig = plt.figure()
# ax1 = fig.add_subplot(111)
# ax1.scatter(real_data[:, 0], real_data[:, 1], s = 10, c ='b', marker="s", label='first')
# fig.savefig(os.path.join(result_dir, 'original.png'), dpi=fig.dpi)


# gan_data = data.Data(cat_dim, code_con_dim, total_con_dim, channel, path, name, split_name, batch_size)
# gan_data.real_data = real_data
# gan_model = infogan.Infogan(gan_data)
# gan_model.train(result_dir, ckpt_dir, log_dir, training_iteration = 50000, G_update_num=5, D_update_num=1, Q_update_num=2)

# del gan_data
# del gan_model


# test_name = 'test13'
# result_dir = os.path.join(save_path, test_name, 'result')
# ckpt_dir =  os.path.join(save_path, test_name, 'weight')
# log_dir =  os.path.join(save_path, test_name, 'weight')
# if not os.path.isdir(result_dir):
# 	os.makedirs(result_dir)
# if not os.path.isdir(ckpt_dir):
# 	os.makedirs(ckpt_dir)
# if not os.path.isdir(log_dir):
# 	os.makedirs(log_dir)



# real_data = dg.noisy_single_sinewave(0.03)
# fig = plt.figure()
# ax1 = fig.add_subplot(111)
# ax1.scatter(real_data[:, 0], real_data[:, 1], s = 10, c ='b', marker="s", label='first')
# fig.savefig(os.path.join(result_dir, 'original.png'), dpi=fig.dpi)


# gan_data = data.Data(cat_dim, code_con_dim, total_con_dim, channel, path, name, split_name, batch_size)
# gan_data.real_data = real_data
# gan_model = infogan.Infogan(gan_data)
# gan_model.train(result_dir, ckpt_dir, log_dir, training_iteration = 50000, G_update_num=5, D_update_num=1, Q_update_num=2)

# del gan_data
# del gan_model


test_name = 'test14'
result_dir = os.path.join(save_path, test_name, 'result')
ckpt_dir =  os.path.join(save_path, test_name, 'weight')
log_dir =  os.path.join(save_path, test_name, 'weight')
if not os.path.isdir(result_dir):
	os.makedirs(result_dir)
if not os.path.isdir(ckpt_dir):
	os.makedirs(ckpt_dir)
if not os.path.isdir(log_dir):
	os.makedirs(log_dir)



real_data = dg.noisy_single_sinewave(0.01)
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.scatter(real_data[:, 0], real_data[:, 1], s = 10, c ='b', marker="s", label='first')
fig.savefig(os.path.join(result_dir, 'original.png'), dpi=fig.dpi)


gan_data = data.Data(cat_dim, code_con_dim, total_con_dim, channel, path, name, split_name, batch_size)
gan_data.real_data = real_data
gan_model = infogan.Infogan(gan_data)
gan_model.train(result_dir, ckpt_dir, log_dir, training_iteration = 50000, G_update_num=5, D_update_num=1, Q_update_num=2)

del gan_data
del gan_model
