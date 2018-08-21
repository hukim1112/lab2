import tensorflow as tf
import os
import random
import math
import sys
from datasets.tfrecord_creator import tf_encoder
import json

class ImageReader(object):
  """Helper class that provides TensorFlow image coding utilities."""
  # Later ImageReader will classify png and jpg automatically and decode that

  def __init__(self):
    # Initializes function that decodes RGB JPEG data.
    self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
    self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)

  def read_image_dims(self, sess, image_data):
    image = self.decode_jpeg(sess, image_data)
    return image.shape[0], image.shape[1]

  def decode_jpeg(self, sess, image_data):
    image = sess.run(self._decode_jpeg,
                     feed_dict={self._decode_jpeg_data: image_data})
    assert len(image.shape) == 3
    assert image.shape[2] == 3
    return image

class tf_converter:
	# this class contains utility functions and variables for creating tfrecord conveniently. 
	# simple usage like below.
	# datasetname = 'flowers'
	# dataset_dir = '/home/dan/prj/generative_models/flower_photos'
	# _NUM_VALIDATION = 400
	# _NUM_SHARD = 5
	# tfrecord = converter.tf_converter(datasetname, dataset_dir, _NUM_VALIDATION, _NUM_SHARD)
	# tfrecord.run()
	def __init__(self, data_name, dataset_dir, _NUM_VALIDATION, _NUM_SHARDS):
		# Seed for repeatability.
		self.dataname = data_name
		self.dataset_dir = dataset_dir
		self._NUM_VALIDATION = _NUM_VALIDATION
		self._NUM_SHARDS = 5
		self._RANDOM_SEED = 0
		self.LABELS_FILENAME = 'labels.txt'

	def run(self):
		  """Runs the download and conversion operation.

		  Args:
		    dataset_dir: The dataset directory where the dataset is stored.
		  """
		  if not tf.gfile.Exists(self.dataset_dir):
		    tf.gfile.MakeDirs(self.dataset_dir)

		  if self._dataset_exists(self.dataset_dir):
		    print('Dataset files already exist. Exiting without re-creating them.')
		    return

		  photo_filenames, class_names = self._get_filenames_and_classes(self.dataset_dir)
		  class_names_to_ids = dict(zip(class_names, range(len(class_names))))


		  # Divide into train and test:
		  random.seed(self._RANDOM_SEED)
		  random.shuffle(photo_filenames)
		  training_filenames = photo_filenames[self._NUM_VALIDATION:]
		  validation_filenames = photo_filenames[:self._NUM_VALIDATION]


		  datainfo = os.path.join(self.dataset_dir, 'datainfo.json')
		  with open(datainfo, 'w') as f:
		      json.dump(
		      	{'split' : {'train' : len(training_filenames), 'validation' : len(validation_filenames)}, 
		      	'num_class' : len(class_names)}, f, indent = 1)		  


		  # First, convert the training and validation sets.
		  self._convert_dataset('train', training_filenames, class_names_to_ids,
		                   self.dataset_dir)
		  self._convert_dataset('validation', validation_filenames, class_names_to_ids,
		                   self.dataset_dir)

		  # Finally, write the labels file:
		  labels_to_class_names = dict(zip(range(len(class_names)), class_names))
		  self.write_label_file(labels_to_class_names, self.dataset_dir, self.LABELS_FILENAME)

		  print('\nFinished converting the dataset!')		

	def write_label_file(self, labels_to_class_names, dataset_dir, filename):
	  """Writes a file with the list of class names.

	  Args:
	    labels_to_class_names: A map of (integer) labels to class names.
	    dataset_dir: The directory in which the labels file should be written.
	    filename: The filename where the class names are written.
	  """
	  labels_filename = os.path.join(dataset_dir, filename)
	  with tf.gfile.Open(labels_filename, 'w') as f:
	    for label in labels_to_class_names:
	      class_name = labels_to_class_names[label]
	      f.write('%d:%s\n' % (label, class_name))

	def _dataset_exists(self, dataset_dir):
	  for split_name in ['train', 'validation']:
	    for shard_id in range(self._NUM_SHARDS):
	      output_filename = self._get_dataset_filename(
	          self.dataset_dir, split_name, shard_id)
	      if not tf.gfile.Exists(output_filename):
	        return False
	  return True	

	def _get_filenames_and_classes(self, dataset_dir):
	  """Returns a list of filenames and inferred class names.

	  Args:
	    dataset_dir: A directory containing a set of subdirectories representing
	      class names. Each subdirectory should contain PNG or JPG encoded images.

	  Returns:
	    A list of image file paths, relative to `dataset_dir` and the list of
	    subdirectories, representing class names.
	  """
	  dataset_root = os.path.join(dataset_dir)
	  directories = []
	  class_names = []
	  for dirname in os.listdir(dataset_root):
	    path = os.path.join(dataset_root, dirname)
	    if os.path.isdir(path):
	      directories.append(path)
	      class_names.append(dirname)

	  photo_filenames = []
	  for directory in directories:
	    for filename in os.listdir(directory):
	      path = os.path.join(directory, filename)
	      photo_filenames.append(path)

	  return photo_filenames, sorted(class_names)


	def _convert_dataset(self, split_name, filenames, class_names_to_ids, dataset_dir):
	  """Converts the given filenames to a TFRecord dataset.

	  Args:
	    split_name: The name of the dataset, either 'train' or 'validation'.
	    filenames: A list of absolute paths to png or jpg images.
	    class_names_to_ids: A dictionary from class names (strings) to ids
	      (integers).
	    dataset_dir: The directory where the converted datasets are stored.
	  """
	  assert split_name in ['train', 'validation']

	  num_per_shard = int(math.ceil(len(filenames) / float(self._NUM_SHARDS)))

	  with tf.Graph().as_default():
	    image_reader = ImageReader()

	    with tf.Session('') as sess:

	      for shard_id in range(self._NUM_SHARDS):
	        output_filename = self._get_dataset_filename(
	            dataset_dir, split_name, shard_id)

	        with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
	          start_ndx = shard_id * num_per_shard
	          end_ndx = min((shard_id+1) * num_per_shard, len(filenames))
	          for i in range(start_ndx, end_ndx):
	            sys.stdout.write('\r>> Converting image %d/%d shard %d' % (
	                i+1, len(filenames), shard_id))
	            sys.stdout.flush()

	            # Read the filename:
	            image_data = tf.gfile.FastGFile(filenames[i], 'rb').read()
	            height, width = image_reader.read_image_dims(sess, image_data)

	            class_name = os.path.basename(os.path.dirname(filenames[i]))
	            class_id = class_names_to_ids[class_name]

	            example = tf_encoder.image_to_tfexample(
	                image_data, b'jpg', height, width, class_id)
	            tfrecord_writer.write(example.SerializeToString())

	  sys.stdout.write('\n')
	  sys.stdout.flush()

	def _get_dataset_filename(self, dataset_dir, split_name, shard_id):
	  output_filename = '%s_%s_%05d-of-%05d.tfrecord' % (
	      self.dataname, split_name, shard_id, self._NUM_SHARDS)
	  return os.path.join(dataset_dir, output_filename)