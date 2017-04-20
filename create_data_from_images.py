import os
import numpy
from PIL import Image
import properties
import image_processor as processor

directories = {
	0: 'images/raw/bex/',
	1: 'images/raw/peet/'
}
size = properties.image_size


def load_files(label, directory):
	images = []
	for f in os.listdir(directory):
		img = processor.pipeline(Image.open(directory + f))
		images.append(processor.convert_to_array(img))

	images_mat = numpy.asarray(images)
	images_mat = processor.add_label(images_mat, label)
	return images_mat


def save_data(mat, file_name):
	print('Created matrix of size {0}'.format(mat.shape))
	numpy.save(file_name, mat)


def create_training_data():
	mat = numpy.empty((0, (size[0]*size[1]) + 1))
	for key in directories:
		rows = load_files(key, directories[key])
		mat = numpy.vstack((mat, rows))

	dataset_size = mat.shape[0]
	print("Dataset size: ", dataset_size)
	train_sz = int(dataset_size * properties.training_test_split)
	print("Training set size: ", train_sz)
	numpy.random.shuffle(mat)
	splits = numpy.vsplit(mat, [train_sz, dataset_size])
	save_data(splits[0], properties.training_data_filename)
	save_data(splits[1], properties.test_data_filename)

create_training_data()


