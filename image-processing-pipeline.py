import os
import numpy
from PIL import Image
import properties

directories = {
	0: 'images/raw/bex/',
	1: 'images/raw/peet/'
}
size = properties.image_size


def load_files(label, directory):
	images = []
	for f in os.listdir(directory):
		img = Image.open(directory+f)
		img = img.convert('L')
		img = img.resize(size)
		img_as_row = numpy.asarray(img)
		img_as_row = img_as_row.reshape(-1)
		images.append(img_as_row)
	images_mat = numpy.asarray(images)
	lab_mat = numpy.empty((images_mat.shape[0], 1))
	lab_mat.fill(label)
	images_mat = numpy.c_[lab_mat, images_mat]
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
