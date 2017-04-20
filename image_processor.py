import numpy
import properties

size = properties.image_size


def pipeline(img):
	img = img.convert('L')
	img = img.resize(size)
	return img


def convert_to_array(img):
	return numpy.asarray(img).reshape(-1)


def add_label(images_mat, label):
	lab_mat = numpy.empty((images_mat.shape[0], 1))
	lab_mat.fill(label)
	return numpy.c_[lab_mat, images_mat]

