import tensorflow as tf
import numpy as np
import properties

number_of_classes = properties.number_of_classes


def load_matrices(filename):
	training_data = np.load(properties.training_data_filename)
	train_split = np.hsplit(training_data, [1, training_data.shape[1]])
	return train_split[1], train_split[0]


train_x, train_y = load_matrices(properties.training_data_filename)
test_x, test_y = load_matrices(properties.test_data_filename)

print("train_y size: ", train_y.shape)
print("test_y size: ", test_y.shape)

size = properties.image_size
flattened_size = size[0]*size[1]
learning_rate = properties.learning_rate

x = tf.placeholder(tf.float32, [None, flattened_size])
W = tf.Variable(tf.zeros([flattened_size, number_of_classes]))
b = tf.Variable(tf.zeros([number_of_classes]))

y = tf.nn.softmax(tf.matmul(x, W) + b)
y_ = tf.placeholder(tf.float32, [None, 1	])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
sess.run(train_step, feed_dict={x: train_x, y_: train_y})

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: test_x, y_: test_y}))

