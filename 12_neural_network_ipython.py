# https://www.tensorflow.org/versions/r0.12/tutorials/mnist/pros/index.html#deep-mnist-for-experts


# Step 1 : Get data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


# Interactive session 
import tensorflow as tf
sess = tf.InteractiveSession()

# setting up placeholder values for x and y_
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])


# filling all the placeholder values with zeros
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))


# initialing the graph
sess.run(tf.global_variables_initializer())


# setting up model
y = tf.matmul(x,W) + b


# setting up loss function
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))


# setting up the training set with loss function
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)


# train the model
for i in range(1000):
  batch = mnist.train.next_batch(100)
  train_step.run(feed_dict={x: batch[0], y_: batch[1]})


# set up a function to determine if predections
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

# setup accuracy function calculations
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# print out accuracy number ~92 percent hopefully
print(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

