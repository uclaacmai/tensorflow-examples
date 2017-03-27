'''
Tensorboard visualization for a basic MNIST classifier
from https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/4_Utils/tensorboard_basic.py
'''

from __future__ import print_function

import tensorflow as tf

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# Parameters
learning_rate = 0.01
training_epochs = 1
batch_size = 100
display_step = 1
logs_path = '/tmp/tensorflow_logs/example'


x = tf.placeholder(tf.float32, [None, 784], name='x-data')
# 0-9 digits recognition => 10 classes
y = tf.placeholder(tf.float32, [None, 10], name='y-data')

# Set model weights
def weight_variable(shape, name = 'weights'):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name = name)

def bias_variable(shape, name = 'bias'):
    initial = tf.constant(0.1, shape = shape)
    return tf.Variable(initial, name = name)

W, b = weight_variable([784, 10]), bias_variable([10])

with tf.name_scope('Model'):
    W_1, b_1 = weight_variable([784, 100], name = 'weights-1'), bias_variable([100], name = 'bias-1')
    with tf.name_scope('FC-layer-1'):
        act_fc_1 = tf.nn.relu(tf.matmul(x, W_1) + b_1, name = 'hidden-layer-activations')
    with tf.name_scope('Softmax-layer'):
        W_2, b_2 = weight_variable([100, 10], name = 'weights-2'), bias_variable([10], name = 'bias-2')
        pred = tf.nn.softmax(tf.matmul(act_fc_1, W_2) + b_2, name = 'softmax-predictions')
with tf.name_scope('Loss'):
    # Minimize error using cross entropy
    cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred), reduction_indices=1))
with tf.name_scope('SGD'):
    # Gradient Descent
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
with tf.name_scope('Accuracy'):
    # Accuracy
    acc = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    acc = tf.reduce_mean(tf.cast(acc, tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()

# Create a summary to monitor cost tensor
tf.summary.scalar("loss", cost)
# Create a summary to monitor accuracy tensor
tf.summary.scalar("accuracy", acc)
# Merge all summaries into a single op
merged_summary_op = tf.summary.merge_all()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    # op to write logs to Tensorboard
    summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples/batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            # Run optimization op (backprop), cost op (to get loss value)
            # and summary nodes
            _, c, summary = sess.run([optimizer, cost, merged_summary_op],
                                     feed_dict={x: batch_xs, y: batch_ys})
            # Write logs at every iteration
            summary_writer.add_summary(summary, epoch * total_batch + i)
            # Compute average loss
            avg_cost += c / total_batch
        # Display logs per epoch step
        if (epoch+1) % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))

    print("Optimization Finished!")

    # Test model
    # Calculate accuracy
    print("Accuracy:", acc.eval({x: mnist.test.images, y: mnist.test.labels}))

    print("Run the command line:\n" \
          "--> tensorboard --logdir=/tmp/tensorflow_logs/examples " \
          "\nThen open up localhost:6006")
