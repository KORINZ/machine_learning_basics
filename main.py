import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

####################################################################
# Batch Linear Classifier (2 types of random points in a 2D plane) #
####################################################################

# affine transformation, prediction == W \dot input + b

num_samples_per_class = 1000

# class 0, shape = (1000, 2)
negative_samples = np.random.multivariate_normal(
    mean=[0, 3],
    cov=[[1, 0.5], [0.5, 1]],
    size=num_samples_per_class)

# class 1, shape = (1000, 2)
positive_samples = np.random.multivariate_normal(
    mean=[3, 0],
    cov=[[1, 0.5], [0.5, 1]],
    size=num_samples_per_class)

# samples
inputs = np.vstack((negative_samples, positive_samples)).astype(np.float32)  # shape = (2000, 2)

# targets
targets = np.vstack((np.zeros((num_samples_per_class, 1), dtype="float32"),
                     np.ones((num_samples_per_class, 1),
                             dtype="float32")))  # [[0. 0. 0. ... 1. 1. 1.]], shape = (2000, 1)

# linear classifier variables
input_dim = 2
output_dim = 1
W = tf.Variable(initial_value=tf.random.uniform(shape=(input_dim, output_dim)))
b = tf.Variable(initial_value=tf.zeros(shape=(output_dim,)))


# forward pass function
def model(inputs):
    return tf.matmul(inputs, W) + b  # [[w1], [w2]] \dot [x, y] + b == w1 * x + w2 * y + b


# loss function (mean squared error)
def square_loss(targets, predictions):
    per_samples_losses = tf.square(targets - predictions)
    return tf.reduce_mean(per_samples_losses)  # average single loss value


# training step function
learning_rate = 0.1  # batch training


def training_step(inputs, targets):
    with tf.GradientTape() as tape:
        predictions = model(inputs)
        loss = square_loss(predictions, targets)
    grad_loss_wrt_W, grad_loss_wrt_b = tape.gradient(loss, [W, b])
    W.assign_sub(grad_loss_wrt_W * learning_rate)  # update weight
    b.assign_sub(grad_loss_wrt_b * learning_rate)  # update bias
    return loss


# Batch training loop
for step in range(50):
    loss = training_step(inputs, targets)
    print(f"Loss at step {step}: {loss:.4f}")

# plotting section
predictions = model(inputs)
x = np.linspace(-2, 5, 100)
y = -W[0] / W[1] * x + (0.5 - b) / W[1]  # y = m * x + b form; model visualized as a line
plt.plot(x, y, "-r")
plt.scatter(inputs[:, 0], inputs[:, 1], c=predictions[:, 0] > 0.5)  # < 0.5, class 0; > 0.5 class 1

# plt.scatter(inputs[:, 0], inputs[:, 1], c=targets[:, 0]) # samples and labels plot
plt.show()
