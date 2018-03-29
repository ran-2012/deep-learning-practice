import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 生成数据
rx = np.linspace(-1, 1, 100)
ry = 2 * rx + np.random.randn(*rx.shape) * 0.3

# 正向
x = tf.placeholder("float")
y = tf.placeholder("float")

w = tf.Variable(tf.random_normal([1]), name = "weight")
b = tf.Variable(tf.zeros([1]), name = "bias")

z = tf.multiply(x, w) + b

# 反向
cost = tf.reduce_mean(tf.square(y - z))
learning_rate = 0.01
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# 运行模型
init = tf.global_variables_initializer()

training_epochs = 20
display_step = 2

with tf.Session() as sess:
	sess.run(init)
	plotdata = {"batch_size": [], "loss": []}

	for epoch in range(training_epochs):
		for (lx, ly) in zip(rx, ry):
			sess.run(optimizer, feed_dict = {x: lx, y: ly})

		if epoch % display_step == 0:
			loss = sess.run(cost, feed_dict = {x: rx, y: ry})
			print("Epoch:", epoch + 1, "cost=", loss, "W=", sess.run(w), "b=", sess.run(b))

			if not (loss == 'NA'):
				plotdata["batch_size"].append(epoch)
				plotdata["loss"].append(loss)
	print("Completed")
	print("cost=", sess.run(cost, feed_dict = {x: rx, y: ry})), "w=", sess.run(w), "b=", sess.run(b)

# 画图
	plt.figure(1)
	plt.subplot(211)
	plt.plot(rx, ry, 'ro', label = 'Raw data')
	plt.plot(rx, sess.run(w) * rx + sess.run(b), label="Fitted line")
	plt.legend()

	plt.subplot(212)
	plt.plot(plotdata["batch_size"], plotdata["loss"], 'b--')
	plt.xlim(0, 19)
	plt.x
	plt.xlabel("Minibatch number")
	plt.ylabel("Loss")
	plt.show()

