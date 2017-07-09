import csv
import pandas as pd
import numpy as np
import tensorflow as tf
import sklearn
from sklearn.model_selection import train_test_split

n_hl1 = 500
n_hl2 = 500
n_hl3 = 500
n_hl4 = 500
n_hl5 = 500

n_classes = 2
batch_size = 100

X = tf.placeholder('float')
Y = tf.placeholder('float')

dataset = open('emails.csv', 'r').read()

def email_classifier(x):

	w1 = tf.random_normal([x, n_hl1])
	b1 = tf.random_normal([n_hl1])

	w2 = tf.random_normal([n_hl1, n_hl2])
	b2 = tf.random_normal([n_hl2])

	w3 = tf.random_normal([n_hl2, n_hl3])
	b3 = tf.random_normal([n_hl3])

	w4 = tf.random_normal([n_hl3, n_hl4])
	b4 = tf.random_normal([n_hl4])

	w5 = tf.random_normal([n_hl4, n_hl5])
	b5 = tf.random_normal([n_hl5])

	w_out = tf.random_normal([n_hl5, n_classes])
	b_out = tf.random_normal([n_classes])

	l1 = tf.add(tf.matmul(x, w1), b1)
	l1 = tf.nn.relu(l1)

	l2 = tf.add(tf.matmul(l1, w2), b2)
	l2 = tf.nn.relu(l2)

	l3 = tf.add(tf.matmul(l2, w3), b3)
	l3 = tf.nn.relu(l3)

	l4 = tf.add(tf.matmul(l3, w4), b4)
	l4 = tf.nn.relu(l4)

	l5 = tf.add(tf.matmul(l4, w5), b5)
	l5 = tf.nn.relu(l5)

	ouput = tf.matmul(l5, w_out) + b_out

	return output

def train_email_classifier(x):

	prediction = email_classifier(x)
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=Y))
	optimizer = tf.train.AdamOptimizer().minimize(cost)

	n_epochs = 10

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())

		for epoch in range(n_epochs):
			epoch_loss = 0
			for _ in range(int()):
				epoch_x = dataset['']
				epoch_y = dataset['']
				_, c = sess.run([optimizer, cost], feed_dict={X: epoch_x, Y: epoch_y})

				epoch_loss += c

			print('Epoch {} in total {} loss: {}'.format(epoch, n_epochs, epoch_loss))

		correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
		accuracy = tf.reduce_mean(tf.cast(correct, 'float'))*100

		print('Accuracy {}'.format(accuracy.eval({X: dataset[''], Y: dataset['']})))

# train_email_classifier(X)
