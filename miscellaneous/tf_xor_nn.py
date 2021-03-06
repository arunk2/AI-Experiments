X = [[0, 0],[0, 1],[1, 0],[1, 1]]
Y = [[0], [1], [1], [0]]

# Neural Network Parameters
N_STEPS = 200000
N_EPOCH = 5000
N_TRAINING = len(X)

N_INPUT_NODES = 2
N_HIDDEN_NODES = 5
N_OUTPUT_NODES  = 1
ACTIVATION = 'tanh' # sigmoid or tanh
COST = 'ACE' # MSE or ACE
LEARNING_RATE = 0.05

if __name__ == '__main__':

	##############################################################################
	### Create placeholders for variables and define Neural Network structure  ###
	### Feed forward 3 layer, Neural Network.                                  ###
	
	x_ = tf.placeholder(tf.float32, shape=[N_TRAINING, N_INPUT_NODES], name="x-input")
	y_ = tf.placeholder(tf.float32, shape=[N_TRAINING, N_OUTPUT_NODES], name="y-input")

	theta1 = tf.Variable(tf.random_uniform([N_INPUT_NODES,N_HIDDEN_NODES], -1, 1), name="theta1")
	theta2 = tf.Variable(tf.random_uniform([N_HIDDEN_NODES,N_OUTPUT_NODES], -1, 1), name="theta2")

	bias1 = tf.Variable(tf.zeros([N_HIDDEN_NODES]), name="bias1")
	bias2 = tf.Variable(tf.zeros([N_OUTPUT_NODES]), name="bias2")


	if ACTIVATION == 'sigmoid':
		### Use a sigmoidal activation function ###
		layer1 = tf.sigmoid(tf.matmul(x_, theta1) + bias1)
		output = tf.sigmoid(tf.matmul(layer1, theta2) + bias2)
	else:
		### Use tanh activation function ###
		layer1 = tf.tanh(tf.matmul(x_, theta1) + bias1)
		output = tf.tanh(tf.matmul(layer1, theta2) + bias2)
	
		output = tf.add(output, 1)
		output = tf.multiply(output, 0.5)

	
	if COST == "MSE":
		# Mean Squared Estimate - the simplist cost function (MSE)
		cost = tf.reduce_mean(tf.square(Y - output)) 
		train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cost)
	else:
		# Average Cross Entropy - better behaviour and learning rate
		cost = - tf.reduce_mean( (y_ * tf.log(output)) + (1 - y_) * tf.log(1.0 - output)  )
		train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cost)


	init = tf.initialize_all_variables()
	sess = tf.Session()
	sess.run(init)


	for i in range(N_STEPS):
		sess.run(train_step, feed_dict={x_: X, y_: Y})
		if i % N_EPOCH == 0:
			print('Batch ', i)
			print('Inference ', sess.run(output, feed_dict={x_: X, y_: Y}))
			print('Cost ', sess.run(cost, feed_dict={x_: X, y_: Y}))
			#print('op: ', sess.run(output))
