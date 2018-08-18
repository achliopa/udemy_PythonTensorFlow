import tensorflow as tf
import gym
import numpy as np

num_inputs = 4
num_hidden = 4
num_outputs = 1 # Prob to go left

# we use variance scaling initializer like in autoencoders from layers API probably because from 4 we go to 1
initializer = tf.contrib.layers.variance_scaling_initializer()

# we add our placeholder
X = tf.placeholder(tf.float32,shape=[None,num_inputs])

# we create our layers using Layer API
hidden_layer_one = tf.layers.dense(X,num_hidden,activation=tf.nn.relu,kernel_initializer=initializer)
hidden_layer_two = tf.layers.dense(hidden_layer_one,num_hidden,activation=tf.nn.relu,kernel_initializer=initializer)
output_layer = tf.layers.dense(hidden_layer_two,num_outputs,activation=tf.nn.sigmoid,kernel_initializer=initializer)

# from one prob (left) we make 2 mutually exclusive to feed in the multinomial
probabilities = tf.concat(axis=1,values=[output_layer,1-output_layer])

# bring back 1 action from the probabilities (output 0 or 1)
action = tf.multinomial(probabilities,num_samples=1)

init = tf.global_variables_initializer()

epi = 50
step_limit = 500
env = gym.make('CartPole-v0')
avg_steps = []


with tf.Session() as sess:

	sess.run(init)
	# or init.run()

	for i_episode in range(epi):

		obs = env.reset()

		for step in range(step_limit):

			# tf wants its input as 1 dimensional array
			action_val = action.eval(feed_dict={X: obs.reshape(1,num_inputs)})
			# we do indexing to extract multinomial results action_val[0][0] is 0 or 1
			obs,reward,done,info = env.step(action_val[0][0])

			if done:
				avg_steps.append(step)
				print('DONE AFTER {} STEPS'.format(step))
				break

print("AFTER {} EPISODES, AVERAGE STEPS PER GAME WAS {}".format(epi,np.mean(avg_steps)))
env.close()