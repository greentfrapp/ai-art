import threading
import multiprocessing
import psutil
import numpy as np
import tensorflow as tf
import scipy.signal
from time import sleep
import os
import sys
from absl import flags
from absl.flags import FLAGS
from keras.applications.resnet50 import ResNet50

from svg_env import SvgEnv

"""
Use the following command to launch Tensorboard:
tensorboard --logdir=worker_0:'./train_0',worker_1:'./train_1',worker_2:'./train_2',worker_3:'./train_3'
"""


## HELPER FUNCTIONS

# Copies one set of variables to another.
# Used to set worker network parameters to those of global network.
def update_target_graph(from_scope,to_scope):
	from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
	to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)
	op_holder = []
	for from_var,to_var in zip(from_vars,to_vars):
		op_holder.append(to_var.assign(from_var))
	return op_holder

# Discounting function used to calculate discounted returns.
def discount(x, gamma):
	return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]

# Used to initialize weights for policy and value output layers
def normalized_columns_initializer(std=1.0):
	def _initializer(shape, dtype=None, partition_info=None):
		out = np.random.randn(*shape).astype(np.float32)
		out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
		return tf.constant(out)
	return _initializer

# Sample from a given distribution
def sample_dist(dist):
	sample = np.random.choice(dist[0],p=dist[0])
	sample = np.argmax(dist == sample)
	return sample

## ACTOR-CRITIC NETWORK

class AC_Network():
	def __init__(self, scope, trainer):
		with tf.variable_scope(scope):

			self.inputs = tf.placeholder(
				shape=[None, 200, 200, 3],
				dtype=tf.float32,
				name='inputs',
			)

			self.conv1 = tf.layers.conv2d(
				inputs=self.inputs,
				filters=16,
				kernel_size=[8,8],
				strides=[4,4],
				padding='valid',
				activation=tf.nn.relu)
			self.conv2 = tf.layers.conv2d(
				inputs=self.conv1,
				filters=32,
				kernel_size=[4,4],
				strides=[2,2],
				padding='valid',
				activation=tf.nn.relu)

			output_length = 1
			for dim in self.conv2.get_shape().as_list()[1:]:
				output_length *= dim

			self.latent_vector = tf.layers.dense(
				inputs=tf.reshape(self.conv2,shape=[-1, output_length]),
				units=256,
				activation=tf.nn.relu)

			self.policy_i_x = tf.layers.dense(
				inputs=self.latent_vector,
				units=50,
				activation=tf.nn.softmax,
				kernel_initializer=normalized_columns_initializer(0.01),
			)
			self.policy_i_y = tf.layers.dense(
				inputs=self.latent_vector,
				units=50,
				activation=tf.nn.softmax,
				kernel_initializer=normalized_columns_initializer(0.01),
			)
			self.policy_ii_x = tf.layers.dense(
				inputs=self.latent_vector,
				units=50,
				activation=tf.nn.softmax,
				kernel_initializer=normalized_columns_initializer(0.01),
			)
			self.policy_ii_y = tf.layers.dense(
				inputs=self.latent_vector,
				units=50,
				activation=tf.nn.softmax,
				kernel_initializer=normalized_columns_initializer(0.01),
			)
			self.policy_iii_x = tf.layers.dense(
				inputs=self.latent_vector,
				units=50,
				activation=tf.nn.softmax,
				kernel_initializer=normalized_columns_initializer(0.01),
			)
			self.policy_iii_y = tf.layers.dense(
				inputs=self.latent_vector,
				units=50,
				activation=tf.nn.softmax,
				kernel_initializer=normalized_columns_initializer(0.01),
			)

			self.policy_color = tf.layers.dense(
				inputs=self.latent_vector,
				units=5,
				activation=tf.nn.softmax,
				kernel_initializer=normalized_columns_initializer(0.01),
			)

			self.value = tf.layers.dense(
				inputs=self.latent_vector,
				units=1,
				kernel_initializer=normalized_columns_initializer(1.0))

			# Only the worker network need ops for loss functions and gradient updating.
			if scope != 'global':
				self.actions_i_x = tf.placeholder(shape=[None], dtype=tf.int32)
				self.actions_onehot_i_x = tf.one_hot(self.actions_i_x, 50, dtype=tf.float32)
				self.actions_i_y = tf.placeholder(shape=[None], dtype=tf.int32)
				self.actions_onehot_i_y = tf.one_hot(self.actions_i_y, 50, dtype=tf.float32)
				self.actions_ii_x = tf.placeholder(shape=[None], dtype=tf.int32)
				self.actions_onehot_ii_x = tf.one_hot(self.actions_ii_x, 50, dtype=tf.float32)
				self.actions_ii_y = tf.placeholder(shape=[None], dtype=tf.int32)
				self.actions_onehot_ii_y = tf.one_hot(self.actions_ii_y, 50, dtype=tf.float32)
				self.actions_iii_x = tf.placeholder(shape=[None], dtype=tf.int32)
				self.actions_onehot_iii_x = tf.one_hot(self.actions_iii_x, 50, dtype=tf.float32)
				self.actions_iii_y = tf.placeholder(shape=[None], dtype=tf.int32)
				self.actions_onehot_iii_y = tf.one_hot(self.actions_iii_y, 50, dtype=tf.float32)
				self.actions_color = tf.placeholder(shape=[None], dtype=tf.int32)
				self.actions_onehot_color = tf.one_hot(self.actions_color, 5, dtype=tf.float32)

				self.target_v = tf.placeholder(shape=[None],dtype=tf.float32)
				self.advantages = tf.placeholder(shape=[None],dtype=tf.float32)

				self.responsible_outputs_i_x = tf.reduce_sum(self.policy_i_x * self.actions_onehot_i_x, [1])
				self.responsible_outputs_i_y = tf.reduce_sum(self.policy_i_y * self.actions_onehot_i_y, [1])
				self.responsible_outputs_ii_x = tf.reduce_sum(self.policy_ii_x * self.actions_onehot_ii_x, [1])
				self.responsible_outputs_ii_y = tf.reduce_sum(self.policy_ii_y * self.actions_onehot_ii_y, [1])
				self.responsible_outputs_iii_x = tf.reduce_sum(self.policy_iii_x * self.actions_onehot_iii_x, [1])
				self.responsible_outputs_iii_y = tf.reduce_sum(self.policy_iii_y * self.actions_onehot_iii_y, [1])
				self.responsible_outputs_color = tf.reduce_sum(self.policy_color * self.actions_onehot_color, [1])

				# Loss functions
				self.value_loss = 0.5 * tf.reduce_sum(tf.square(self.target_v - tf.reshape(self.value,[-1])))

				self.entropy = - tf.reduce_sum(self.policy_i_x * tf.log(tf.clip_by_value(self.policy_i_x, 1e-20, 1.0))) # avoid NaN with clipping when value in policy becomes zero
				self.entropy += - tf.reduce_sum(self.policy_i_y * tf.log(tf.clip_by_value(self.policy_i_y, 1e-20, 1.0))) # avoid NaN with clipping when value in policy becomes zero
				self.entropy += - tf.reduce_sum(self.policy_ii_x * tf.log(tf.clip_by_value(self.policy_ii_x, 1e-20, 1.0))) # avoid NaN with clipping when value in policy becomes zero
				self.entropy += - tf.reduce_sum(self.policy_ii_y * tf.log(tf.clip_by_value(self.policy_ii_y, 1e-20, 1.0))) # avoid NaN with clipping when value in policy becomes zero
				self.entropy += - tf.reduce_sum(self.policy_iii_x * tf.log(tf.clip_by_value(self.policy_iii_x, 1e-20, 1.0))) # avoid NaN with clipping when value in policy becomes zero
				self.entropy += - tf.reduce_sum(self.policy_iii_y * tf.log(tf.clip_by_value(self.policy_iii_y, 1e-20, 1.0))) # avoid NaN with clipping when value in policy becomes zero
				self.entropy += - tf.reduce_sum(self.policy_color * tf.log(tf.clip_by_value(self.policy_color, 1e-20, 1.0))) # avoid NaN with clipping when value in policy becomes zero

				self.policy_loss = - tf.reduce_sum(tf.log(tf.clip_by_value(self.responsible_outputs_i_x, 1e-20, 1.0)) * self.advantages)
				self.policy_loss += - tf.reduce_sum(tf.log(tf.clip_by_value(self.responsible_outputs_i_y, 1e-20, 1.0)) * self.advantages)
				self.policy_loss += - tf.reduce_sum(tf.log(tf.clip_by_value(self.responsible_outputs_ii_x, 1e-20, 1.0)) * self.advantages)
				self.policy_loss += - tf.reduce_sum(tf.log(tf.clip_by_value(self.responsible_outputs_ii_y, 1e-20, 1.0)) * self.advantages)
				self.policy_loss += - tf.reduce_sum(tf.log(tf.clip_by_value(self.responsible_outputs_iii_x, 1e-20, 1.0)) * self.advantages)
				self.policy_loss += - tf.reduce_sum(tf.log(tf.clip_by_value(self.responsible_outputs_iii_y, 1e-20, 1.0)) * self.advantages)
				self.policy_loss += - tf.reduce_sum(tf.log(tf.clip_by_value(self.responsible_outputs_color, 1e-20, 1.0)) * self.advantages)

				self.loss = 0.5 * self.value_loss + self.policy_loss - self.entropy * 0.01

				# Get gradients from local network using local losses
				local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
				# self.gradients - gradients of loss wrt local_vars
				self.gradients = tf.gradients(self.loss,local_vars)
				self.var_norms = tf.global_norm(local_vars)
				grads,self.grad_norms = tf.clip_by_global_norm(self.gradients,40.0)

				# Apply local gradients to global network
				global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
				self.apply_grads = trainer.apply_gradients(zip(grads,global_vars))

## WORKER AGENT

class Worker():
	def __init__(self,name,trainer,model_path,global_episodes):
		self.name = "worker_" + str(name)
		self.number = name
		self.model_path = model_path
		self.trainer = trainer
		self.global_episodes = global_episodes
		self.increment = self.global_episodes.assign_add(1)
		self.episode_rewards = []
		self.episode_lengths = []
		self.episode_mean_values = []
		self.summary_writer = tf.summary.FileWriter("train_"+str(self.number))

		# Create the local copy of the network and the tensorflow op to copy global paramters to local network
		self.local_AC = AC_Network(self.name,trainer)
		self.update_local_ops = update_target_graph('global',self.name)
		
		print('Initializing environment #{}...'.format(self.number))
		global _resnet
		self.env = SvgEnv('env_{}'.format(name), _resnet)
		
	def train(self,rollout,sess,gamma,bootstrap_value):
		rollout = np.array(rollout)
		obs = rollout[:,0]
		actions_i_x = rollout[:,1]
		actions_i_y = rollout[:,2]
		actions_ii_x = rollout[:,3]
		actions_ii_y = rollout[:,4]
		actions_iii_x = rollout[:,5]
		actions_iii_y = rollout[:,6]
		actions_color = rollout[:,7]
		rewards = rollout[:,8]
		next_obs = rollout[:,9]
		values = rollout[:,11]
		
		# Here we take the rewards and values from the rollout, and use them to calculate the advantage and discounted returns
		# The advantage function uses generalized advantage estimation from [2]
		self.rewards_plus = np.asarray(rewards.tolist() + [bootstrap_value])
		discounted_rewards = discount(self.rewards_plus,gamma)[:-1]
		self.value_plus = np.asarray(values.tolist() + [bootstrap_value])
		advantages = rewards + gamma * self.value_plus[1:] - self.value_plus[:-1]
		advantages = discount(advantages,gamma)

		# Update the global network using gradients from loss
		# Generate network statistics to periodically save
		feed_dict = {self.local_AC.target_v:discounted_rewards,
			self.local_AC.inputs:np.stack(obs).reshape(-1,200,200,3),
			self.local_AC.actions_i_x:actions_i_x,
			self.local_AC.actions_i_y:actions_i_y,
			self.local_AC.actions_ii_x:actions_ii_x,
			self.local_AC.actions_ii_y:actions_ii_y,
			self.local_AC.actions_iii_x:actions_iii_x,
			self.local_AC.actions_iii_y:actions_iii_y,
			self.local_AC.actions_color:actions_color,
			self.local_AC.advantages:advantages}
		
		v_l,p_l,e_l,g_n,v_n, _ = sess.run([self.local_AC.value_loss,
			self.local_AC.policy_loss,
			self.local_AC.entropy,
			self.local_AC.grad_norms,
			self.local_AC.var_norms,
			self.local_AC.apply_grads],
			feed_dict=feed_dict)
		return v_l / len(rollout),p_l / len(rollout),e_l / len(rollout), g_n,v_n
		
	def work(self,max_episode_length,gamma,sess,coord,saver):
		episode_count = sess.run(self.global_episodes)
		total_steps = 0
		print ("Starting worker " + str(self.number))
		with sess.as_default(), sess.graph.as_default():
			while not coord.should_stop():
				# Download copy of parameters from global network
				sess.run(self.update_local_ops)
				
				episode_buffer = []
				episode_values = []
				episode_reward = 0
				episode_step_count = 0
				episode_end = False
				
				# Start new episode
				obs_stack, reward, episode_end = self.env.reset()
				s_obs = obs_stack
				
				while not episode_end:

					# Take an action using distributions from policy networks' outputs
					i_x_dist, i_y_dist, ii_x_dist, ii_y_dist, iii_x_dist, iii_y_dist, color_dist, v = sess.run(
						[
							self.local_AC.policy_i_x,
							self.local_AC.policy_i_y,
							self.local_AC.policy_ii_x,
							self.local_AC.policy_ii_y,
							self.local_AC.policy_iii_x,
							self.local_AC.policy_iii_y,
							self.local_AC.policy_color,
							self.local_AC.value,
						],
						feed_dict={self.local_AC.inputs: obs_stack})

					i_x = sample_dist(i_x_dist)
					i_y = sample_dist(i_y_dist)
					ii_x = sample_dist(ii_x_dist)
					ii_y = sample_dist(ii_y_dist)
					iii_x = sample_dist(iii_x_dist)
					iii_y = sample_dist(iii_y_dist)
					color = sample_dist(color_dist)

					if color == 0:
						rgb = '#000000'
					elif color == 1:
						rgb = '#FF0000'
					elif color == 2:
						rgb = '#00FF00'
					elif color == 3:
						rgb = '#0000FF'
					elif color == 4:
						rgb = '#FFFF00'

					a = {
						'color': rgb,
						'vertices': [
							[i_x * 4, i_y * 4],
							[ii_x * 4, ii_y * 4],
							[iii_x * 4, iii_y * 4],
						],
					}
					
					obs_stack, r, episode_end = self.env.step(action=a)

					if not episode_end:
						s1_obs = obs_stack
					else:
						s1_obs = s_obs
					
					# Append latest state to buffer
					episode_buffer.append([s_obs, i_x, i_y, ii_x, ii_y, iii_x, iii_y, color, r, s1_obs, episode_end, v[0,0]])
					episode_values.append(v[0,0])

					episode_reward += r
					s_obs = s1_obs
					total_steps += 1
					episode_step_count += 1
					
					# If the episode hasn't ended, but the experience buffer is full, then we make an update step using that experience rollout
					if len(episode_buffer) == 30 and not episode_end and episode_step_count != max_episode_length - 1:
						# Since we don't know what the true final return is, we "bootstrap" from our current value estimation
						v1 = sess.run(self.local_AC.value, 
							feed_dict={self.local_AC.inputs: obs_stack})[0,0]
						v_l,p_l,e_l,g_n,v_n = self.train(episode_buffer,sess,gamma,v1)
						episode_buffer = []
						sess.run(self.update_local_ops)
					if episode_end:
						if self.env.chosen_class == -1:
							global _chosen_class
							if _chosen_class == -1:
								_chosen_class = np.argmax(self.env.predictions)
							self.env.chosen_class = _chosen_class
						break

				self.episode_rewards.append(episode_reward)
				self.episode_lengths.append(episode_step_count)
				self.episode_mean_values.append(np.mean(episode_values))
				episode_count += 1

				global _max_score, _running_avg_score, _episodes, _steps
				if _max_score < episode_reward:
					_max_score = episode_reward
				_running_avg_score = (2.0 / 101) * (episode_reward - _running_avg_score) + _running_avg_score
				_episodes[self.number] = episode_count
				_steps[self.number] = total_steps

				print("{}-{} Step #{} Episode #{} Reward: {}".format(self.name, self.env.chosen_class, total_steps, episode_count, episode_reward))
				print("Total Steps: {}\tTotal Episodes: {}\tMax Score: {}\tAvg Score: {}".format(np.sum(_steps), np.sum(_episodes), _max_score, _running_avg_score))

				# Update the network using the episode buffer at the end of the episode
				if len(episode_buffer) != 0:
					v_l,p_l,e_l,g_n,v_n = self.train(episode_buffer,sess,gamma,0.0)

				if episode_count % 5 == 0 and episode_count != 0:
					if episode_count % 250 == 0 and self.name == 'worker_0':
						saver.save(sess,self.model_path+'/model-'+str(episode_count)+'.cptk')
						print ("Saved Model")

					mean_reward = np.mean(self.episode_rewards[-5:])
					mean_length = np.mean(self.episode_lengths[-5:])
					mean_value = np.mean(self.episode_mean_values[-5:])
					summary = tf.Summary()
					summary.value.add(tag='Perf/Reward', simple_value=float(mean_reward))
					summary.value.add(tag='Perf/Length', simple_value=float(mean_length))
					summary.value.add(tag='Perf/Value', simple_value=float(mean_value))
					summary.value.add(tag='Losses/Value Loss', simple_value=float(v_l))
					summary.value.add(tag='Losses/Policy Loss', simple_value=float(p_l))
					summary.value.add(tag='Losses/Entropy', simple_value=float(e_l))
					summary.value.add(tag='Losses/Grad Norm', simple_value=float(g_n))
					summary.value.add(tag='Losses/Var Norm', simple_value=float(v_n))
					self.summary_writer.add_summary(summary, episode_count)

					self.summary_writer.flush()
				if self.name == 'worker_0':
					sess.run(self.increment)

def main():
	max_episode_length = 300
	gamma = .99 # Discount rate for advantage estimation and reward discounting
	load_model = False
	model_path = './model'

	tf.reset_default_graph()

	tf.gfile.MakeDirs(model_path)

	with tf.device("/cpu:0"): 
		global_episodes = tf.Variable(0,dtype=tf.int32,name='global_episodes',trainable=False)
		trainer = tf.train.AdamOptimizer(learning_rate=1e-4)
		master_network = AC_Network('global',None) # Generate global network
		#num_workers = multiprocessing.cpu_count() # Set workers to number of available CPU threads
		num_workers = psutil.cpu_count() # Set workers to number of available CPU threads
		num_workers = 8
		global _chosen_class, _max_score, _running_avg_score, _steps, _episodes, _resnet
		_chosen_class = -1
		_max_score = 0
		_running_avg_score = 0
		_steps = np.zeros(num_workers)
		_episodes = np.zeros(num_workers)
		_resnet = ResNet50(weights='imagenet')
		workers = []
		# Create worker classes
		for i in range(num_workers):
			workers.append(Worker(i,trainer,model_path,global_episodes))
		saver = tf.train.Saver(max_to_keep=5)

	with tf.Session() as sess:
		coord = tf.train.Coordinator()
		if load_model == True:
			print ('Loading Model...')
			ckpt = tf.train.get_checkpoint_state(model_path)
			saver.restore(sess,ckpt.model_checkpoint_path)
		else:
			sess.run(tf.global_variables_initializer())
			
		# This is where the asynchronous magic happens
		# Start the "work" process for each worker in a separate thread
		worker_threads = []
		for worker in workers:
			worker_work = lambda: worker.work(max_episode_length,gamma,sess,coord,saver)
			t = threading.Thread(target=(worker_work))
			t.start()
			sleep(0.5)
			worker_threads.append(t)
		coord.join(worker_threads)


if __name__ == '__main__':
	main()
