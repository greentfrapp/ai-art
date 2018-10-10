import tensorflow as tf
import numpy as np


def update_target_graph(from_scope,to_scope):
	from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
	to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)
	op_holder = []
	for from_var,to_var in zip(from_vars,to_vars):
		op_holder.append(to_var.assign(from_var))
	return op_holder

def average_learner_vars(var_buffer):
	res = []
	for variables in zip(*var_buffer):
		res.append(np.mean(variables, axis=0))
	return res

def get_reptile_update(old_weights, new_weights, epsilon=1e-3):
	gradient = [old_weights[i] - learner_var for i, learner_var in enumerate(new_weights)]
	update = [old_weights[i] - epsilon * gradient_var for i, gradient_var in enumerate(gradient)]
	return update

def get_fomaml_update(old_weights, gradient, epsilon=1e-3):
	update = [old_weights[i] - epsilon * gradient_var for i, gradient_var in enumerate(gradient)]
	return update

class VariableState:
	"""
	Manage the state of a set of variables.
	"""
	def __init__(self, session, variables, metatype="Reptile"):
		self._session = session
		self._variables = variables
		self._placeholders = [tf.placeholder(v.dtype.base_dtype, shape=v.get_shape()) for v in variables]
		assigns = [tf.assign(v, p) for v, p in zip(self._variables, self._placeholders)]
		self._assign_op = tf.group(*assigns)
		self.metatype = metatype

	def export_variables(self):
		"""
		Save the current variables.
		"""
		return self._session.run(self._variables)

	def import_variables(self, values):
		"""
		Restore the variables.
		"""
		self._session.run(self._assign_op, feed_dict=dict(zip(self._placeholders, values)))
