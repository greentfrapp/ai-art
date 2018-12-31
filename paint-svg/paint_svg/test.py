from spinup import ppo
import tensorflow as tf
import gym
import paint_svg
from paint_svg.algos.ppo.ppo import ppo

env_fn = lambda : gym.make('PaintSvg-v0')

ac_kwargs = dict(hidden_sizes=[64,64], activation=tf.nn.relu)

logger_kwargs = dict(output_dir='test', exp_name='paint_svg')

ppo(env_fn=env_fn, ac_kwargs=ac_kwargs, steps_per_epoch=5000, epochs=250, logger_kwargs=logger_kwargs)

# env = gym.make('PaintSvg-v0')
# env.reset()