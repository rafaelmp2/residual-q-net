import numpy as np
import torch
from torch.distributions import one_hot_categorical
import time
import numpy as np

class RolloutWorker:
	def __init__(self, env, agents, args):
		self.env = env
		self.agents = agents
		self.episode_limit = args.episode_limit
		self.n_actions = args.n_actions
		self.n_agents = args.n_agents
		self.state_shape = args.state_shape
		self.obs_shape = args.obs_shape
		self.args = args

		self.epsilon = args.epsilon
		self.anneal_epsilon = args.anneal_epsilon
		self.min_epsilon = args.min_epsilon

		print("RolloutWorker initialized")

	def generate_episode(self, episode_num=None, evaluate=False, epoch_num=None, eval_epoch=None): 
		# lists to store whole episode info
		obs_ep, actions_ep, reward_ep, state_ep, avail_actions_ep, actions_onehot_ep, terminate, padded = [], [], [], [], [], [], [], []
		self.env.reset()
		terminated = [False] * self.n_agents  # because for this env terminated returns a bool list with a value for each agent to say when each one reached the goal
		step = 0
		episode_reward = 0  
		last_action = np.zeros((self.args.n_agents, self.args.n_actions))  # matrix 2D with list of actions for each agent 
		self.agents.policy.init_hidden(1)

		won = False  # check if episode resulted in win state

		epsilon = 0 if evaluate else self.epsilon

		while not all(terminated):  # all because for this envs terminated is returned as a list that says if the agent reached is goal
		    obs = self.env.get_agent_obs()  # gets 2d array with the individual observations for each agent ate each 2nd index
		    state = np.array(obs).flatten()  # state is 1d array with the features of the observations of each agent in a row
		    actions, avail_actions, actions_onehot = [], [], []  # for single state info
		    for agent_id in range(self.n_agents):
		    	# every actions are available for these envs being used from ma_gym, so value 1 for every action of each agent
		    	# 1 at index i means action i is avail
		    	avail_action = [1] * self.n_actions  # avail actions for agent_i 

		    	#choose an action for agent_i; decentralized exec
		    	action = self.agents.choose_action(obs[agent_id], last_action[agent_id], agent_id, avail_action, epsilon, evaluate)

		    	# generate a vector of 0s and 1s of the corresponding action; actions chosen gets 1 and rest is 0
		    	action_onehot = np.zeros(self.args.n_actions)
		    	action_onehot[action] = 1

		    	# adds action info to corresponding lists
		    	actions.append(action)
		    	actions_onehot.append(action_onehot)
		    	avail_actions.append(avail_action)
		    	last_action[agent_id] = action_onehot

		    _, reward, terminated, _ = self.env.step(actions)

		    obs_ep.append(obs)
		    state_ep.append(state)

		    # need to reshape the list of actions into a vector with shape (n_agents, 1) to store in the buffer
		    actions_ep.append(np.reshape(actions, [self.n_agents, 1]))
		    actions_onehot_ep.append(actions_onehot)
		    avail_actions_ep.append(avail_actions)
		    reward_ep.append([sum(reward)])  # reward returned for this env is a list with a reward for each agent, so sum
		    terminate.append([all(terminated)])  # terminated for this env is a bool list which says if each agent reached the goal or not
		    padded.append([0.])
		    episode_reward += sum(reward)
		    step += 1
		    if self.args.epsilon_anneal_scale == 'step':
		    	epsilon = epsilon - self.anneal_epsilon if epsilon > self.min_epsilon else epsilon


		# handle last obs
		obs = self.env.get_agent_obs() 
		state = np.array(obs).flatten()
		obs_ep.append(obs)
		state_ep.append(state)
		o_next = obs_ep[1:]
		s_next = state_ep[1:]
		obs_ep = obs_ep[:-1]
		state_ep = state_ep[:-1]

		# last obs needs to calculate the avail_actions sep, then calculate target_q
		avail_actions = [[1] * self.n_actions for _ in range(self.n_agents)]  # same as above, as in this env everything is avail
		
		avail_actions_ep.append(avail_actions)
		avail_actions_next = avail_actions_ep[1:]
		avail_actions_ep = avail_actions_ep[:-1]


		# the generated episode must be self.episode_limit long, so if it terminated before this size it has to be filled, everything is filled with 1's
		for i in range(step, self.episode_limit):
		    obs_ep.append(np.zeros((self.n_agents, self.obs_shape)))
		    actions_ep.append(np.zeros([self.n_agents, 1]))
		    state_ep.append(np.zeros(self.state_shape))
		    reward_ep.append([0.])
		    o_next.append(np.zeros((self.n_agents, self.obs_shape)))
		    s_next.append(np.zeros(self.state_shape))
		    actions_onehot_ep.append(np.zeros((self.n_agents, self.n_actions)))
		    avail_actions_ep.append(np.zeros((self.n_agents, self.n_actions)))
		    avail_actions_next.append(np.zeros((self.n_agents, self.n_actions)))
		    padded.append([1.])
		    terminate.append([1.])


		#create episode batch to add to buffer
		episode = dict(obs=obs_ep.copy(),
		               state=state_ep.copy(),
		               actions=actions_ep.copy(),
		               reward=reward_ep.copy(),
		               avail_actions=avail_actions_ep.copy(),
		               obs_next=o_next.copy(),
		               state_next=s_next.copy(),
		               avail_actions_next=avail_actions_next.copy(),
		               actions_onehot=actions_onehot_ep.copy(),
		               padded=padded.copy(),
		               terminated=terminate.copy()
		               )
		
		# add extra episode dimension to the dict values, TODO not yet very sure why
		for key in episode.keys():
		    episode[key] = np.array([episode[key]])
		if not evaluate:
			self.epsilon = epsilon
		
		return episode, episode_reward, won






## worker for communication
class CommRolloutWorker:
	def __init__(self, env, agents, args):
		self.env = env
		self.agents = agents
		self.episode_limit = args.episode_limit
		self.n_actions = args.n_actions
		self.n_agents = args.n_agents
		self.state_shape = args.state_shape
		self.obs_shape = args.obs_shape
		self.args = args

		self.epsilon = args.epsilon
		self.anneal_epsilon = args.anneal_epsilon
		self.min_epsilon = args.min_epsilon

		print("CommRolloutWorker initialized")


	def generate_episode(self, episode_reward=None, evaluate=False):
		# lists to store whole episode info
		obs_ep, actions_ep, reward_ep, state_ep, avail_actions_ep, actions_onehot_ep, terminate, padded = [], [], [], [], [], [], [], []
		self.env.reset()
		terminated = [False] * self.n_agents  # because for this env terminated returns a bool list with a value for each agent to say when each one reached the goal
		step = 0
		episode_reward = 0  
		last_action = np.zeros((self.args.n_agents, self.args.n_actions))  # matrix 2D with list of actions for each agent 
		self.agents.policy.init_hidden(1)

		won = False  # check if episode resulted in win state

		epsilon = 0 if evaluate else self.epsilon

		while not all(terminated):  # all because for this envs terminated is returned as a list that says if the agent reached is goal
		    obs = self.env.get_agent_obs()  # gets 2d array with the individual observations for each agent at each 2nd index
		    state = np.array(obs).flatten()  # state is 1d array with the features of the observations of each agent in a row
		    actions, avail_actions, actions_onehot = [], [], []  # for single state info

		    weights = self.agents.get_action_weights(np.array(obs), last_action)

		    for agent_id in range(self.n_agents):
		    	# every actions are available for these envs being used from ma_gym, so value 1 for every action of each agent
		    	# 1 at index i means action i is avail
		    	avail_action = [1] * self.n_actions  # avail actions for agent_i 

		    	action = self.agents.choose_action(weights[agent_id], avail_action, epsilon, evaluate)

		    	# generate a vector of 0s and 1s of the corresponding action; actions chosen gets 1 and rest is 0
		    	action_onehot = np.zeros(self.args.n_actions)
		    	action_onehot[action] = 1

		    	# adds action info to corresponding lists
		    	actions.append(action)
		    	actions_onehot.append(action_onehot)
		    	avail_actions.append(avail_action)
		    	last_action[agent_id] = action_onehot

		    _, reward, terminated, _ = self.env.step(actions)

		    obs_ep.append(obs)
		    state_ep.append(state)

		    # need to reshape the list of actions into a vector with shape (n_agents, 1) to store in the buffer
		    actions_ep.append(np.reshape(actions, [self.n_agents, 1]))
		    actions_onehot_ep.append(actions_onehot)
		    avail_actions_ep.append(avail_actions)
		    reward_ep.append([sum(reward)])  # reward returned for this env is a list with a reward for each agent, so sum
		    terminate.append([all(terminated)])  # terminated for this env is a bool list which says if each agent reached the goal or not
		    padded.append([0.])
		    episode_reward += sum(reward)
		    step += 1
		    if self.args.epsilon_anneal_scale == 'step':
		    	epsilon = epsilon - self.anneal_epsilon if epsilon > self.min_epsilon else epsilon

		# handle last obs
		obs = self.env.get_agent_obs() 
		state = np.array(obs).flatten()  
		obs_ep.append(obs)
		state_ep.append(state)
		o_next = obs_ep[1:]
		s_next = state_ep[1:]
		obs_ep = obs_ep[:-1]
		state_ep = state_ep[:-1]

		# last obs needs to calculate the avail_actions sep, then calculate target_q
		avail_actions = [[1] * self.n_actions for _ in range(self.n_agents)]  # same as above, as in this env everything is avail
		
		avail_actions_ep.append(avail_actions)
		avail_actions_next = avail_actions_ep[1:]
		avail_actions_ep = avail_actions_ep[:-1]


		# the generated episode must be self.episode_limit long, so if it terminated before this size it has to be filled, everything is filled with 1's
		for i in range(step, self.episode_limit):
		    obs_ep.append(np.zeros((self.n_agents, self.obs_shape)))
		    actions_ep.append(np.zeros([self.n_agents, 1]))
		    state_ep.append(np.zeros(self.state_shape))
		    reward_ep.append([0.])
		    o_next.append(np.zeros((self.n_agents, self.obs_shape)))
		    s_next.append(np.zeros(self.state_shape))
		    actions_onehot_ep.append(np.zeros((self.n_agents, self.n_actions)))
		    avail_actions_ep.append(np.zeros((self.n_agents, self.n_actions)))
		    avail_actions_next.append(np.zeros((self.n_agents, self.n_actions)))
		    padded.append([1.])
		    terminate.append([1.])


		#create episode batch to add to buffer
		episode = dict(obs=obs_ep.copy(),
		               state=state_ep.copy(),
		               actions=actions_ep.copy(),
		               reward=reward_ep.copy(),
		               avail_actions=avail_actions_ep.copy(),
		               obs_next=o_next.copy(),
		               state_next=s_next.copy(),
		               avail_actions_next=avail_actions_next.copy(),
		               actions_onehot=actions_onehot_ep.copy(),
		               padded=padded.copy(),
		               terminated=terminate.copy()
		               )
		
		# add extra episode dimension to the dict values, TODO not yet very sure why
		for key in episode.keys():
		    episode[key] = np.array([episode[key]])
		if not evaluate:
		    self.epsilon = epsilon
		    
		return episode, episode_reward, won






