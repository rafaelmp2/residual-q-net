from algos.vdn import VDN
from algos.qmix import QMix
from algos.coma import COMA
from algos.qtran_base import QtranBase
from algos.rqn import RQN
import torch
import numpy as np
from torch.distributions import Categorical

# no communication agents
class Agents:
	def __init__(self, args):
		self.n_actions = args.n_actions
		self.n_agents = args.n_agents
		self.state_shape = args.state_shape
		self.obs_shape = args.obs_shape

		if args.alg == 'vdn':
			self.policy = VDN(args)
			print("VDN policy initialized")
		elif args.alg == 'qmix':
			self.policy = QMix(args)
			print("QMix policy initialized")
		elif args.alg == 'coma':
			self.policy = COMA(args)
			print("COMA policy initialized")
		elif args.alg == 'qtran_base':
			self.policy = QtranBase(args)
			print("QTRANBASE policy initialized")
		elif args.alg == 'rqn':
			self.policy = RQN(args)
			print('RQN policy initialized')
		else:
			raise Exception("No such algorithm!")

		self.args = args



	def choose_action(self, obs, last_action, agent_num, avail_actions, epsilon, maven_z=None, evaluate=False):
		
		inputs = obs.copy()
		avail_actions_ind = np.nonzero(avail_actions)[0]  # index of actions which can be choose

		# transform agent_num to onehot vector
		agent_id = np.zeros(self.n_agents)
		agent_id[agent_num] = 1.

		if self.args.last_action:
		    inputs = np.hstack((inputs, last_action))  # concatenates arrays column wise (horizontally)
		if self.args.reuse_network:
		    inputs = np.hstack((inputs, agent_id))
		hidden_state = self.policy.eval_hidden[:, agent_num, :]

		# transform the shape of inputs from (42,) to (1,42)
		inputs = torch.tensor(inputs, dtype=torch.float32).unsqueeze(0)
		avail_actions = torch.tensor(avail_actions, dtype=torch.float32).unsqueeze(0)

		# cuda
		if self.args.cuda:
			inputs = inputs.cuda()
			hidden_state = hidden_state.cuda()
			

		q_value, self.policy.eval_hidden[:, agent_num, :] = self.policy.eval_rnn(inputs, hidden_state)

		# if the algo is coma, choose the actions from softmax
		if self.args.alg == 'coma':
			action = self._choose_action_from_softmax(q_value.cpu(), avail_actions, epsilon, evaluate)
		else:
			q_value[avail_actions == 0.0] = - float("inf")
			# epsilon greedy
			if np.random.uniform() < epsilon:
				action = np.random.choice(avail_actions_ind)  # picks an action from the available actions array
			else:
				action = torch.argmax(q_value)

		return action


	def _get_max_episode_len(self, batch):
		terminated = batch['terminated']
		episode_num = terminated.shape[0]  # number of episode batches inside this batch
		max_episode_len = 0
		for episode_idx in range(episode_num):
		    for transition_idx in range(self.args.episode_limit):
		        if terminated[episode_idx, transition_idx, 0] == 1:  
		            if transition_idx + 1 >= max_episode_len:
		                max_episode_len = transition_idx + 1
		            break
		return max_episode_len

	def train(self, batch, train_step, epsilon=None):  

	# different episode has different length, so we need to get max length of the batch
		max_episode_len = self._get_max_episode_len(batch)  # inside batch there are several episode batches; as they may have different sizes, gets the bigger
		for key in batch.keys():
		    batch[key] = batch[key][:, :max_episode_len] 
		self.policy.learn(batch, max_episode_len, train_step, epsilon)

		# savind model
		if train_step > 0 and train_step % self.args.save_cycle == 0:
		    self.policy.save_model(train_step)


	def _choose_action_from_softmax(self, inputs, avail_actions, epsilon, evaluate=False):
		# inputs refers to q_value of all actions
		action_num = avail_actions.sum(dim=1, keepdim=True).float().repeat(1, avail_actions.shape[-1])  # sum of avail_actions
		# converts output of actor network into a prob dist with softmax
		prob = torch.nn.functional.softmax(inputs, dim=-1)

		# noise of epsilon
		prob = ((1 - epsilon) * prob + torch.ones_like(prob) * epsilon / action_num)
		prob[avail_actions == 0] = 0.0  # unavailable actions get 0 prob

		# note that after setting the unavaible actions prob to 0, the sum in prob is not 1, but no need to regularize because torch.distributions.categorical will be regularized
		# categorical is not used during training so the probability of the action performed during training needs to be regularized again

		if epsilon == 0 and evaluate:
			action = torch.argmax(prob)
		else:
			action = Categorical(prob).sample().long()

		return action



# communication agents
class CommAgents:
	def __init__(self, args):
		self.n_actions = args.n_actions
		self.n_agents = args.n_agents
		self.state_shape = args.state_shape
		self.obs_shape = args.obs_shape

		# for now let coma be the only available algo to support the communication algorithm
		if args.alg.find('coma') > -1:
			self.policy = COMA(args)
			print("COMA policy Communication agent initialized")
		elif args.alg.find('vdn') > -1:
			self.policy = VDN(args)
			print("VDN policy Communication agent initialized")
		elif args.alg.find('qmix') > -1:
			self.policy = QMix(args)
			print("QMIX policy Communication agent initialized")
		else:
			raise Exception("No such algorithm!")

		self.args = args


	# gets the probabilty according the weights and then pick the action according epsilon
	def choose_action(self, weights, avail_actions, epsilon, evaluate=None):
		weights = weights.unsqueeze(0)
		avail_actions = torch.tensor(avail_actions, dtype=torch.float32).unsqueeze(0)
		action_num = avail_actions.sum(dim=1, keepdim=True).float().repeat(1, avail_actions.shape[-1])  # number of actions that can be selected

		# converts the output of the actor net into a prob dist through softmax
		prob = torch.nn.functional.softmax(weights, dim=-1)

		# add noise to the prob dist during training
		prob = ((1 - epsilon) * prob + torch.ones_like(prob) * epsilon / action_num)
		prob[avail_actions == 0] = 0.0  # change the porb of unavailable actions to 0

		'''
		because we have changed probs to 0, now the sum of all the probs is not 1; however, regularization is not needed
		because torch categorical will be regularized; this categorical dist is not used during training, thus the prob of the 
		action taken during training needs to be regularized
		'''

		if epsilon == 0 and evaluate:
			# selects bigger when evaluating
			action = torch.argmax(prob)
		else:
			# else sample from the prob dist
			action = Categorical(prob).sample().long()

		return action


	def get_action_weights(self, obs, last_action):
		obs = torch.tensor(obs, dtype=torch.float32)
		last_action = torch.tensor(last_action, dtype=torch.float32)
		inputs = list()
		inputs.append(obs)

		# add last action and agent num to obs
		if self.args.last_action:
			inputs.append(last_action)
		
		if self.args.reuse_network:
			inputs.append(torch.eye(self.args.n_agents))

		inputs = torch.cat([x for x in inputs], dim=1)

		weights, self.policy.eval_hidden = self.policy.eval_rnn(inputs, self.policy.eval_hidden)
		weights = weights.reshape(self.args.n_agents, self.args.n_actions)

		return weights.cpu()


	def _get_max_episode_len(self, batch):
		terminated = batch['terminated']
		episode_num = terminated.shape[0]  # number of episode batches inside this batch
		max_episode_len = 0
		for episode_idx in range(episode_num):
		    for transition_idx in range(self.args.episode_limit):
		        if terminated[episode_idx, transition_idx, 0] == 1: 
		            if transition_idx + 1 >= max_episode_len:
		                max_episode_len = transition_idx + 1
		            break
		return max_episode_len


	def train(self, batch, train_step, epsilon=None): 

	# different episode has different length, so we need to get max length of the batch
		max_episode_len = self._get_max_episode_len(batch)  # inside batch there are several episode batches; as they may have different sizes, gets the bigger
		for key in batch.keys():
		    batch[key] = batch[key][:, :max_episode_len]  
		self.policy.learn(batch, max_episode_len, train_step, epsilon)

		# savind model
		if train_step > 0 and train_step % self.args.save_cycle == 0:
		    self.policy.save_model(train_step)


		    




