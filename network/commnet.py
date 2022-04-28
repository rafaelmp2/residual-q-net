import torch
import torch.nn as nn
import torch.nn.functional as F


# input obs of all agents，output probability distribution of all agents
class CommNet(nn.Module):
	def __init__(self, input_shape, args):
		super(CommNet, self).__init__()
		self.encoding = nn.Linear(input_shape, args.rnn_hidden_dim)
		self.f_obs = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)  # used to remember the previous obs
		self.f_comm = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)  # for communication
		self.decoding = nn.Linear(args.rnn_hidden_dim, args.n_actions)

		self.args = args
		self.input_shape = input_shape



	def forward(self, obs, hidden_state):
		obs_encoding = torch.sigmoid(self.encoding(obs)) # .reshape(-1, self.args.n_agents, self.args.rnn_hidden_dim) ?TODO
		h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)

		# gets h after passing f_obs for the first time
		h_out = self.f_obs(obs_encoding, h_in)

		# communication self.args.k times
		for k in range(self.args.k):
			if k == 0:
				h = h_out
				c = torch.zeros_like(h)
			else:
				# Convert h into n_agents dimension for communication
				h = h.reshape(-1, self.args.n_agents, self.args.rnn_hidden_dim)

				# for each agent add the h of the other agents
				# let the last dimension contain the h of all agents
				c = h.reshape(-1, 1, self.args.n_agents * self.args.rnn_hidden_dim)
				c = c.repeat(1, self.args.n_agents, 1)  # now each agent has h of all agents

				# set each agents won h to 0
				mask = (1 - torch.eye(self.args.n_agents))  # generates 2d diagonal matrix
				mask = mask.view(-1, 1).repeat(1, self.args.rnn_hidden_dim).view(self.args.n_agents, -1)  # (n_agents, n_agents * rnn_hidden_dim))

				c = c * mask.unsqueeze(0)

				# Because the h of all agents is in the last dimension, it cannot be added directly 
				# So expand one dimension first, and then remove after adding

				c = c.reshape(-1, self.args.n_agents, self.args.n_agents, self.args.rnn_hidden_dim)
				c = c.mean(dim=-2)  # (episode_num * max_episode_len, n_agents, rnn_hidden_dim)
				h = h.reshape(-1, self.args.rnn_hidden_dim)
				c = c.reshape(-1, self.args.rnn_hidden_dim)

			h = self.f_comm(c, h)

		# after the communication is over, the weight of all actions of each agent 
		# is calculated, and the probability is calculated when the action is selected in agent.py
		weights = self.decoding(h)

		return weights, h_out


