import torch
import torch.nn as nn
import torch.nn.functional as F


# Joint action-value network, input state, hidden_state of all agents, actions of all agents, output the corresponding joint Q value
class QtranQBase(nn.Module):
	def __init__(self, args):
		super(QtranQBase, self).__init__()
		self.args = args

		# encodes hidden state and actions of each agent so they can be added to obtain a joint hidden state and action values
		ae_input = self.args.rnn_hidden_dim + self.args.n_actions
		self.hidden_action_encoding = nn.Sequential(nn.Linear(ae_input, ae_input),
                                             nn.ReLU(),
                                             nn.Linear(ae_input, ae_input))

		# enter the sum of state hidden state and actions of all agents after encoding and summing
		q_input = self.args.state_shape + self.args.n_actions + self.args.rnn_hidden_dim
		self.q = nn.Sequential(nn.Linear(q_input, self.args.qtran_hidden_dim),
                               nn.ReLU(),
                               nn.Linear(self.args.qtran_hidden_dim, self.args.qtran_hidden_dim),
                               nn.ReLU(),
                               nn.Linear(self.args.qtran_hidden_dim, 1))

	# because the hidden states of all agents have been calculated before, the joint q value can calculate all the transitions at one time instead of one by one
	# shape (episode_num, max_episode_len, n_agents, n_actions)
	def forward(self, state, hidden_states, actions):
		episode_num, max_episode_len, n_agents, _ = actions.shape
		hidden_actions = torch.cat([hidden_states, actions], dim=-1)
		hidden_actions = hidden_actions.reshape(-1, self.args.rnn_hidden_dim + self.args.n_actions)
		hidden_actions_encoding = self.hidden_action_encoding(hidden_actions)
		hidden_actions_encoding = hidden_actions_encoding.reshape(episode_num * max_episode_len, n_agents, -1)  # change back to n_agents dimension to sum
		hidden_actions_encoding = hidden_actions_encoding.sum(dim=-2)

		inputs = torch.cat([state.reshape(episode_num * max_episode_len, -1), hidden_actions_encoding], dim=-1)
		q = self.q(inputs)
		return q


# input current state and hidden state of all the agents, output V value
class QtranV(nn.Module):
	def __init__(self, args):
		super(QtranV, self).__init__()
		self.args = args

		# hidden state inputs are encoded so they can be summed and outputted an approximate joint hiddent_state
		hidden_input = self.args.rnn_hidden_dim
		self.hidden_encoding = nn.Sequential(nn.Linear(hidden_input, hidden_input),
                                             nn.ReLU(),
                                             nn.Linear(hidden_input, hidden_input))

		# enters state and sum of hidden state after summing
		v_input = self.args.state_shape + self.args.rnn_hidden_dim
		self.v = nn.Sequential(nn.Linear(v_input, self.args.qtran_hidden_dim),
		                       nn.ReLU(),
		                       nn.Linear(self.args.qtran_hidden_dim, self.args.qtran_hidden_dim),
		                       nn.ReLU(),
		                       nn.Linear(self.args.qtran_hidden_dim, 1))


	def forward(self, state, hidden):
		episode_num, max_episode_len, n_agents, _ = hidden.shape
		state = state.reshape(episode_num * max_episode_len, -1)
		hidden_encoding = self.hidden_encoding(hidden.reshape(-1, self.args.rnn_hidden_dim))
		hidden_encoding = hidden_encoding.reshape(episode_num * max_episode_len, n_agents, -1).sum(dim=-2)
		inputs = torch.cat([state, hidden_encoding], dim=-1)
		v = self.v(inputs)
		return v