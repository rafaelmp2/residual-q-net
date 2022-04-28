import torch.nn as nn
import torch
import torch.nn.functional as F

class QMixNet(nn.Module):
	def __init__(self, args):
		super(QMixNet, self).__init__()
		self.args = args

		# linear layers cant process multi dimensional arrays normally
		# as hyper needs to be a n-D matrix, first shuold output the vector with size row*column and then convert it to a matrix
		# n_agents is the input dimension using hyper_w1 as the parameter and qmix_hidden_dim is the number of the net hidden layer params 
		# so, the matrix of (experience number, n_agents * qmix_hidden_dim) is obtained through hyper_w1

		# NOTE: linear weigth refers to the learnable weigths of the module of shape defined and linear bias refers
		# to the learnable bias of the module of the shape defined 
		
		if args.two_hyper_layers:
			self.hyper_w1 = nn.Sequential(nn.Linear(args.state_shape, args.hyper_hidden_dim),
											nn.ReLU(),
											nn.Linear(args.hyper_hidden_dim, args.n_agents * args.qmix_hidden_dim))

			# after this layer get matrix of (exp numb, 1)
			self.hyper_w2 = nn.Sequential(nn.Linear(args.state_shape, args.hyper_hidden_dim),
											nn.ReLU(),
											nn.Linear(args.hyper_hidden_dim, args.n_agents * args.qmix_hidden_dim))

		else:
			self.hyper_w1 = nn.Linear(args.state_shape, args.n_agents * args.qmix_hidden_dim)

			# same as before
			self.hyper_w2 = nn.Linear(args.state_shape, args.qmix_hidden_dim * 1)

		# hyper_w1 requires a matrix of the same dimensions NOTE: linear weigth, linear bias
		self.hyper_b1 = nn.Linear(args.state_shape, args.qmix_hidden_dim)
		# same for hyper_w2
		self.hyper_b2 = nn.Sequential(nn.Linear(args.state_shape, args.qmix_hidden_dim),
									nn.ReLU(),
									nn.Linear(args.qmix_hidden_dim, 1)
									)

	# the states have shape (episode_num, max_episode_len, state_shape)
	# q_values should have shape (episode_num, max_episode_len, n_agents)
	def forward(self, q_values, states):
		episode_num = q_values.size(0)
		q_values = q_values.view(-1, 1, self.args.n_agents)  # return same tensor with defined shape
		states = states.reshape(-1, self.args.state_shape)

		w1 = torch.abs(self.hyper_w1(states))
		b1 = self.hyper_b1(states)

		w1 = w1.view(-1, self.args.n_agents, self.args.qmix_hidden_dim)
		b1 = b1.view(-1, 1, self.args.qmix_hidden_dim)

		hidden = F.elu(torch.bmm(q_values, w1) + b1)  # bmm batch matrix matrix product of matrices: q_values * w1, where q_values is (b*n*m) tensor and w1 (b*m*p)

		w2 = torch.abs(self.hyper_w2(states))
		b2 = self.hyper_b2(states)

		w2 = w2.view(-1, self.args.qmix_hidden_dim, 1)
		b2 = b2.view(-1, 1, 1)

		q_total = torch.bmm(hidden, w2) + b2
		q_total = q_total.view(episode_num, -1, 1)

		return q_total


