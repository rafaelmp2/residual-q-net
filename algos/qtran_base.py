import torch
import torch.nn as nn
import os
from network.base_net import RNN
from network.qtran_net import QtranQBase, QtranV

class QtranBase:
	def __init__(self, args):
		self.n_agents = args.n_agents
		self.n_actions = args.n_actions
		self.state_shape = args.state_shape
		self.obs_shape = args.obs_shape
		self.args = args

		rnn_input_shape = self.obs_shape

		if args.last_action:
			rnn_input_shape += self.n_actions
		if args.reuse_network:
			rnn_input_shape += self.n_agents

		self.eval_rnn = RNN(rnn_input_shape, args)
		self.target_rnn = RNN(rnn_input_shape, args)

		self.eval_joint_q = QtranQBase(args)  # joint action-value network
		self.target_joint_q = QtranQBase(args)

		self.v = QtranV(args)

		self.model_dir = args.model_dir + '/' + args.alg

		# to load a model
		if self.args.load_model:
		    if os.path.exists(self.model_dir + '/rnn_net_params.pkl'):
		        path_rnn = self.model_dir + '/rnn_net_params.pkl'
		        path_joint_q = self.model_dir + '/joint_q_params.pkl'
		        path_v = self.model_dir + '/v_params.pkl'
		        map_location = 'cuda:0' if self.args.cuda else 'cpu'
		        self.eval_rnn.load_state_dict(torch.load(path_rnn, map_location=map_location))
		        self.eval_joint_q.load_state_dict(torch.load(path_joint_q, map_location=map_location))
		        self.v.load_state_dict(torch.load(path_v, map_location=map_location))
		        print('Successfully load the model: {}, {} and {}'.format(path_rnn, path_joint_q, path_v))
		    else:
		        raise Exception("No model!") 


		self.target_rnn.load_state_dict(self.eval_rnn.state_dict())
		self.target_joint_q.load_state_dict(self.eval_joint_q.state_dict())

		self.eval_parameters = list(self.eval_joint_q.parameters()) + \
								list(self.v.parameters()) + list(self.eval_rnn.parameters())

		if args.optimizer == 'RMS':
			self.optimizer = torch.optim.RMSprop(self.eval_parameters, lr=args.lr)

		self.eval_hidden = None
		self.target_hidden = None


	def learn(self, batch, max_episode_len, train_step, epsilon=None):
		episode_num = batch['obs'].shape[0]
		self.init_hidden(episode_num)

		#convert data in batch to tensor
		for key in batch.keys():  
		    if key == 'actions':
		        batch[key] = torch.tensor(batch[key], dtype=torch.long)
		    else:
		        batch[key] = torch.tensor(batch[key], dtype=torch.float32)

		actions, reward, avail_actions, avail_actions_next, terminated = batch['actions'], batch['reward'], \
																		batch['avail_actions'], batch['avail_actions_next'], \
																		batch['terminated']

		mask = (1 - batch["padded"].float()).squeeze(-1)

		individual_q_evals, individual_q_targets, hidden_evals, hidden_targets = self._get_individual_q(batch, max_episode_len)

		# Get the local optimal action of each agent at the current moment and the next moment and its one_hot representation
		individual_q_clone = individual_q_evals.clone()
		individual_q_clone[avail_actions == 0.0] = - 999999
		individual_q_targets[avail_actions_next == 0.0] = - 999999

		opt_onehot_eval = torch.zeros(*individual_q_clone.shape)
		opt_action_eval = individual_q_clone.argmax(dim=3, keepdim=True)
		opt_onehot_eval = opt_onehot_eval.scatter(-1, opt_action_eval[:, :].cpu(), 1)

		opt_onehot_target = torch.zeros(*individual_q_targets.shape)
		opt_action_target = individual_q_targets.argmax(dim=3, keepdim=True)
		opt_onehot_target = opt_onehot_target.scatter(-1, opt_action_target[:, :].cpu(), 1)

		# l_td
		# Calculate joint_q and v
		# The dimensions of joint_q and v are (number of episodes, max_episode_len, 1), and joint_q is also used in the subsequent l_nopt
		joint_q_evals, joint_q_targets, v = self.get_qtran(batch, hidden_evals, hidden_targets, opt_onehot_target)

		# loss
		y_dqn = reward.squeeze(-1) + self.args.gamma * joint_q_targets * (1 - terminated.squeeze(-1))
		td_error = joint_q_evals - y_dqn.detach()
		l_td = ((td_error * mask)**2).sum() / mask.sum()

		# l_opt
		#add the q value to the local optimal action
		q_sum_opt = individual_q_clone.max(dim=-1)[0].sum(dim=-1)

		joint_q_hat_opt, _, _ = self.get_qtran(batch, hidden_evals, hidden_targets, opt_onehot_eval, hat=True)
		opt_error = q_sum_opt -joint_q_hat_opt.detach() + v  # When calculating l_opt, joint_q_hat_opt needs to be fixed
		l_opt = ((opt_error * mask)**2).sum() / mask.sum()

		# l_nopt
		q_individual = torch.gather(individual_q_evals, dim=-1, index=actions).squeeze(-1)
		q_sum_nopt = q_individual.sum(dim=-1)

		nopt_error = q_sum_nopt - joint_q_evals.detach() + v  # When calculating l_nopt, joint_q_evals needs to be fixed
		nopt_error = nopt_error.clamp(max=0)
		l_nopt = ((nopt_error * mask)**2).sum() / mask.sum()


		loss = l_td + self.args.lambda_opt * l_opt + self.args.lambda_nopt * l_nopt
		self.optimizer.zero_grad()
		loss.backward()
		torch.nn.utils.clip_grad_norm_(self.eval_parameters, self.args.grad_norm_clip)
		self.optimizer.step()

		# update target networks
		if train_step > 0 and train_step % self.args.target_update_cycle == 0:
		    self.target_rnn.load_state_dict(self.eval_rnn.state_dict())
		    self.target_joint_q.load_state_dict(self.eval_joint_q.state_dict())




	def _get_individual_q(self, batch, max_episode_len):
		episode_num = batch['obs'].shape[0]
		q_evals, q_targets, hidden_evals, hidden_targets = [], [], [], []

		for transition_idx in range(max_episode_len):
			inputs, inputs_next = self._get_individual_inputs(batch, transition_idx)

			if transition_idx == 0:
				_, self.target_hidden = self.target_rnn(inputs, self.eval_hidden)
			q_eval, self.eval_hidden = self.eval_rnn(inputs, self.eval_hidden)
			q_target, self.target_hidden = self.target_rnn(inputs_next, self.target_hidden)
			hidden_eval, hidden_target = self.eval_hidden.clone(), self.target_hidden.clone()

			q_eval = q_eval.view(episode_num, self.n_agents, -1)
			q_target = q_target.view(episode_num, self.n_agents, -1)
			hidden_eval = hidden_eval.view(episode_num, self.n_agents, -1)
			hidden_target = hidden_target.view(episode_num, self.n_agents, -1)

			q_evals.append(q_eval)
			q_targets.append(q_target)
			hidden_evals.append(hidden_eval)
			hidden_targets.append(hidden_target)

		# The obtained q_eval and q_target are a list, the list contains max_episode_len arrays, the dimension of the array is (number of episodes, n_agents, n_actions)
		#  Convert the list into an array of (number of episodes, max_episode_len, n_agents, n_actions)

		q_evals = torch.stack(q_evals, dim=1)
		q_targets = torch.stack(q_targets, dim=1)
		hidden_evals = torch.stack(hidden_evals, dim=1)
		hidden_targets = torch.stack(hidden_targets, dim=1)

		return q_evals, q_targets, hidden_evals, hidden_targets



	def _get_individual_inputs(self, batch, transition_idx):
		obs, obs_next, actions_onehot = batch['obs'][:, transition_idx], \
										batch['obs_next'][:, transition_idx], batch['actions_onehot'][:]

		episode_num = batch['obs'].shape[0]
		inputs, inputs_next = [], []
		inputs.append(obs)
		inputs_next.append(obs_next)

		if self.args.last_action:
			if transition_idx == 0:
				inputs.append(torch.zeros_like(actions_onehot[:, transition_idx]))
			else:
				inputs.append(actions_onehot[:, transition_idx - 1])
			inputs_next.append(actions_onehot[:, transition_idx])

		if self.args.reuse_network:
			inputs.append(torch.eye(self.args.n_agents).unsqueeze(0).expand(episode_num, -1, -1))
			inputs_next.append(torch.eye(self.args.n_agents).unsqueeze(0).expand(episode_num, -1, -1))

			'''
			TODO: read and refrase
			It is necessary to put three of the obs together, and the data of episode_num episodes 
			and self.args.n_agents agents are combined into 40 (40,96) data.

			Because all agents here share a neural network, each data is brought 
			with its own number, so it is still its own data
			'''

		inputs = torch.cat([x.reshape(episode_num * self.args.n_agents, -1) for x in inputs], dim=1)
		inputs_next = torch.cat([x.reshape(episode_num * self.args.n_agents, -1) for x in inputs_next], dim=1)

		# TODO note from github: Check if inputs_next is equivalent to moving inputs backward
		return inputs, inputs_next


	def get_qtran(self, batch, hidden_evals, hidden_targets, local_opt_actions, hat=False):
		episode_num, max_episode_len, _, _ = hidden_targets.shape
		states = batch['state'][:, :max_episode_len]
		states_next = batch['state_next'][:, :max_episode_len]
		actions_onehot = batch['actions_onehot'][:, :max_episode_len]

		if hat:
			# The dimensions of q_eval, q_target, and v output by the neural network are (episode_num * max_episode_len, 1)
			q_evals = self.eval_joint_q(states, hidden_evals, local_opt_actions)
			q_targets = None
			v = None

			# Change the q_eval dimension back to (episode_num, max_episode_len)
			q_evals = q_evals.view(episode_num, -1, 1).squeeze(-1)
		else:
			q_evals = self.eval_joint_q(states, hidden_evals, actions_onehot)
			q_targets = self.target_joint_q(states_next, hidden_targets, local_opt_actions)
			v = self.v(states, hidden_evals)
			# Change the dimensions of q_eval, q_target, and v back to (episode_num, max_episode_len)

			q_evals = q_evals.view(episode_num, -1, 1).squeeze(-1)
			q_targets = q_targets.view(episode_num, -1, 1).squeeze(-1)
			v = v.view(episode_num, -1, 1).squeeze(-1)

		return q_evals, q_targets, v


	def init_hidden(self, episode_num):
		# initializes eval_hidden and target_hidden for each agent of each episode, as in DQN there is a net and a target net to stabilize learning

		self.eval_hidden = torch.zeros((episode_num, self.n_agents, self.args.rnn_hidden_dim))
		self.target_hidden = torch.zeros((episode_num, self.n_agents, self.args.rnn_hidden_dim))


	def save_model(self, train_step):
		num = str(train_step // self.args.save_cycle)
		if not os.path.exists(self.model_dir):
			os.makedirs(self.model_dir)
		torch.save(self.eval_joint_q.state_dict(), self.model_dir + '/' + num + '_joint_q_params.pkl')
		torch.save(self.eval_rnn.state_dict(),  self.model_dir + '/' + num + '_rnn_net_params.pkl')
		torch.save(self.v.state_dict(),  self.model_dir + '/' + num + '_v_params.pkl')