import torch
import os
from network.base_net import RNN
from network.coma_critic import ComaCritic
from network.commnet import CommNet

class COMA:
	def __init__(self, args):
		self.n_actions = args.n_actions
		self.n_agents = args.n_agents
		self.state_shape = args.state_shape
		self.obs_shape = args.obs_shape
		actor_input_shape = self.obs_shape  # actor net input dimension is the same of the rnn of vdn and qmix
		critic_input_shape = self._get_critic_input_shape()  # dimensions of inpput of critic network

		# input dimension for rnn according to the params
		if args.last_action:
		    actor_input_shape += self.n_actions
		if args.reuse_network:
		    actor_input_shape += self.n_agents

		self.args = args


		if self.args.alg == 'coma':
			# each agent selects the action net to output the probs corresponding the its actions; when using this prob, softmax needs to be recalculated
			self.eval_rnn = RNN(actor_input_shape, args)
			print("COMA alg initialized")
		elif self.args.alg == 'coma+commnet':
			self.eval_rnn = CommNet(actor_input_shape, args)
			print("COMA+COMMNET initialized")
		else:
			raise Exception("No such algorithm!")


		# gets joint q value of all the exe actions of the current agent
		# then uses this q value and the prob of the actor net to get the advantage
		self.eval_critic = ComaCritic(critic_input_shape, self.args)
		self.target_critic = ComaCritic(critic_input_shape, self.args)

		self.model_dir = args.model_dir + '/' + args.alg

		if self.args.load_model:
		    if os.path.exists(self.model_dir + '/rnn_params.pkl'):
		        path_rnn = self.model_dir + '/rnn_params.pkl'
		        path_coma = self.model_dir + '/critic_params.pkl'
		        self.eval_rnn.load_state_dict(torch.load(path_rnn))
		        self.eval_critic.load_state_dict(torch.load(path_coma))
		        print('Successfully load the model: {} and {}'.format(path_rnn, path_coma))
		    else:
		    	raise Exception("No such model!")

		# make params of the target network the same
		self.target_critic.load_state_dict(self.eval_critic.state_dict())

		self.rnn_parameters = list(self.eval_rnn.parameters())
		self.critic_parameters = list(self.eval_critic.parameters())

		if args.optimizer == "RMS":
		    self.critic_optimizer = torch.optim.RMSprop(self.critic_parameters, lr=args.lr_critic)
		    self.rnn_optimizer = torch.optim.RMSprop(self.rnn_parameters, lr=args.lr_actor)


		self.args = args
		self.eval_hidden = None


	def _get_critic_input_shape(self):
		# state
		input_shape = self.state_shape

		#obs
		input_shape += self.obs_shape

		# agent_id
		input_shape += self.n_agents

		# curr and previous (*2) actions of all the agents 
		input_shape += self.n_actions * self.n_agents * 2

		return input_shape


	def learn(self, batch, max_episode_len, train_step, epsilon):
		episode_num = batch['obs'].shape[0]  # gets number of episode batches in batch
		self.init_hidden(episode_num)

		#convert data in batch to tensor
		for key in batch.keys():  
		    if key == 'actions':
		        batch[key] = torch.tensor(batch[key], dtype=torch.long)
		    else:
		        batch[key] = torch.tensor(batch[key], dtype=torch.float32)

		# coma doesnt use relay buffer, so next actions not needed
		actions, reward, avail_actions, terminated = batch['actions'], batch['reward'],  batch['avail_actions'], \
													batch['terminated']

		# used to set the td error of the filled experiments to 0, not to affect learning
		mask = (1 - batch["padded"].float()).repeat(1, 1, self.n_agents)

		# calculate each agents q value based oon experience to follow the new critic net
		# then calculate prob of execution of each action to get the advantage and update the actor
		q_values = self._train_critic(batch, max_episode_len, train_step)  # train critic net and get q value of all actions of each agent
		action_prob = self._get_action_prob(batch, max_episode_len, epsilon)  # prob of all actions of each agent

		q_taken = torch.gather(q_values, dim=3, index=actions).squeeze(3)
		pi_taken = torch.gather(action_prob, dim=3, index=actions).squeeze(3)  # prob of the selected action of each agent
		pi_taken[mask == 0] = 1.0  # becuase we want to take logarithms, for the filled experiences the probs are 0 so let them become 1
		log_pi_taken = torch.log(pi_taken)

		# calculate advantage: calculate baseline to compare the actions of each agent with a default actions
		baseline = (q_values * action_prob).sum(dim=3, keepdim=True).squeeze(3).detach()
		advantage = (q_taken - baseline).detach()
		loss = -((advantage * log_pi_taken) * mask).sum() / mask.sum()
		self.rnn_optimizer.zero_grad()
		loss.backward()
		torch.nn.utils.clip_grad_norm_(self.rnn_parameters, self.args.grad_norm_clip)
		self.rnn_optimizer.step()

	def _get_critic_inputs(self, batch, transition_idx, max_episode_len):
		# get experiences of this transition_idx on the episode
		obs, obs_next, state, state_next = batch['obs'][:, transition_idx], batch['obs_next'][:, transition_idx], \
											batch['state'][:, transition_idx], batch['state_next'][:, transition_idx]

		actions_onehot = batch['actions_onehot'][:, transition_idx]
		if transition_idx != max_episode_len - 1:
			actions_onehot_next = batch['actions_onehot'][:, transition_idx + 1]
		else:
			actions_onehot_next = torch.zeros(*actions_onehot.shape)

		# because all agents have the same state, s and s_next are 2-d, there is no n_agents dimension
		# so, s has to be converted to 3-d
		state = state.unsqueeze(1).expand(-1, self.n_agents, -1)
		state_next = state_next.unsqueeze(1).expand(-1, self.n_agents, -1)
		episode_num = obs.shape[0]

		# coma uses a centralised critic, i.e, it uses the same critic for all agents so the actions of each agent in the last dimension have to be changed to the actions of all the agents
		actions_onehot = actions_onehot.view((episode_num, 1, -1)).repeat(1, self.n_agents, 1)
		actions_onehot_next = actions_onehot_next.view((episode_num, 1, -1)).repeat(1, self.n_agents, 1)

		# if it is the first experience, let the previous action be a vector of zero
		if transition_idx == 0:
			actions_onehot_last = torch.zeros_like(actions_onehot)  # NOTE: zeros_like receives a tensor to make it into a matrix of zeros while zeros creates a matrix of zeros with the defined shape
		else:
			actions_onehot_last = batch['actions_onehot'][:, transition_idx - 1]
			actions_onehot_last = actions_onehot_last.view((episode_num, 1, -1)).repeat(1, self.n_agents, 1)

		inputs, inputs_next = [], []

		# add info to list
		inputs.append(state)
		inputs_next.append(state_next)

		inputs.append(obs)
		inputs_next.append(obs_next)

		# add last action
		inputs.append(actions_onehot_last)
		inputs_next.append(actions_onehot)

		# NOTE: for coma, the input is just the actions of the other agents and not the action of the current agent
		action_mask = (1 - torch.eye(self.n_agents))  # generate 2d diagonal matrix
		action_mask = action_mask.view(-1, 1).repeat(1, self.n_actions).view(self.n_agents, -1)
		inputs.append(actions_onehot * action_mask.unsqueeze(0))
		inputs_next.append(actions_onehot_next * action_mask.unsqueeze(0))

		# becuase of the input 3d data, each dimension represents (episode number, agent number, inputs dimension) 
		inputs.append(torch.eye(self.n_agents).unsqueeze(0).expand(episode_num, -1, -1))
		inputs_next.append(torch.eye(self.n_agents).unsqueeze(0).expand(episode_num, -1, -1))

		#to transform its dimensions from (episode_num, n_agents, inputs) three-dimensional to (episode_num * n_agents, inputs) two-dimensional
		inputs = torch.cat([x.reshape(episode_num * self.n_agents, -1) for x in inputs], dim=1)
		inputs_next = torch.cat([x.reshape(episode_num * self.n_agents, -1) for x in inputs_next], dim=1)

		return inputs, inputs_next


	def _get_q_values(self, batch, max_episode_len):
		episode_num = batch['obs'].shape[0]
		q_evals, q_targets = [], []
		for transition_idx in range(max_episode_len):
			inputs, inputs_next = self._get_critic_inputs(batch, transition_idx, max_episode_len)

			# The neural network inputs (episode_num * n_agents, inputs) two-dimensional data, and obtains (episode_num * n_agents, n_actions) two-dimensional data
			q_eval = self.eval_critic(inputs)
			q_target = self.target_critic(inputs_next)

			# change dimensions of the q value back to (ep_num, n_agents, n_actions)
			q_eval = q_eval.view(episode_num, self.n_agents, -1)
			q_target = q_target.view(episode_num, self.n_agents, -1)
			q_evals.append(q_eval)
			q_targets.append(q_target)

		#The obtained q_evals and q_targets are a list, and the list contains max_episode_len arrays. The dimensions of the array are (episode number, n_agents, n_actions)
		# Convert the list into an array of (episode number, max_episode_len, n_agents, n_actions)
		q_evals = torch.stack(q_evals, dim=1)
		q_targets = torch.stack(q_targets, dim=1)

		return q_evals, q_targets


	def _get_actor_inputs(self, batch, transition_idx):
		# decentralised actor, decentralised execution; actor -> policy, maps states to actions
		# take the experience of the transition_idx on all the episodes
		obs, actions_onehot = batch['obs'][:, transition_idx], batch['actions_onehot'][:]
		episode_num = obs.shape[0]
		inputs = []
		inputs.append(obs)

		if self.args.last_action:
			if transition_idx == 0:
				inputs.append(torch.zeros_like(actions_onehot[:, transition_idx]))
			else:
				inputs.append(actions_onehot[:, transition_idx - 1])
		if self.args.reuse_network:
			# same as above
			inputs.append(torch.eye(self.args.n_agents).unsqueeze(0).expand(episode_num, -1, -1))

		inputs = torch.cat([x.reshape(episode_num * self.args.n_agents, -1) for x in inputs], dim=1)

		return inputs


	def _get_action_prob(self, batch, max_episode_len, epsilon):
		episode_num = batch['obs'].shape[0]
		avail_actions = batch['avail_actions']  # coma doesnt need the target actor
		action_prob = []
		for transition_idx in range(max_episode_len):
			inputs = self._get_actor_inputs(batch, transition_idx)

			outputs, self.eval_hidden = self.eval_rnn(inputs, self.eval_hidden)
			outputs = outputs.view(episode_num, self.n_agents, -1)
			prob = torch.nn.functional.softmax(outputs, dim=-1)
			action_prob.append(prob)

		action_prob = torch.stack(action_prob, dim=1).cpu()

		action_num = avail_actions.sum(dim=-1, keepdim=True).float().repeat(1, 1, 1, avail_actions.shape[-1])  # number of actions that can be selected
		action_prob = ((1 - epsilon) * action_prob + torch.ones_like(action_prob) * epsilon / action_num)
		action_prob[avail_actions == 0] = 0.0

		# regularize probability of actions that cant be performed
		action_prob = action_prob / action_prob.sum(dim=-1, keepdim=True)

		# set to 0 again to avoid errors
		action_prob[avail_actions == 0] = 0.0

		return action_prob

	def init_hidden(self, episode_num):
		# initializes eval hidden for each agent
		self.eval_hidden = torch.zeros((episode_num, self.n_agents, self.args.rnn_hidden_dim))


	def _train_critic(self, batch, max_episode_len, train_step):
		# centralised critic, centralised training, learns values for state action pairs
		actions, reward, avail_actions, terminated = batch['actions'], batch['reward'], batch['avail_actions'], batch['terminated']
		actions_next = actions[:, 1:]
		padded_actions_next = torch.zeros(*actions[:, -1].shape, dtype=torch.long).unsqueeze(1)
		actions_next = torch.cat((actions_next, padded_actions_next), dim=1)
		mask = (1 - batch['padded'].float()).repeat(1, 1, self.n_agents)  # set td error of the filed experiences to 0, not to affect learning

		q_evals, q_next_target = self._get_q_values(batch, max_episode_len)
		q_values = q_evals.clone()  # to return at the end to calculate advantage and update the actor

		# take q values corresponding to each agent action and remove last dim as it only has one value
		q_evals = torch.gather(q_evals, dim=3, index=actions).squeeze(3)
		q_next_target = torch.gather(q_next_target, dim=3, index=actions_next).squeeze(3)
		targets = td_lambda_target(batch, max_episode_len, q_next_target.cpu(), self.args)

		td_error = targets.detach() - q_evals
		masked_td_error = mask * td_error  # to erase filled experience

		loss = (masked_td_error ** 2).sum() / mask.sum()

		self.critic_optimizer.zero_grad()
		loss.backward()
		torch.nn.utils.clip_grad_norm_(self.critic_parameters, self.args.grad_norm_clip)
		self.critic_optimizer.step()

		if train_step > 0 and train_step % self.args.target_update_cycle == 0:
			self.target_critic.load_state_dict(self.eval_critic.state_dict())

		return q_values

	def save_model(self, train_step):
		num = str(train_step // self.args.save_cycle)
		if not os.path.exists(self.model_dir):
			os.makedirs(self.model_dir)
		torch.save(self.eval_critic.state_dict(), self.model_dir + '/' + num + '_critic_params.pkl')
		torch.save(self.eval_rnn.state_dict(),  self.model_dir + '/' + num + '_rnn_params.pkl')



# this should be on a separate file, but as we are using just coma for now this shuold be enough
# temporal difference lambda - note: in td-lambda, lambda refers to the trace decay param, the higher it is the larger is the credit given to rewards from distant states and actions
def td_lambda_target(batch, max_episode_len, q_targets, args):
	episode_num = batch['obs'].shape[0]
	mask = (1 - batch['padded'].float()).repeat(1, 1, args.n_agents)
	terminated = (1 - batch['terminated'].float()).repeat(1, 1, args.n_agents)
	reward = batch['reward'].repeat((1, 1, args.n_agents))

	'''
	each experience has several n_step_return so give a max_episode_len dim to install n_step_return in last dim, the nth number represents n+1 step
	because the length of each episode in the batch may be different, a mask is needed to set the extra n-step return to 0, or it will affect the lambda return
	the lambda return in the nth experience is related to all n-step return after it
	if it is not set to 0, it is to late to do it after calculate td error
	terminated is used to set the q_targets and reward beyond the current episode length to 0
	'''
	n_step_return = torch.zeros((episode_num, max_episode_len, args.n_agents, max_episode_len))
	for transition_idx in range(max_episode_len - 1, -1, -1):
		# calculates step_return for this transition in the current episode
		# the obs on transition_idx has max_episode_len transition_idx returns and calculates each step return sep; also the index corresponding to n step return is n-1
		n_step_return[:, transition_idx, :, 0] = (reward[:, transition_idx] + args.gamma * q_targets[:, transition_idx] * \
													terminated[:, transition_idx] * mask[:, transition_idx])

		for n in range(1, max_episode_len - transition_idx):
			# n step return at time t = r + gamma * (n-1 step return at time t + 1)
			# but for n=1, step return =r + gamma * (Q at time t + 1)
			n_step_return[:, transition_idx, :, n] = (reward[:, transition_idx] + args.gamma * n_step_return[:, transition_idx + 1, :, n-1] * \
													mask[:, transition_idx])


	# lambda_return.shape = (episode_num, max_episode_len，n_agents)

	lambda_return = torch.zeros((episode_num, max_episode_len, args.n_agents))
	for transition_idx in range(max_episode_len):
		returns = torch.zeros((episode_num, args.n_agents))
		for n in range(1, max_episode_len - transition_idx):
			returns += pow(args.td_lambda, n-1) * n_step_return[:, transition_idx, :, n-1]
		lambda_return[:, transition_idx] = (1 - args.td_lambda) * returns + pow(args.td_lambda, max_episode_len - transition_idx - 1) * \
											n_step_return[:, transition_idx, :, max_episode_len - transition_idx - 1]


	return lambda_return

