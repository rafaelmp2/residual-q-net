import gym
import ma_gym
from common.arguments import common_args, value_mix_args, coma_args, commnet_args
from runner import Runner


N_EXPERIMENTS = 1

if __name__ == '__main__':

	args = common_args()

	# create ma_gym environment
	env = gym.make(args.env)

	print("Environment {} initialized".format(args.env))

	'''
	n_actions: number of actions in the environment for each agent
	n_agents: number of agents in the environment
	state_shape: a state is an array with all the values that describe the current state, i.e., all the
	features of the state
	obs_shape: an observation in a partially observable env is what each agent can see; an array with the
	values that describe what each agent can see
	episode_limit: maximum number of steps until which the episode will run if a terminal state wasnt reached
	before

	'''

	args.n_actions = env.action_space[0].n
	args.n_agents = env.n_agents

	if args.env == 'Combat-v0':
		args.state_shape = env.observation_space[0].shape[0] * env.n_agents
		args.obs_shape = env.observation_space[0].shape[0] 
	elif args.env == 'PredatorPrey7x7-v0':
		args.state_shape = env.observation_space[0].shape[0] * args.n_agents 
		args.obs_shape = env.observation_space[0].shape[0]
	elif args.env == 'Switch2-v0':
		args.state_shape = env.observation_space[0].shape[0] * args.n_agents
		args.obs_shape = env.observation_space[0].shape[0] 
	elif args.env == 'PredatorPrey5x5-v0':
		args.state_shape = env.observation_space[0].shape[0] * args.n_agents
		args.obs_shape = env.observation_space[0].shape[0]
	elif args.env == 'Checkers-v0':
		args.state_shape = env.observation_space[0].shape[0] * args.n_agents
		args.obs_shape = env.observation_space[0].shape[0]
		
	args.episode_limit = env._max_steps

	# load args
	if args.alg == 'vdn' or args.alg == 'qmix' or args.alg == 'qtran_base' or args.alg == 'rqn':
		args = value_mix_args(args)
	elif args.alg.find('coma') > -1:
		args = coma_args(args)
	else:
		raise Exception('No such algorithm!')

	if args.alg.find('commnet') > -1:
		args = commnet_args(args)

	print("CUDA set to", args.cuda)

	runner = Runner(env, args)

	# parameterize run according to the number of independent experiments to run, i.e., independent sets of n_epochs over the model; default is 1
	if args.learn:
		runner.run(N_EXPERIMENTS)