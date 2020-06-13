from StockEnv import StockEnv
from StockEnvTest import StockEnvTest
import torch
import numpy as np
from config import Config
import matplotlib.pyplot as plt
from collections import deque
from maddpg import MADDPG
import argparse
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--train', '-t', action='store_true', help='verbose flag')
args = parser.parse_args()

env = StockEnv()
# number of agents
num_agents = env.agent_count
print('Number of agents:', num_agents)
# size of each action
action_size = len(env.action_space[0])
print('Size of each action:', action_size)
# examine the state space
states = env.state
state_size = states.shape[1]
print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
print('The state for the first agent looks like:', states[0])

config = Config()
config.update_every = 1
config.batch_size = 64
config.buffer_size = int(1e6)
config.discount = 0.99
config.tau = 0.01
config.seed = 2
config.lr_actor = 1e-4
config.lr_critic = 1e-3
config.action_size = action_size
config.state_size = state_size
config.num_agents = num_agents
ma = MADDPG(config)

def train(n_episodes = 1):
	assets = []
	asset_window = deque(maxlen=100)
	for i_episode in range(n_episodes):
		env.reset()
		states= env.state
		ma.reset()
		while True:
			#print("Day: ", env.day)
			actions = ma.act(states)
			#print("Actions: ", actions)
			next_states, rewards, dones, _ = env.step(actions)
			ma.step(states, actions, rewards, next_states, dones)
			states = next_states
			if np.any(dones):
				break
		max_asset = np.max(env.asset_memory)
		asset_window.append(max_asset)
		assets.append(max_asset)	
		
		print("Episode: ", i_episode)
		print("Maximum asset value in this episode: ", max_asset)
		print("Critic loss: ", ma.loss[0])
		print("Actor loss: ", ma.loss[1])
		print("Timestep for MADDPG: ", ma.t_step)

        # periodic model checkpoint
		if i_episode % 4 == 0:
			torch.save(ma.agents[0].actor_local.state_dict(), 'checkpoint_actor.pth')
			torch.save(ma.agents[0].critic_local.state_dict(), 'checkpoint_critic.pth')
			# print('\rEpisode {}\tAverage Profit: {:.2f}\tCritic Loss: {:-11.10f}\tActor Loss: {:-10.6f}\t'
		    #       't_step {:-8d}'.
		    #       format(i_episode, np.mean(profits_window), ma.loss[0], ma.loss[1], ma.t_step))

			plt.plot(env.asset_memory[0],'r')
			plt.plot(env.asset_memory[1],'g')
			plt.show()        
			
		"""
		# Stopping the training after the avg profit of 10000 is reached
		if np.mean(profits_window) >= 10000:
			print('\nEnvironment solved in {:d} episodes!\tAverage Profit: {:.2f}'.format(i_episode,
		                                                                                 np.mean(profits_window)))
			torch.save(ma.agents[0].actor_local.state_dict(), 'checkpoint_actor.pth')
			torch.save(ma.agents[0].critic_local.state_dict(), 'checkpoint_critic.pth')
			break
		"""        

def test():

	dji = pd.read_csv("./Data/^DJI.csv")
	test_dji=dji[dji['Date']>'2014-01-01']
	dji_price=test_dji['Adj Close']
	dji_date = test_dji['Date']
	daily_return = dji_price.pct_change(1)
	daily_return=daily_return[1:]
	daily_return.reset_index()
	initial_amount = 10000

	total_amount=initial_amount
	account_growth=list()
	account_growth.append(initial_amount)

	for i in range(len(daily_return)):
	    total_amount = total_amount * daily_return.iloc[i] + total_amount
	    account_growth.append(total_amount)

	for player in ma.agents:
		player.actor_local.load_state_dict(torch.load('checkpoint_actor.pth'))
		player.critic_local.load_state_dict(torch.load('checkpoint_critic.pth'))
	
	envtest = StockEnvTest()
	envtest.reset()
	states = envtest.state
	ma.reset()
	while True:
		#print("Day: ", envtest.day)
		actions = ma.act(states)
		#print("Actions: ", actions)
		next_states, rewards, dones, _ = envtest.step(actions)
		ma.step(states, actions, rewards, next_states, dones)
		states = next_states
		if np.any(dones):
			break
	max_asset = np.max(envtest.asset_memory)
	print("Maximum asset value in this episode: ", max_asset)
	print("Critic loss: ", ma.loss[0])
	print("Actor loss: ", ma.loss[1])
	print("Timestep for MADDPG: ", ma.t_step)

	plt.plot(envtest.asset_memory[0],'r')
	plt.plot(envtest.asset_memory[1],'g')
	plt.plot(account_growth, 'b')
	plt.savefig('/home/prahlad/results/multiagent/test_stock_multi.png')
	plt.show()


print("Training!")
train()
print("Testing!")
test()