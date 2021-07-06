import torch
import random
import numpy as np
from collections import deque
import gym
from model import Linear_QNet, QTrainer
from helper import plot
from time import sleep
from datetime import datetime

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001
RESUME = False
RENDER = False
RESUME_FILE_NAME = "model_69.pth"

class Agent:

	def __init__(self):
		self.n_games = 0
		self.epsilon = 0 # randomness
		self.gamma = 0.9 # discount rate
		self.memory = deque(maxlen=MAX_MEMORY) # automatically remove the oldes memory
		self.model = Linear_QNet(128,128,128,4)
		self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)


	def get_state(self, obs):
		'''get the state of the game based on the observation'''
		return np.array(obs, dtype=int)

	def remember(self, state, action, reward, next_state, done):
		self.memory.append((state, action, reward, next_state, done)) # popleft if MAX_MEMORY is reached

	def train_long_memory(self):
		if len(self.memory) > BATCH_SIZE:
			mini_sample = random.sample(self.memory, BATCH_SIZE)
		else:
			mini_sample = self.memory

		states, actions, rewards, next_states, dones = zip(*mini_sample)
		self.trainer.train_step(states, actions, rewards, next_states, dones)

	def train_short_memory(self, state, action, reward, next_state, done):
		self.trainer.train_step(state, action, reward, next_state, done)

	def get_action(self, state):
		# random moves: tradeoff exploration /exploitation
		self.epsilon = 80 - self.n_games
		final_move = 0
		if random.randint(0,200) < self.epsilon:
			final_move = random.randint(0,3)
		else:
			state0 = torch.tensor(state, dtype=torch.float)
			prediction = self.model(state0) # prediction is still a raw value like [5.0, 2.3, 1.8]
			final_move = torch.argmax(prediction).item() # get index of max value [5.0, 2.3, 1.8] -> 0

		return final_move


def train():
	plot_scores = []
	plot_mean_scores = []
	plot_min_last_10_scores = []
	total_score = 0
	record = 0
	agent = Agent()
	env = gym.make("Breakout-ram-v0")
	
	if (RESUME):
		file_name="./models/{}".format(RESUME_FILE_NAME)
		checkpoint = torch.load(file_name)
		agent.model.load_state_dict(checkpoint['state_dict'])
		agent.trainer.optimizer.load_state_dict(checkpoint['optimizer'])
		agent.n_games = checkpoint['epoch']

	observation = env.reset()
	score = 0
	currentLives = 5

	while True:
		if RENDER:
			env.render()

		# 1. get the old state
		state_old = agent.get_state(observation)

		# 2. get move based on current state
		final_move = agent.get_action(state_old)

		# 3. perform move based on finale_move and get new state
		observation, reward, done, info = env.step(final_move)

		# If life is lost then give negative reward
		if info["ale.lives"] < currentLives:
			reward = -1
			currentLives -= 1
			done = True
		score += reward
		state_new = agent.get_state(observation)

		# 4. train the short memory of the agent
		agent.train_short_memory(state_old, final_move, reward, state_new, done)

		# 5. store in the memory
		agent.remember(state_old, final_move, reward, state_new, done)

		if done:
			# train the long memory
			print("Epoch: {}\tScore: {}\tData: {}".format(agent.n_games,score,datetime.now()))
			agent.n_games += 1
			agent.train_long_memory()
			# set new record
			if score > record:
				record = score
				checkpoint = {
					'epoch': agent.n_games,
					'state_dict': agent.model.state_dict(),
					'optimizer': agent.trainer.optimizer.state_dict()
				}
				file_name="model_{}.pth".format(record)
				agent.model.save(checkpoint,file_name)

			# plot the results
			plot_scores.append(score)
			total_score += score
			mean_score = total_score/agent.n_games
			plot_mean_scores.append(mean_score)
			if (len(plot_scores) > 10 ):
				plot_min_last_10_scores.append(min(plot_scores[-10:]))
			else:
				plot_min_last_10_scores.append(0)
			plot(plot_scores,plot_mean_scores,plot_min_last_10_scores)
			
			# reset env
			observation = env.reset()
			score = 0
			currentLives = 5


if __name__=='__main__':
	train()