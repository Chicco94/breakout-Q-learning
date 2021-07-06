import gym
env = gym.make("Breakout-ram-v0")
observation = env.reset()
for _ in range(1000):
	env.render()
	print(env.action_space)
	action = env.action_space.sample() # your agent here (this takes random actions)
	print(action)	
	observation, reward, done, info = env.step(action)
	print(len(observation), reward, done, info)
	
	print("")
	if done:
		observation, reward, done, info = env.step(action)
		exit(0)
		observation = env.reset()
env.close()