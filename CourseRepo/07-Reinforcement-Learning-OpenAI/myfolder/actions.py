import gym

env = gym.make('CartPole-v0')

observation = env.reset()

for t in range(1000):

	env.render()

	cart_pos, cart_vel, pole_ang, ang_vel = observation

	if pole_ang > 0:
		action = 1
	else: 
		action = 0

	observation, reward, done, info = env.step(action)