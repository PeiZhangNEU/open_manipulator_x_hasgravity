import gym

# env = gym.make('Hopper-v2')
env = gym.make('HalfCheetah-v2')

env.reset()
for i in range(1000):
    env.render()
    env.step(env.action_space.sample())

print(env.observation_space)
print(env.action_space)