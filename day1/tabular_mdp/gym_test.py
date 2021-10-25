import gym


env = gym.make('Pendulum-v0')

state = env.reset()
done = False

while not done:
    state, reward, done, info = env.step(env.action_space.sample())
    env.render()

env.close()
