import gymnasium
import random
from EarlyLanguageEnv_beg import EarlyLanguageEnvBeg

env = EarlyLanguageEnvBeg()
num_episodes = 5

for episode in range(num_episodes):
    observation = env.reset()
    done = False

    while not done:
        action = env.action_space.sample()
        observation, reward, done, truncated, info = env.step(action)
        env.render()

env.close()
