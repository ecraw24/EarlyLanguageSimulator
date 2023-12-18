import gym
import random
from EarlyLanguageEnv_beg import EarlyLanguageEnvBeg

# Create an instance of your environment
env = EarlyLanguageEnvBeg()

# Number of episodes to run
num_episodes = 5

for episode in range(num_episodes):
    observation = env.reset()
    done = False

    while not done:
        # Random action
        action = env.action_space.sample()

        # Take the action in the environment
        observation, reward, done, truncated, info = env.step(action)

        # Render the environment
        env.render()

# Close the environment when done
env.close()
