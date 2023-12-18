import numpy as np
import gymnasium
import random
from EarlyLanguageEnv_beg import EarlyLanguageEnvBeg
import logging

# Create your environment
env = EarlyLanguageEnvBeg()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Function to display a simple text-based progress bar
def display_progress_bar(percentage, bar_length=40):
    num_chars = int(percentage * bar_length)
    bar = '[' + '#' * num_chars + '-' * (bar_length - num_chars) + ']'
    return f'{bar} {percentage * 100:.2f}%'

NO_ACTION_PROBABILITY = 0.1  # 10% chance to take no action

def generate_valid_action_for_year(year_index, env):
    # With some probability, choose to take no action
    if np.random.rand() < NO_ACTION_PROBABILITY:
        return [0, 0, 0, 0]  # Representing no action

    max_consonants = len(env.child_dictionary['year 3']['consonants'])
    max_vowels = len(env.child_dictionary['year 3']['vowels'])

    if year_index == 0:  # Year 1: consonant then vowel
        consonant_index = np.random.randint(len(env.child_dictionary['year 1']['consonants']))
        vowel_index = np.random.randint(len(env.child_dictionary['year 1']['vowels']))
        return [consonant_index, vowel_index, 0, 0]
    elif year_index == 1:  # Year 2: consonant and vowel, must be duplicated
        consonant_index = np.random.randint(len(env.child_dictionary['year 2']['consonants']))
        vowel_index = np.random.randint(len(env.child_dictionary['year 2']['vowels']))
        return [consonant_index, vowel_index, consonant_index, vowel_index]
    else:  # Year 3: consonant, vowel, consonant, vowel (not necessarily duplicated)
        actions = []
        for _ in range(2):  # Two pairs of consonant and vowel
            consonant_index = np.random.randint(max_consonants) + 1
            vowel_index = np.random.randint(max_vowels) + 1
            actions.extend([consonant_index, vowel_index])
        return actions


# Define the state size and action size
state_size = 24 * 365 * 3 + 1 # One state for each hour in the 3 years
action_size = np.prod(env.action_space.nvec)  # Product of dimensions of the action space

# Initialize the Q-table with the correct dimensions
q_table = np.zeros((state_size, action_size))

# Hyperparameters
learning_rate = 0.5
discount_rate = 0.75
epsilon = 1.0
max_epsilon = 1.0
min_epsilon = 0.01
decay_rate = 0.001

# Total number of hours in the simulation
total_hours = 24 * 365 * 3 + 1

# Training loop
num_episodes = 25
for episode in range(num_episodes):
    observation, _ = env.reset()
    done = False
    total_rewards = 0

    while not done:
        # Calculate the state index
        state_index = observation['hour_of_day'] + (observation['day_of_year'] - 1) * 24 + observation['year_index'] * 24 * 365

        # Exploration-exploitation tradeoff
        if random.uniform(0, 1) < epsilon:
            action = generate_valid_action_for_year(observation['year_index'], env)
        else:
            action = np.unravel_index(np.argmax(q_table[state_index]), env.action_space.nvec)  # Exploit learned values

        new_observation, reward, done, _, info = env.step(action)

        # Update Q-table
        new_state_index = new_observation['hour_of_day'] + (new_observation['day_of_year'] - 1) * 24 + new_observation['year_index'] * 24 * 365
        
        try:
            q_table[state_index, np.ravel_multi_index(action, env.action_space.nvec)] += learning_rate * (reward + discount_rate * np.max(q_table[new_state_index]) - q_table[state_index, np.ravel_multi_index(action, env.action_space.nvec)])
        except ValueError as e:
            logging.error(f"Error updating Q-table: {e}")
            logging.error(f"Action: {action}, Action Space NVec: {env.action_space.nvec}")
        total_rewards += reward
        observation = new_observation

        # Calculate and display the progress bar
        current_hour = observation['hour_of_day'] + (observation['day_of_year'] - 1) * 24 + observation['year_index'] * 24 * 365
        progress_percentage = current_hour / total_hours
        print(f'\rEpisode: {episode}, Progress: {display_progress_bar(progress_percentage)}', end='')

    # Reduce epsilon
    epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)

    print(f"\nEpisode: {episode}, Total Reward: {total_rewards}, Epsilon: {epsilon}")

env.close()
