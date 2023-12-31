import numpy as np
import gymnasium
import random
from EarlyLanguageEnv_beg import EarlyLanguageEnvBeg
import logging
import matplotlib.pyplot as plt
import pandas as pd

env = EarlyLanguageEnvBeg()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# progress bar
def display_progress_bar(percentage, bar_length=40):
    num_chars = int(percentage * bar_length)
    bar = '[' + '#' * num_chars + '-' * (bar_length - num_chars) + ']'
    return f'{bar} {percentage * 100:.2f}%'

def generate_valid_action_for_year(year_index, env):

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


state_size = 24 * 365 * 3 + 1 # One state for each hour in the 3 years
action_size = np.prod(env.action_space.nvec)  # dimensions of the action space

# q-table
q_table = np.zeros((state_size, action_size))

# hyperparameters
learning_rate = 0.5
discount_rate = 0.25 # how much a child cares about LT reward
epsilon = 1.0
max_epsilon = 1.0
min_epsilon = 0.01
decay_rate = 0.001

total_hours = 24 * 365 * 3 + 1

# data collection
episode_rewards = []
epsilons = []
cookie_counts = []
parent_response_scores = []
no_actions = []
wrong_guesses = []

# loop
num_episodes = 1000
for episode in range(num_episodes):
    observation, _ = env.reset()
    done = False
    total_rewards = 0
    cookie_count = 0
    parent_responses = 0
    no_action = 0
    wrong_guess = 0

    while not done:
        state_index = observation['hour_of_day'] + (observation['day_of_year'] - 1) * 24 + observation['year_index'] * 24 * 365

        # tradeoff
        if random.uniform(0, 1) < epsilon:
            action = generate_valid_action_for_year(observation['year_index'], env)
        else:
            action = np.unravel_index(np.argmax(q_table[state_index]), env.action_space.nvec)  # Exploit learned values

        new_observation, reward, done, _, info = env.step(action)

        # q-table
        new_state_index = new_observation['hour_of_day'] + (new_observation['day_of_year'] - 1) * 24 + new_observation['year_index'] * 24 * 365
        
        try:
            q_table[state_index, np.ravel_multi_index(action, env.action_space.nvec)] += learning_rate * (reward + discount_rate * np.max(q_table[new_state_index]) - q_table[state_index, np.ravel_multi_index(action, env.action_space.nvec)])
        except ValueError as e:
            logging.error(f"Error updating Q-table: {e}")
            logging.error(f"Action: {action}, Action Space NVec: {env.action_space.nvec}")
        
        # data collection
        total_rewards = reward
        cookie_count = info['Cookie Count']
        parent_responses = info['Parent Responses']
        no_action = info['No Action Steps']
        wrong_guess = info['Wrong Guesses']

        observation = new_observation

        episode_rewards.append(total_rewards)
        epsilons.append(epsilon)
        cookie_counts.append(cookie_count)
        parent_response_scores.append(parent_responses)
        no_actions.append(no_action)
        wrong_guesses.append(wrong_guess)

        # progress bar
        current_hour = observation['hour_of_day'] + (observation['day_of_year'] - 1) * 24 + observation['year_index'] * 24 * 365
        progress_percentage = current_hour / total_hours
        print(f'\rEpisode: {episode}, Progress: {display_progress_bar(progress_percentage)}', end='')

    # reduce epsilon
    epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)

    print(f"\nEpisode: {episode}, Total Reward: {total_rewards:.2f}, "
        f"Cookie Count: {cookie_count:.2f}, Parent Response Score: {parent_responses:.2f}, "
        f"No Action: {no_action:.2f}, Wrong Guesses: {wrong_guess:.2f}, "
        f"Epsilon: {epsilon:.2f}")

env.close()

# plot results
plt.figure(figsize=(12, 12))

# rewards
plt.subplot(2, 3, 1)
plt.plot(episode_rewards)
plt.title('Total Rewards per Episode')
plt.xlabel('Episode')
plt.ylabel('Total Reward')

# cookie counts
plt.subplot(2, 3, 2)
plt.plot(cookie_counts)
plt.title('Cookie Counts per Episode')
plt.xlabel('Episode')
plt.ylabel('Cookie Count')

# parent response
plt.subplot(2, 3, 3)
plt.plot(parent_response_scores)
plt.title('Parent Response Scores per Episode')
plt.xlabel('Episode')
plt.ylabel('Parent Response Score')

# no action
plt.subplot(2, 3, 4)
plt.plot(no_actions)
plt.title('No Actions per Episode')
plt.xlabel('Episode')
plt.ylabel('No Action')

# wrong guesses
plt.subplot(2, 3, 5)
plt.plot(wrong_guesses)
plt.title('Wrong Guesses per Episode')
plt.xlabel('Episode')
plt.ylabel('Wrong Guesses')

# epsilon
plt.subplot(2, 3, 6)
plt.plot(epsilons)
plt.title('Epsilon Values Over Episodes')
plt.xlabel('Episode')
plt.ylabel('Epsilon')

plt.tight_layout()
plt.show()




# target_sentences = [['g', 'oo'], ['k', 'oo', 'k', 'oo'], ['k', 'oo', 'k', 'ee']]

# flattened_data = []

# for episode_info in episode_data:
#     episode_number = episode_info['Episode']
#     total_reward = episode_info['Total Reward']
#     for year, data_list in episode_info['Year Data'].items():
#         for data in data_list:
#             child_sentence = data.get('Child Sentence', [])
#             if child_sentence in target_sentences:
#                 flattened_data.append({
#                     'Episode #': episode_number,
#                     'Year': year,
#                     'Reward': total_reward,
#                     'Parent Sentence': data.get('Parent Sentence', ''),
#                     'Child Sentence': child_sentence,
#                     'Cookie Count': data.get('Cookie Count', 0),
#                     'Parent Response': data.get('Parent Responses', 0),
#                     'No Action': data.get('No Action Steps', 0),
#                     'Wrong Guess': data.get('Wrong Guesses', 0)
#                 })

# df = pd.DataFrame(flattened_data)
# df.to_csv('episode_data.csv', index=False)