import numpy as np
import random
from collections import deque
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from keras.regularizers import l2
from EarlyLanguageEnv_beg import EarlyLanguageEnvBeg
import logging
import matplotlib.pyplot as plt
from tqdm import tqdm

class DQNAgent:
    def __init__(self, state_size, action_size, action_space):
        self.state_size = state_size
        self.action_size = action_size
        self.action_space = action_space
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount 
        self.epsilon = 1.0  # exploration 
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Dense(64, input_dim=self.state_size, activation='relu', kernel_regularizer=l2(0.01)))
        model.add(Dropout(0.2))
        model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.01)))
        model.add(Dropout(0.2))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, env):
        if np.random.rand() <= self.epsilon:
            action_index = random.randrange(self.action_size)
        else:
            # Exploitation
            act_values = self.model.predict(state)
            action_index = np.argmax(act_values[0])  # best action

        # single integer --> multi-dimensional action
        multi_dimensional_action = np.unravel_index(action_index, env.action_space.nvec)
        return multi_dimensional_action

    def action_to_index(self, action):
        # tuple like (0, 0, 0, 1) to single index
        index = 0
        multiplier = 1
        for action_part, space_size in zip(reversed(action), reversed(self.action_space.nvec)):
            index += action_part * multiplier
            multiplier *= space_size

        return index

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            state = np.array(state).astype('float32').reshape(1, self.state_size)
            next_state = np.array(next_state).astype('float32').reshape(1, self.state_size)

            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])

            target_f = self.model.predict(state)
            action_index = self.action_to_index(action)
            target_f[0, action_index] = target

            self.model.fit(state, target_f, epochs=1, verbose=1)
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

def encode_parent_response(parent_response, phoneme_to_index, max_length):
    # single list of phonemes
    flat_response = [phoneme for sublist in parent_response for phoneme in sublist]

    encoded_response = np.zeros(max_length, dtype=int)
    for i, phoneme in enumerate(flat_response):
        if i >= max_length:
            break
        encoded_response[i] = phoneme_to_index.get(phoneme, 0)  # 0 for unknown
    return encoded_response
    
def flatten_state(observation, phoneme_to_index, max_response_length=50):
    hour_of_day = np.array(observation['hour_of_day']).reshape(-1)
    day_of_year = np.array(observation['day_of_year']).reshape(-1)
    year_index = np.array(observation['year_index']).reshape(-1)

    # parent_response of diff lengths
    #parent_response = np.array(observation['parent_response'], dtype=object)
    encoded_response = encode_parent_response(observation['parent_response'], phoneme_to_index, max_response_length)
    flattened_state = np.concatenate([hour_of_day, day_of_year, year_index, encoded_response])

    return flattened_state

def standardize_parent_response(parent_response, max_length):
    standardized_response = np.zeros(max_length, dtype=int)
    response_length = min(len(parent_response), max_length)
    standardized_response[:response_length] = parent_response[:response_length]
    return standardized_response

def print_progress(episode, total_episodes, update_frequency=10):
    if episode % update_frequency == 0 or episode == total_episodes - 1:
        bar_length = 30
        progress = episode / total_episodes
        block = int(round(bar_length * progress))
        text = "\rProgress: [{0}] {1:.2f}% Episode: {2}/{3}".format(
            "#" * block + "-" * (bar_length - block), progress * 100, episode, total_episodes
        )
        print(text, end="")

env = EarlyLanguageEnvBeg()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# setup initial variables for episodes
action_size = env.action_space.n
phoneme_list = env.adult_dictionary['consonants'] + env.adult_dictionary['vowels']
phoneme_to_index = {phoneme: i+1 for i, phoneme in enumerate(phoneme_list)}  # Starting from 1, 0 is reserved for padding
example_observation, _ = env.reset()
flattened_example_observation = flatten_state(example_observation, phoneme_to_index)
state_size = len(flattened_example_observation)
agent = DQNAgent(state_size, action_size, env.action_space)
batch_size = 64

# set up data tracking
total_episodes = 2
scores = []  
epsilons = []  
cookie_counts = []  
parent_response_scores = []  
no_actions = []  
wrong_guesses = []  
epsilons = []  

for e in tqdm(range(total_episodes), desc="Training Progress", unit="episode"):
    #print_progress(e, total_episodes, update_frequency=100)  
    observation, _ = env.reset()
    state = flatten_state(observation, phoneme_to_index)
    state_size = len(state)
    state = np.reshape(state, [1, state_size])
    score = 0  
    cookie_count = 0
    parent_responses = 0
    no_action = 0
    wrong_guess = 0

    done = False
    while not done:
        action = agent.act(state, env)
        next_observation, reward, done, _ , info = env.step(action)
        next_state = flatten_state(next_observation, phoneme_to_index)
        state_size = len(next_state)
        next_state = np.reshape(next_state, [1, state_size])
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        score = reward
        cookie_count = info['Cookie Count']
        parent_responses = info['Parent Responses']
        no_action = info['No Action Steps']
        wrong_guess = info['Wrong Guesses']

        if len(agent.memory) > batch_size:
            agent.replay(batch_size)

        if done:
            print(f"Episode: {e+1}/{total_episodes}, Score: {score}, Epsilon: {agent.epsilon:.2f}")
            break

    scores.append(score)
    epsilons.append(agent.epsilon)
    cookie_counts.append(cookie_count)
    parent_response_scores.append(parent_responses)
    no_actions.append(no_action)
    wrong_guesses.append(wrong_guess)

# plot results
plt.figure(figsize=(12, 12))

# rewards
plt.subplot(2, 3, 1)
plt.plot(score)
plt.title('Total Rewards per Episode')
plt.xlabel('Episode')
plt.ylabel('Total Reward')

# cookie counts 
plt.subplot(2, 3, 2)
plt.plot(cookie_counts)
plt.title('Cookie Counts per Episode')
plt.xlabel('Episode')
plt.ylabel('Cookie Count')

# response 
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