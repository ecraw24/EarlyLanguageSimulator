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

class DQNAgent:
    def __init__(self, state_size, action_size, action_space):
        self.state_size = state_size
        self.action_size = action_size
        self.action_space = action_space
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
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
            # Exploration: random action
            action_index = random.randrange(self.action_size)
        else:
            # Exploitation: choose best action based on model's prediction
            act_values = self.model.predict(state)
            action_index = np.argmax(act_values[0])  # returns index of the best action

        # Convert the single integer action index into a multi-dimensional action
        multi_dimensional_action = np.unravel_index(action_index, env.action_space.nvec)
        return multi_dimensional_action

    def action_to_index(self, action):
        # Assuming action is a tuple like (0, 0, 0, 1)
        # Convert this tuple to a single index.
        # The method of conversion depends on how your actions are structured.
        # One common approach for a MultiDiscrete action space is to treat it
        # like a number in mixed radix notation.

        index = 0
        multiplier = 1
        for action_part, space_size in zip(reversed(action), reversed(self.env.action_space.nvec)):
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

            action_index = self.action_to_index(action)  # Convert the action tuple to an index
            target_f[0, action_index] = target

            self.model.fit(state, target_f, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
def flatten_state(observation, max_response_length=50):
    # Flatten hour_of_day, day_of_year, and year_index
    hour_of_day = np.array(observation['hour_of_day']).reshape(-1)
    day_of_year = np.array(observation['day_of_year']).reshape(-1)
    year_index = np.array(observation['year_index']).reshape(-1)

    # Handle variable-length parent_response
    parent_response = np.array(observation['parent_response'], dtype=object)
    if len(parent_response) < max_response_length:
        # Pad shorter responses
        padding = np.zeros(max_response_length - len(parent_response), dtype=int)
        parent_response = np.concatenate([parent_response, padding])
    else:
        # Truncate longer responses
        parent_response = parent_response[:max_response_length]

    # Concatenate all flattened parts
    flattened_state = np.concatenate([hour_of_day, day_of_year, year_index, parent_response])

    return flattened_state


def standardize_parent_response(parent_response, max_length):
    standardized_response = np.zeros(max_length, dtype=int)
    response_length = min(len(parent_response), max_length)
    standardized_response[:response_length] = parent_response[:response_length]
    return standardized_response

def print_progress(episode, total_episodes):
    bar_length = 30
    progress = episode / total_episodes
    block = int(round(bar_length * progress))
    text = "\rProgress: [{0}] {1:.2f}% Episode: {2}/{3}".format(
        "#" * block + "-" * (bar_length - block), progress * 100, episode, total_episodes
    )
    print(text, end="")

# Create your environment
env = EarlyLanguageEnvBeg()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Now use this state_size to initialize your DQNAgent
action_size = env.action_space.n
example_observation, _ = env.reset()
flattened_example_observation = flatten_state(example_observation)
state_size = len(flattened_example_observation)

agent = DQNAgent(state_size, action_size, env.action_space)
batch_size = 64

# Training loop with data collection for graphing
total_episodes = 10
scores = []  # To store total reward/score per episode
epsilons = []  # To store epsilon values per episode
cookie_counts = []  # Number of cookies per episode
parent_response_scores = []  # Parent response scores per episode
no_actions = []  # Number of no actions per episode
wrong_guesses = []  # Number of wrong guesses per episode
epsilons = []  # Epsilon values for each episode

for e in range(total_episodes):
    print_progress(e, total_episodes)  
    observation, _ = env.reset()
    state = flatten_state(observation)
    state_size = len(state)
    state = np.reshape(state, [1, state_size])
    score = 0  # Reset score for the episode
    cookie_count = 0
    parent_responses = 0
    no_action = 0
    wrong_guess = 0

    done = False
    while not done:
        action = agent.act(state, env)
        next_observation, reward, done, _ , info = env.step(action)
        next_state = flatten_state(next_observation)
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

    print()

# Plotting the results with each variable in its own graph
plt.figure(figsize=(12, 12))

# Plotting total rewards per episode
plt.subplot(2, 3, 1)
plt.plot(score)
plt.title('Total Rewards per Episode')
plt.xlabel('Episode')
plt.ylabel('Total Reward')

# Plotting cookie counts per episode
plt.subplot(2, 3, 2)
plt.plot(cookie_counts)
plt.title('Cookie Counts per Episode')
plt.xlabel('Episode')
plt.ylabel('Cookie Count')

# Plotting parent response scores per episode
plt.subplot(2, 3, 3)
plt.plot(parent_response_scores)
plt.title('Parent Response Scores per Episode')
plt.xlabel('Episode')
plt.ylabel('Parent Response Score')

# Plotting no actions per episode
plt.subplot(2, 3, 4)
plt.plot(no_actions)
plt.title('No Actions per Episode')
plt.xlabel('Episode')
plt.ylabel('No Action')

# Plotting wrong guesses per episode
plt.subplot(2, 3, 5)
plt.plot(wrong_guesses)
plt.title('Wrong Guesses per Episode')
plt.xlabel('Episode')
plt.ylabel('Wrong Guesses')

# Plotting epsilon values over episodes
plt.subplot(2, 3, 6)
plt.plot(epsilons)
plt.title('Epsilon Values Over Episodes')
plt.xlabel('Episode')
plt.ylabel('Epsilon')

# Adjust layout for better fit
plt.tight_layout()
plt.show()