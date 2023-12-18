import numpy as np
import random
from collections import deque
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from EarlyLanguageEnv_beg import EarlyLanguageEnvBeg
import logging
import matplotlib.pyplot as plt

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
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
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma * np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
def flatten_state(observation):
    # Flatten the observation dictionary into a single array
    return np.concatenate([
        np.array(observation['hour_of_day']).reshape(-1),
        np.array(observation['day_of_year']).reshape(-1),
        np.array(observation['year_index']).reshape(-1),
        np.array(observation['parent_response']).reshape(-1)
    ])

# Create your environment
env = EarlyLanguageEnvBeg()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

example_observation, _ = env.reset()
flattened_example_observation = flatten_state(example_observation)
state_size = flattened_example_observation.shape[0]

# Now use this state_size to initialize your DQNAgent
action_size = env.action_space.n
agent = DQNAgent(state_size, action_size)
batch_size = 100

# Training loop with data collection for graphing
total_episodes = 10
scores = []  # To store total reward/score per episode
epsilons = []  # To store epsilon values per episode

for e in range(total_episodes):
    observation, _ = env.reset()
    state = flatten_state(observation)
    state = np.reshape(state, [1, state_size])
    score = 0  # Reset score for the episode

    for time in range(500):
        action = agent.act(state)
        next_observation, reward, done, _ , _= env.step(action)
        next_state = flatten_state(next_observation)
        next_state = np.reshape(next_state, [1, state_size])
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        score += reward

        if len(agent.memory) > batch_size:
            agent.replay(batch_size)

        if done:
            print(f"Episode: {e+1}/{total_episodes}, Score: {score}, Epsilon: {agent.epsilon:.2f}")
            break

    scores.append(score)
    epsilons.append(agent.epsilon)

# Plotting the results
plt.figure(figsize=(12, 5))

# Plotting scores
plt.subplot(1, 2, 1)
plt.plot(scores)
plt.title('Scores per Episode')
plt.xlabel('Episode')
plt.ylabel('Score')

# Plotting epsilons
plt.subplot(1, 2, 2)
plt