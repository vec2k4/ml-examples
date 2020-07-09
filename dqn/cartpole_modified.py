# Taken from: https://github.com/GaetanJUVIN/Deep_QLearning_CartPole
# Tutorial at: https://medium.com/@gtnjuvin/my-journey-into-deep-q-learning-with-keras-and-gym-3e779cc12762

import gym
import random
import os
import datetime
import numpy as np
import tensorflow.keras.backend as K
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from gym.envs.registration import register

from env import CartPoleExtEnv

# Keras backend functions: https://www.tensorflow.org/api_docs/python/tf/keras/backend

def mse(y_true, y_pred):
    return K.mean(K.square(y_true-y_pred))

def mse_verbose(y_true, y_pred):
    y_pred = K.print_tensor(y_pred, "y_pred")
    y_true = K.print_tensor(y_true, "y_true")
    diff = K.print_tensor(y_true-y_pred, "diff")
    squared = K.print_tensor(K.square(diff), "squared")
    mean = K.print_tensor(K.mean(squared), "mean")
    return mean

def action_augmentation_loss(y_true, y_pred):
    return K.sum(K.square(y_true-y_pred))

def action_augmentation_loss_verbose(y_true, y_pred):
    y_pred = K.print_tensor(y_pred, "y_pred")
    y_true = K.print_tensor(y_true, "y_true")
    diff = K.print_tensor(y_true-y_pred, "diff")    
    squared = K.print_tensor(K.square(diff), "squared")
    sum = K.print_tensor(K.sum(squared), "sum")
    return sum

class Agent():
    def __init__(self, state_size, action_size, env):
        self.weight_backup = "cartpole_weight.h5"
        self.state_size = state_size
        self.action_size = action_size
        self.env = env
        self.memory = deque(maxlen=2000)
        self.learning_rate = 0.001
        self.gamma = 0.95
        self.exploration_rate = 0.0 # 1.0
        self.exploration_min = 0.0 #0.01
        self.exploration_decay = 0.0 #0.995
        self.brain = self._build_model()
        self.target_brain = self._build_model()
        self.refresh_target()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss=action_augmentation_loss, optimizer=Adam(lr=self.learning_rate))

        if os.path.isfile(self.weight_backup):
            model.load_weights(self.weight_backup)
            self.exploration_rate = self.exploration_min
        return model

    def save_model(self):
        #self.brain.save(self.weight_backup)
        pass

    def act(self, state):
        if np.random.rand() <= self.exploration_rate:
            action = random.randrange(self.action_size)
            return action
        act_values = self.brain.predict(state)
        action = np.argmax(act_values[0])
        return action

    def remember(self, state, action, reward, next_state, done, memory_current, memory_next):
        self.memory.append((state, action, reward, next_state, done, memory_current, memory_next))

    def replay(self, sample_batch_size):
        if len(self.memory) < sample_batch_size:
            return
        
        #reward_index = np.array([[1], [1]])
        sample_batch = random.sample(self.memory, sample_batch_size)
        print("-> Training", end="")
        for state, action, reward, next_state, done, memory_current, memory_next in sample_batch:
            target = np.array([reward for _ in range(self.action_size)])           
            if not done:
                #start = datetime.datetime.now()
                rewards = np.array([m[1] for m in memory_current])
                next_states = np.array([m[0] for m in memory_next]).reshape(self.action_size, self.state_size)
                q_next = self.brain.predict_on_batch(next_states)
                next_actions = np.argmax(q_next, axis=1)
                q_next_hat = self.target_brain.predict_on_batch(next_states)
                q_next_hat_values = np.array([q[next_actions[i]] for i, q in enumerate(q_next_hat)])
                target = rewards + self.gamma * q_next_hat_values
                #end = datetime.datetime.now()
                #print("Time new:", (end-start))

                #start = datetime.datetime.now()
                #rewards_old = [memory_current[i][1] for i in range(len(memory_current))]
                #next_states_old = [memory_next[i][0] for i in range(len(memory_next))]
                #q_next_old = [self.brain.predict(s) for s in next_states_old]
                #next_actions_old = [np.argmax(q) for q in q_next_old]
                #q_next_hat_old = [self.target_brain.predict(s) for s in next_states_old]
                #q_next_hat_values_old = [q_next_hat_old[i][0][next_actions_old[i]] for i in range(len(q_next_hat_old))]
                #target_old = np.array(rewards_old) + self.gamma * np.array(q_next_hat_values_old)
                #end = datetime.datetime.now()
                #print("Time old:", (end-start))

                print(".", end="", flush=True)

            target = np.reshape(target, [1, self.action_size])

            self.brain.fit(state, target, epochs=1, verbose=0)
            print("*", end="", flush=True)
        if self.exploration_rate > self.exploration_min:
            self.exploration_rate *= self.exploration_decay
        print();  

    def refresh_target(self):
        self.target_brain.set_weights(self.brain.get_weights())

class CartPole:
    def __init__(self):
        self.sample_batch_size = 32
        self.episodes = 10000
        self.env = gym.make('CartPole-v999')

        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n
        self.agent = Agent(self.state_size, self.action_size, self.env)

        print(f"Action space: {self.env.action_space}")


    def run(self):
        scores = list()
        try:
            for index_episode in range(1, self.episodes):
                state = self.env.reset()
                state = np.reshape(state, [1, self.state_size])

                done = False
                index = 0
                print(f"Episode: {index_episode}")
                print("-> Remembering", end="")
                while not done:
#                    self.env.render()

                    memory_current = list()
                    for i in range(self.action_size):
                        ns, r, d = self.env.get_step(i)
                        ns = np.reshape(ns, [1, self.state_size])
                        memory_current.append((ns, r, d))

                    action = self.agent.act(state)

                    next_state, reward, done, _ = self.env.step(action)
                    next_state = np.reshape(next_state, [1, self.state_size])

                    memory_next = None
                    if not done:
                        memory_next = list()
                        for i in range(self.action_size):
                            ns, r, d = self.env.get_step(i)
                            ns = np.reshape(ns, [1, self.state_size])
                            memory_next.append((ns, r, d))
                        #for a in range(len(memory_next)):
                        #    print(f"State for action {a}: {memory_next[a]}")

                    self.agent.remember(state, action, reward, next_state, done, memory_current, memory_next)
                    print(".", end="", flush=True)
                    state = next_state
                    index += 1
                print()
                scores.append(index+1)
                print(f"-> Score: {index+1}")
                print(f"-> Min Score: {np.min(scores):.0f}, Max Score: {np.max(scores):.0f}, Median Score: {np.median(scores):.0f}, Mean Score: {np.mean(scores):.2f},")
                self.agent.replay(self.sample_batch_size)

                if index_episode % 5 == 0:
                    self.agent.refresh_target()
                    print("-> Target refreshed!")
                print("===========================")
        finally:
            self.agent.save_model()

if __name__ == "__main__":
    register(
        id='CartPole-v999',
        entry_point='env:CartPoleExtEnv',
        max_episode_steps=500,
        reward_threshold=475.0,
    )

    cartpole = CartPole()
    cartpole.run()