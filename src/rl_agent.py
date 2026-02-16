import numpy as np
import random
import pickle
import os

class QLearningAgent:
    def __init__(self, actions, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.q_table = {}  # Map (state) -> [q_values]
        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def get_state(self, gap_level, time_of_day):
        """
        Discretize gap and time into a tuple state.
        gap_level: 'Surplus', 'Balanced', 'Shortage'
        time_of_day: 'Morning', 'Day', 'Evening', 'Night'
        """
        return (gap_level, time_of_day)

    def choose_action(self, state):
        if np.random.uniform(0, 1) < self.epsilon:
            return random.choice(self.actions)
        
        if state not in self.q_table:
            self.q_table[state] = np.zeros(len(self.actions))
            
        return self.actions[np.argmax(self.q_table[state])]

    def learn(self, state, action_idx, reward, next_state):
        if state not in self.q_table:
            self.q_table[state] = np.zeros(len(self.actions))
        if next_state not in self.q_table:
            self.q_table[next_state] = np.zeros(len(self.actions))
            
        old_value = self.q_table[state][action_idx]
        next_max = np.max(self.q_table[next_state])
        
        new_value = (1 - self.alpha) * old_value + self.alpha * (reward + self.gamma * next_max)
        self.q_table[state][action_idx] = new_value

    def save(self, filepath="models/q_agent.pkl"):
        directory = os.path.dirname(filepath)
        if not os.path.exists(directory):
            os.makedirs(directory)
        with open(filepath, "wb") as f:
            pickle.dump(self.q_table, f)
        print(f"RL Agent saved to {filepath}")

    def load(self, filepath="models/q_agent.pkl"):
        if os.path.exists(filepath):
            with open(filepath, "rb") as f:
                self.q_table = pickle.load(f)
            print(f"RL Agent loaded from {filepath}")
