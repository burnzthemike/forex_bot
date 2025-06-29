# rl_agent.py

import numpy as np
import pickle
from config import (
    ALPHA, GAMMA,
    EPSILON_START, EPSILON_MIN, EPSILON_DECAY
)

ACTIONS = [-1, 0, 1]  # short, hold, long

class QLearningAgent:
    def __init__(self, actions=ACTIONS, verbose=False):
        self.actions = actions
        self.alpha = ALPHA
        self.gamma = GAMMA
        self.epsilon = EPSILON_START
        self.epsilon_min = EPSILON_MIN
        self.epsilon_decay = EPSILON_DECAY
        self.q_table = {}
        self.verbose = verbose

    def validate_state(self, state):
        if not isinstance(state, tuple):
            raise ValueError(f"State must be tuple, got {type(state)}")
        for s in state:
            if not isinstance(s, int):
                raise ValueError(f"Each element in state must be int, got {type(s)}")
        return state

    def get_qs(self, state):
        state = self.validate_state(state)
        if state not in self.q_table:
            self.q_table[state] = np.zeros(len(self.actions))
        return self.q_table[state]

    def choose_action(self, state):
        state = self.validate_state(state)
        if np.random.rand() < self.epsilon:
            action = np.random.choice(self.actions)
            if self.verbose:
                print(f"[RL] Exploring | State: {state} | Action: {action}")
        else:
            qs = self.get_qs(state)
            action = self.actions[np.argmax(qs)]
            if self.verbose:
                print(f"[RL] Exploiting | State: {state} | Action: {action} | Qs: {qs}")
        return action

    def learn(self, state, action, reward, next_state):
        state = self.validate_state(state)
        next_state = self.validate_state(next_state)
        action_idx = self.actions.index(action)

        current_q = self.get_qs(state)[action_idx]
        next_max_q = np.max(self.get_qs(next_state))
        target_q = reward + self.gamma * next_max_q

        self.q_table[state][action_idx] += self.alpha * (target_q - current_q)

        if self.epsilon > self.epsilon_min:
            self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

        if self.verbose:
            print(f"[RL] Learn | State: {state} | Action: {action} | Reward: {reward:.4f} | New Q: {self.q_table[state][action_idx]:.4f} | Epsilon: {self.epsilon:.4f}")

    def save(self, path="q_table.pkl"):
        with open(path, "wb") as f:
            pickle.dump(self.q_table, f)

    def load(self, path="q_table.pkl"):
        try:
            with open(path, "rb") as f:
                self.q_table = pickle.load(f)
        except FileNotFoundError:
            self.q_table = {}

# ------------------------
# Discretization Function
# ------------------------

def discretize_state(price_change_pct, rsi, ema_cross, macd_cross, volatility):
    pc_bins = [-0.001, 0.001]        # 3 buckets: [-1, 0, 1]
    rsi_bins = [30, 70]              # 3 buckets: [0: oversold, 1: neutral, 2: overbought]
    vol_bins = [0.0005, 0.0015]      # 3 buckets: low, medium, high

    pc_bucket = int(np.digitize([price_change_pct], pc_bins)[0]) - 1  # shift to -1, 0, 1
    rsi_bucket = int(np.digitize([rsi], rsi_bins)[0])
    vol_bucket = int(np.digitize([volatility], vol_bins)[0])

    ema_cross = int(ema_cross) if ema_cross in (-1, 0, 1) else 0
    macd_cross = int(macd_cross) if macd_cross in (-1, 0, 1) else 0

    return (pc_bucket, rsi_bucket, ema_cross, macd_cross, vol_bucket)
