from options import *
from util import *
import numpy as np
import gymnasium as gym
from tqdm import tqdm
from abc import ABC, abstractmethod
from hyperparameters import *


class TaxiAgent(ABC):
    def __init__(self, gamma=GAMMA, lr=LR, epsilon=EPS):
        self.env = gym.make('Taxi-v3')
        self.gamma = gamma
        self.lr = lr
        self.epsilon = epsilon
        self.q_values: np.ndarray
        self.options: list[TaxiOption]

    @abstractmethod
    def execute_opt(self, state, action):
        pass

    def train(self, n_episodes=1000):
        reward_over_episodes = []
        for i in tqdm(range(n_episodes)):
            state = self.env.reset()[0]
            done = False
            total_reward = 0

            while not done:
                action = epsilon_greedy(state, self.q_values, None, self.epsilon)
                if action < self.env.action_space.n:
                    next_state, reward, done, _, _ = self.env.step(action)
                    self.q_values[state, action] += self.lr * (reward + self.gamma * self.q_values[next_state].max() - self.q_values[state, action])
                else:
                    next_state, reward = self.execute_opt(state, action)
                    explicit_state = extract_info(next_state)
                    done = explicit_state["passenger_loc"] ==explicit_state["dest"]
                state = next_state
                total_reward += reward
            reward_over_episodes.append(total_reward)
        return reward_over_episodes

    def test(self):
        self.env = gym.make('Taxi-v3', render_mode='human')
        done=False
        state = self.env.reset()[0]
        while not done:
            action = epsilon_greedy(state, self.q_values, None, 0)
            if action < self.env.action_space.n:
                next_state, reward, done, _, _ = self.env.step(action)
            else:
                next_state, reward = self.execute_opt(state, action)
                explicit_state = extract_info(next_state)
                done = explicit_state["passenger_loc"] ==explicit_state["dest"]
            state = next_state


class SMDPAgent(TaxiAgent):
    def __init__(self, get_put_options=False):
        super().__init__()
        self.options = [GetOption(self.env), PutOption(self.env)] if get_put_options else [GoToOption(self.env, loc_i) for loc_i in range(4)]
        self.q_values = np.zeros((self.env.observation_space.n, self.env.action_space.n + len(self.options)))

    def execute_opt(self, state, action):
        r_tptau = 0
        tau = 0
        opt_done = False
        origin_state = state
        total_reward = 0
        if self.options[action - self.env.action_space.n].is_final(state):
            opt_done = True
            r_tptau = -10
            total_reward = -10
            next_state = state
            tau = 1
        while not opt_done:
            _, next_state, reward, opt_done = self.options[action - self.env.action_space.n].step(self.env, state)
            total_reward += reward
            r_tptau += reward * (self.gamma ** tau)
            tau += 1
            state = next_state
        self.q_values[origin_state, action] += self.lr * (
                r_tptau + self.gamma ** tau * self.q_values[next_state].max() - self.q_values[
                        origin_state, action])
        return state, total_reward


class IntraOptAgent(TaxiAgent):
    def __init__(self, get_put_options=False):
        super().__init__()
        self.options = [GetOption(self.env), PutOption(self.env)] if get_put_options else [GoToOption(self.env, loc_i) for loc_i in range(4)]
        self.q_values = np.zeros((self.env.observation_space.n, self.env.action_space.n + len(self.options)))

    def execute_opt(self, state, option_num):
        total_reward = 0
        opt_done=False
        if self.options[option_num - self.env.action_space.n].is_final(state):
            self.q_values[state, option_num] += self.lr * -10
            opt_done = True
            total_reward = -10
        while not opt_done:
            opt_act, next_state, reward, opt_done = self.options[option_num - self.env.action_space.n].step(self.env, state)
            total_reward += reward
            self.q_values[state, option_num] += self.lr * (reward
                        + self.gamma * int(opt_done) * self.q_values[next_state].max()
                        + self.gamma * (1-int(opt_done)) * self.q_values[next_state, option_num]
                        - self.q_values[state, option_num])
            self.q_values[state, opt_act] += self.lr * (reward + self.gamma * self.q_values[next_state].max()-self.q_values[state, opt_act])
            for _other_opt_num in range(len(self.options)):
                other_opt = self.options[_other_opt_num]
                other_opt_num = _other_opt_num + self.env.action_space.n
                if other_opt_num != option_num and other_opt.q_values[state].argmax() == opt_act:
                    other_opt_done = other_opt.is_final(next_state)
                    self.q_values[state, other_opt_num] += self.lr * (reward
                                                                   + self.gamma * int(other_opt_done) * self.q_values[
                                                                       next_state].max()
                                                                   + self.gamma * (1 - int(other_opt_done)) * self.q_values[
                                                                       next_state, other_opt_num])
            state = next_state
        return state, total_reward

