import util as ut
import numpy as np
import gymnasium as gym
import random
from abc import ABC, abstractmethod
from hyperparameters import *

def epsilon_greedy(state: int, q_values: np.ndarray, nb_action: int = None, epsilon=EPS):
    if random.random() < epsilon:
        return random.randint(0, nb_action-1 if nb_action is not None else q_values.shape[1]-1)
    else:
        return q_values[state, :nb_action].argmax() if nb_action is not None else np.argmax(q_values[state])


class TaxiOption(ABC):
    def __init__(self, env: gym.Env, nb_actions: int):
        self.q_values = np.zeros((env.observation_space.n, nb_actions))
        self.nb_actions = nb_actions

    @abstractmethod
    def is_final(self, state: int):
        pass

    def update_q_values(self, state: int, action: int, next_state: int, reward: float):
        self.q_values[state, action] += LR * (
                    reward + GAMMA * self.q_values[next_state].max() - self.q_values[state, action])

    def choose_action(self, state):
        return epsilon_greedy(state, self.q_values, self.nb_actions)

    def act(self, env: gym.Env, action: int):
        next_state, reward, done, _, _ = env.step(action)
        return next_state, reward, self.is_final(next_state) or done

    def step(self, env: gym.Env, state: int):
        action = self.choose_action(state)
        next_state, reward, done = self.act(env, action)
        self.update_q_values(state, action, next_state, reward)
        return action, next_state, reward, done


class GoToOption(TaxiOption):
    def __init__(self, env: gym.Env, location_num: int):
        super().__init__(env, env.action_space.n-2)
        self.nb_actions = env.action_space.n-2
        self.location = env.unwrapped.locs[location_num]

    def is_final(self, state: int):
        explicit_state = ut.extract_info(state)
        return explicit_state["taxi_row"] == self.location[0] and explicit_state["taxi_col"] == self.location[1]


class GetOption(TaxiOption):
    def __init__(self, env: gym.Env):
        super().__init__(env, env.action_space.n)

    def is_final(self, state: int):
        explicit_state = ut.extract_info(state)
        return explicit_state["passenger_loc"] == 4


class PutOption(TaxiOption):
    def __init__(self, env: gym.Env):
        super().__init__(env, env.action_space.n)

    def is_final(self, state: int):
        explicit_state = ut.extract_info(state)
        return explicit_state["passenger_loc"] != 4
