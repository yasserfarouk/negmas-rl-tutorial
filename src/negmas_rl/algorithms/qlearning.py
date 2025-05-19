from typing import Any, Self

import numpy as np
from gymnasium import Space
from gymnasium.spaces import Discrete, MultiDiscrete
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule


class QLearning:
    def __init__(
        self,
        policy: Any | None,
        env: GymEnv,
        learning_rate: float | Schedule,
        policy_kwargs: dict[str, Any] | None = None,
        stats_window_size: int = 100,
        tensorboard_log: str | None = None,
        verbose: int = 0,
        device: str = "auto",
        support_multi_env: bool = False,
        monitor_wrapper: bool = True,
        seed: int | None = None,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        supported_action_spaces: tuple[tuple[Space, ...], ...] | None = None,
        alpha: float = 0.1,
        gamma: float = 0.99,
        epsilon: float = 0.1,
    ) -> None:
        _ = policy
        self._alpha = alpha
        self._gamma = gamma
        self._epsilon = epsilon
        self._env = env
        self._learning_rate = learning_rate
        self._policy_kwargs = policy_kwargs
        self._stats_window_size = stats_window_size
        self._tensorboard_log = tensorboard_log
        self._verbose = verbose
        self._device = device
        self._support_multi_env = support_multi_env
        self._monitor_wrapper = monitor_wrapper
        self._seed = seed
        self._use_sde = use_sde
        self._sde_sample_freq = sde_sample_freq
        self._supported_action_spaces = supported_action_spaces
        self._qtable: np.ndarray = np.ndarray([])
        self._num_timesteps_at_start = 0
        self.num_timesteps = 0

    def predict(
        self,
        observation: np.ndarray | dict[str, np.ndarray],
        state: tuple[np.ndarray, ...] | None = None,
        episode_start: np.ndarray | None = None,
        deterministic: bool = False,
    ) -> tuple[np.ndarray, tuple[np.ndarray, ...] | None]:
        """
        Get the policy action from an observation (and optional hidden state).
        Includes sugar-coating to handle different observations (e.g. normalizing images).

        :param observation: the input observation
        :param state: The last hidden states (can be None, used in recurrent policies)
        :param episode_start: The last masks (can be None, used in recurrent policies)
            this correspond to beginning of episodes,
            where the hidden states of the RNN must be reset.
        :param deterministic: Whether or not to return deterministic actions.
        :return: the model's action and the next hidden state
            (used in recurrent policies)
        """
        if isinstance(env.observation_space, Discrete):
            state = tuple(
                int(observation),  # type: ignore
            )
        else:
            state = tuple(observation)  # type: ignore
        return (np.argmax(self._qtable[state]), None)

    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 100,
        tb_log_name: str = "run",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> Self:
        env, epsilon = self._env, self._epsilon
        alpha, gamma = self._alpha, self._gamma
        n_train = 0
        current_step = 0
        state, info = env.reset()
        assert isinstance(env.action_space, Discrete)
        if isinstance(env.observation_space, Discrete):
            n_states = [env.observation_space.n]
        elif isinstance(env.observation_space, MultiDiscrete):
            n_states = env.observation_space.nvec
        else:
            raise ValueError(
                f"Observation space of type {type(env.observation_space)} is not supported by q-learning"
            )
        if isinstance(env.action_space, Discrete):
            n_actions = [env.action_space.n]
        elif isinstance(env.action_space, MultiDiscrete):
            n_actions = env.action_space.nvec
        else:
            raise ValueError(
                f"action space of type {type(env.action_space)} is not supported by q-learning"
            )
        self._qtable = np.zeros(n_states + n_actions)
        if reset_num_timesteps:
            self.num_timesteps = 0
        while current_step < total_timesteps:
            if isinstance(env.observation_space, Discrete):
                state = tuple(
                    int(state),  # type: ignore
                )
            else:
                state = tuple(state)  # type: ignore
            # Choose action (epsilon-greedy)
            if np.random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()  # Explore
            else:
                action = np.argmax(self._qtable[state])  # Exploit

            # Take action and observe next state and reward
            next_state, reward, terminated, truncated, info = env.step(action)
            if isinstance(env.action_space, Discrete):
                action = tuple(
                    int(action),  # type: ignore
                )
            else:
                action = tuple(action)  # type: ignore
            if isinstance(env.observation_space, Discrete):
                next_state = tuple(
                    int(next_state),  # type: ignore
                )
            else:
                next_state = tuple(next_state)  # type: ignore
            done = terminated or truncated

            # Update Q-value
            old_value = self._qtable[state + action]
            next_max = np.max(self._qtable[next_state])
            new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
            self._qtable[state + action] = new_value

            state = next_state
            current_step += 1
            self.num_timesteps += 1

            if done:
                state, info = env.reset()
        return self
