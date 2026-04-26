"""A simple RL wrapper for negotiations using negmas."""

from gymnasium.envs.registration import register

register(
    id="Negotiation-v0",
    entry_point="negmas_rl.env:NegoEnv",
    max_episode_steps=None,
)

register(
    id="MANegotiation-v0",
    entry_point="negmas_rl.env:NegoMultiEnv",
    max_episode_steps=None,
)
