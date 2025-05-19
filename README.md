# negmas-rl

[![Documentation Status](https://readthedocs.org/projects/negmas-rl/badge/?version=latest)](https://negmas-rl.readthedocs.io/en/latest/?badge=latest)
[![Code style: ruff-format](https://img.shields.io/badge/code%20style-ruff_format-6340ac.svg)](https://github.com/astral-sh/ruff)
[![PyPI](https://img.shields.io/pypi/v/negmas-rl)](https://pypi.org/project/negmas-rl)

A simple RL wrapper for negotiations using negmas. This version is used for the AAMAS 2025 Tutorial on Reinforcement Learning for Automated Negotiation
## Installation

To install, run

```
(.venv) $ pip install git+https://github.com/yasserfarouk/negmas-rl-tutorial.git
```

## Development

To run style checks:

```
$ git clone https://github.com/yasserfarouk/negmas-rl-tutorial.git
$ cd negmas-rl-tutorial
$ uv sync --all-extras --dev
$ source .venv/bin/source
(.venv) $ uv pip install pre-commit
(.venv) $ pre-commit run -a
```

You can run all tests to check that everything is in order using:

```
(.venv) $ python -m pytest tests
```

