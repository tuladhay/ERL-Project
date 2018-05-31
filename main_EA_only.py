import numpy as np
import random
import time
import gym
from gym import wrappers
import argparse
import math
from collections import namedtuple
from itertools import count
import random
from operator import attrgetter
import gym
import numpy as np
from gym import wrappers
import torch
from ddpg import DDPG
from naf import NAF
from normalized_actions import NormalizedActions
from ounoise import OUNoise
from replay_memory import ReplayMemory, Transition
import pickle

if __name__=="__main__":
    random.seed(1234)
    np.random.seed(1234)
    env = gym.make('FrozenLake-v0')
    env.seed(0)

    num_population = 10
    num_steps = 20

