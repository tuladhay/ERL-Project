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


def parse_arguments():
    global parser
    parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
    parser.add_argument('--algo', default='DDPG',
                        help='algorithm to use: DDPG | NAF')
    parser.add_argument('--env-name', default="HalfCheetah-v2",
                        help='name of the environment to run')
    parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                        help='discount factor for reward (default: 0.99)')
    parser.add_argument('--tau', type=float, default=0.001, metavar='G',
                        help='discount factor for model (default: 0.001)')
    parser.add_argument('--noise_scale', type=float, default=0.3, metavar='G',
                        help='initial noise scale (default: 0.3)')
    parser.add_argument('--final_noise_scale', type=float, default=0.3, metavar='G',
                        help='final noise scale (default: 0.3)')
    parser.add_argument('--exploration_end', type=int, default=100, metavar='N',
                        help='number of episodes with noise (default: 100)')
    parser.add_argument('--seed', type=int, default=4, metavar='N',
                        help='random seed (default: 4)')
    parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                        help='batch size (default: 128)')
    parser.add_argument('--num_steps', type=int, default=1000, metavar='N',
                        help='max episode length (default: 1000)')
    parser.add_argument('--num_episodes', type=int, default=1000, metavar='N',
                        help='number of episodes (default: 1000)')
    parser.add_argument('--hidden_size', type=int, default=128, metavar='N',
                        help='number of episodes (default: 128)')
    parser.add_argument('--updates_per_step', type=int, default=5, metavar='N',
                        help='model updates per simulator step (default: 5)')
    parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
                        help='size of replay buffer (default: 1000000)')
    parser.add_argument('--render', action='store_true',
                        help='render the environment')

class Evo():
    def __init__(self, num_evo_actors, evo_episodes=1):
        '''
        :param num_evo_actors: This is the number of genes/actors you want to have in the population
        :param evo_episodes: This is the number of evaluation episodes for each gene. See Algo1: 7, and Table 1
        population: initalizes 10 genes/actors
        num_elites: number of genes/actor that are selected, and do not undergo mutation (unless they are
                    selected again in the tournament selection
        tournament_genes: number of randomly selected genes to take the max(fitness) from,
                        and then put it back into the population

        noise_mean: mean for the gaussian noise for mutation
        noise_stddev: standard deviation for the gaussian noise for mutation
        '''
        self.num_actors = num_evo_actors
        self.population = [DDPG(args.gamma, args.tau, args.hidden_size,
                           env.observation_space.shape[0], env.action_space) for _ in range(10)]
        print("Initializing Evolutionary Actors")
        self.evo_episodes = evo_episodes
        self.num_elites = 3
        self.tournament_genes = 3

        self.noise_mean = 0.0
        self.noise_stddev = 0.01

    def initialize_fitness(self):
        for gene in self.population:
            gene.fitness = 0.0
        print("Initialized gene fitness")

    def evaluate_pop(self):
        for gene in self.population:
            fitness = []
            for i_episode in range(self.evo_episodes):
                state = torch.Tensor([env.reset()])
                episode_reward = 0
                for t in range(args.num_steps):
                    action = gene.select_action(state)
                    next_state, reward, done, _ = env.step(action.numpy()[0])
                    episode_reward += reward

                    action = torch.Tensor(action)
                    mask = torch.Tensor([not done])
                    next_state = torch.Tensor([next_state])
                    reward = torch.Tensor([reward])

                    memory.push(state, action, mask, next_state, reward)
                    state = next_state

                    if done:
                        break
                    # <end of time-steps>
                fitness.append(episode_reward)
                # <end of episodes>
            fitness = sum(fitness) / self.evo_episodes  # Algo2: 12
            gene.fitness = fitness
        print("All genes evaluated")
        print()

    def rank_population_selection(self):
        '''
        This function takes the current evaluated population (of k , then ranks them according to their fitness,
        then selects a number of elites (e), and then selects a set S of (k-e) using tournament selection.
        It then calls the mutation function to add mutation to the set S of genes.
        In the end this will replace the current population with a new one.
        '''
        ranked_pop = sorted(evo.population, key=lambda x: x.fitness, reverse=True)  # Algo1: 9
        elites = ranked_pop[:self.num_elites]
        set_s = []

        for i in range(len(ranked_pop)-len(elites)):
            tournament_genes = [random.choice(ranked_pop) for _ in range(self.tournament_genes)]
            tournament_winner = max(tournament_genes, key=attrgetter('fitness'))
            set_s.append(tournament_winner)

        mutated_set_S = self.mutation(set_s)
        # TODO: append the mutated set to population along with elites

    def mutation(self, set):
        """
        :param set: This is the set of (k-e) genes that are going to be mutated by adding noise
        :return: Returns the mutated set of (k-e) genes

        Adds noise to the weights and biases of each layer of the network
        """
        for gene in set:
            noise = np.random.normal(loc=self.noise_mean, scale=self.noise_stddev,
                                     size=np.shape(set[0].actor.linear1.weight))
            noise = torch.FloatTensor(noise)
            gene.actor.linear1.weight.data = gene.actor.linear1.weight.data + noise






if __name__ == "__main__":
    parse_arguments()
    args = parser.parse_args()

    env = NormalizedActions(gym.make(args.env_name))
    env = wrappers.Monitor(env, '/tmp/{}-experiment'.format(args.env_name), force=True)
    env.seed(args.seed)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    '''
    DEFINE THE ACTOR RL AGENT
    '''
    if args.algo == "NAF":
        agent = NAF(args.gamma, args.tau, args.hidden_size,
                          env.observation_space.shape[0], env.action_space)
        print("Initialized NAF")
    else:
        agent = DDPG(args.gamma, args.tau, args.hidden_size,
                          env.observation_space.shape[0], env.action_space)
        print("Initialized DDPG actor")

    '''
    DEFINE REPLAY BUFFER AND NOISE
    '''
    memory = ReplayMemory(args.replay_size)
    ounoise = OUNoise(env.action_space.shape[0])


    '''
    Initialize the Evolution Part
    '''
    evo = Evo(10)
    evo.initialize_fitness()
    evo.evaluate_pop()

    '''
    MOVE THE TRAINING CODE BELOW TO ITS RESPECTIVE FUNCTIONS
    '''
    rewards = []
    for i_episode in range(args.num_episodes):
        if i_episode < args.num_episodes // 2:
            state = torch.Tensor([env.reset()])     # algo line 6
            ounoise.scale = (args.noise_scale - args.final_noise_scale) * max(0, args.exploration_end -
                                                                              i_episode) / args.exploration_end + args.final_noise_scale
            ounoise.reset()
            episode_reward = 0
            for t in range(args.num_steps):     # line 7
                # forward pass through the actor network
                action = agent.select_action(state, ounoise)    # line 8
                next_state, reward, done, _ = env.step(action.numpy()[0])   # line 9
                episode_reward += reward

                action = torch.Tensor(action)
                mask = torch.Tensor([not done])
                next_state = torch.Tensor([next_state])
                reward = torch.Tensor([reward])

                if i_episode % 10 == 0:
                    env.render()

                memory.push(state, action, mask, next_state, reward)    # line 10

                state = next_state

                if len(memory) > args.batch_size * 5:
                    for _ in range(args.updates_per_step):
                        transitions = memory.sample(args.batch_size)    # line 11
                        batch = Transition(*zip(*transitions))

                        agent.update_parameters(batch)

                if done:

                    break
            rewards.append(episode_reward)
        else:
            state = torch.Tensor([env.reset()])
            episode_reward = 0
            for t in range(args.num_steps):
                action = agent.select_action(state)

                next_state, reward, done, _ = env.step(action.numpy()[0])
                episode_reward += reward

                next_state = torch.Tensor([next_state])

                if i_episode % 10 == 0:
                    env.render()

                state = next_state
                if done:
                    break

            rewards.append(episode_reward)
        print("Episode: {}, noise: {}, reward: {}, average reward: {}".format(i_episode, ounoise.scale,
                                                                              rewards[-1], np.mean(rewards[-100:])))

    env.close()
