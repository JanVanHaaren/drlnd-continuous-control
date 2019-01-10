import argparse
import sys
from collections import deque

import matplotlib.pyplot as plt
import numpy as np
import torch
from unityagents import UnityEnvironment

from agent_ddpg import Agents


def run_ddpg(environment, agents, weights_actor, weights_critic, n_episodes=2000, max_t=1000):
    """Run Deep Deterministic Policy Gradient Learning for the given agents in the given environment.

    Params
    ======
        environment (UnityEnvironment):
        agents (Agents):
        weights_actor (str):
        weights_critic (str):
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
    """
    brain_name = environment.brain_names[0]

    scores = []  # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores

    for i_episode in range(1, n_episodes + 1):
        environment_info = environment.reset(train_mode=True)[brain_name]  # reset the environment
        state = environment_info.vector_observations
        score = np.zeros(agents.num_agents)

        for t in range(max_t):
            action = agents.act(state)
            environment_info = environment.step(action)[brain_name]  # send the action to the environment
            next_state = environment_info.vector_observations
            rewards = environment_info.rewards
            dones = environment_info.local_done
            agents.step(state, action, rewards, next_state, dones)
            state = next_state
            score += rewards

            if np.any(dones):
                print('\tSteps: ', t)
                break

        scores_window.append(np.mean(score))
        scores.append(np.mean(score))

        print('\rEpisode {}\tAverage score: {:.2f}\tScore: {:.3f}'.format(i_episode, np.mean(scores_window), np.mean(score)), end='')
        average_score = np.mean(scores_window)

        if i_episode % 25 == 0 or average_score > 30:
            print('\rEpisode {}\tAverage score: {:.2f}'.format(i_episode, average_score))
            torch.save(agents.actor_local.state_dict(), '{}'.format(weights_actor))
            torch.save(agents.critic_local.state_dict(), '{}'.format(weights_critic))

        if average_score >= 30.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage score: {:.2f}'.format(i_episode, average_score))
            break

    return scores


def plot_scores(scores, plot_name):
    fig = plt.figure()
    fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode')
    plt.savefig(plot_name)
    plt.show()


def setup_environment(file_name):
    environment = UnityEnvironment(file_name=file_name)
    brain_name = environment.brain_names[0]
    brain = environment.brains[brain_name]
    environment_info = environment.reset(train_mode=True)[brain_name]
    action_size = brain.vector_action_space_size
    state_size = len(environment_info.vector_observations[0])

    return environment, action_size, state_size


def main(arguments):
    parameters = parse_arguments(arguments)

    # Specify the path to the environment
    file_name = 'Reacher_Linux/Reacher.x86_64'

    # Set up the environment
    environment, action_size, state_size = setup_environment(file_name)

    # Set up the agents
    agents = Agents(state_size=state_size, action_size=action_size, num_agents=parameters.number, random_seed=0)

    # Run the Deep Deterministic Policy Gradient Learning algorithm
    scores = run_ddpg(environment=environment, agents=agents, weights_actor=parameters.actor, weights_critic=parameters.critic, n_episodes=parameters.episodes)

    # Plot the scores
    plot_scores(scores, parameters.plot)

    # Close the environment
    environment.close()


def parse_arguments(arguments):
    parser = argparse.ArgumentParser(description='Run the Navigation trainer.')
    parser.add_argument('--episodes', '-e', required=False, type=int, default=1000, help='Number of episodes to run.')
    parser.add_argument('--actor', '-a', required=True, type=str, help='Path to a file to store the network weights for the actor.')
    parser.add_argument('--critic', '-c', required=True, type=str, help='Path to a file to store the network weights for the critic.')
    parser.add_argument('--number', '-n', required=False, type=int, default=20, help='Number of agents.')
    parser.add_argument('--plot', '-p', required=False, type=str, default='plot.png', help='Path to a file to store a plot of the scores.')

    return parser.parse_args(arguments)


if __name__ == '__main__':
    main(sys.argv[1:])
