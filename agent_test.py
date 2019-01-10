import argparse
import sys

import numpy as np
import torch
from unityagents import UnityEnvironment

from agent_ddpg import Agents


def run_agent(environment, agents):
    brain_name = environment.brain_names[0]
    environment_info = environment.reset(train_mode=False)[brain_name]
    states = environment_info.vector_observations
    scores = np.zeros(agents.num_agents)

    while True:
        actions = agents.act(states)
        environment_info = environment.step(actions)[brain_name]
        next_states = environment_info.vector_observations
        rewards = environment_info.rewards
        dones = environment_info.local_done
        agents.step(states, actions, rewards, next_states, dones)
        scores += rewards
        states = next_states

        if np.any(dones):
            break

    print('Score: {}'.format(scores))


def setup_environment(file_name):
    environment = UnityEnvironment(file_name=file_name)
    brain_name = environment.brain_names[0]
    brain = environment.brains[brain_name]
    environment_info = environment.reset(train_mode=False)[brain_name]
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

    # Load the network weights
    agents.actor_local.load_state_dict(torch.load('{}'.format(parameters.actor)))
    agents.critic_local.load_state_dict(torch.load('{}'.format(parameters.critic)))

    # Run the agent
    run_agent(environment=environment, agents=agents)

    # Close the environment
    environment.close()


def parse_arguments(arguments):
    parser = argparse.ArgumentParser(description='Run the Reacher tester.')
    parser.add_argument('--actor', '-a', required=True, type=str, help='Path to a file to load the network weights for the actor.')
    parser.add_argument('--critic', '-c', required=True, type=str, help='Path to a file to load the network weights for the critic.')
    parser.add_argument('--number', '-n', required=False, type=int, default=20, help='Number of agents.')

    return parser.parse_args(arguments)


if __name__ == '__main__':
    main(sys.argv[1:])
