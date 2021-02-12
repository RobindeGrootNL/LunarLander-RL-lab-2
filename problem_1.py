# Núria Casals 950801-T740
# Robin de Groot 981116-T091

# Copyright [2020] [KTH Royal Institute of Technology] Licensed under the
# Educational Community License, Version 2.0 (the "License"); you may
# not use this file except in compliance with the License. You may
# obtain a copy of the License at http://www.osedu.org/licenses/ECL-2.0
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS"
# BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing
# permissions and limitations under the License.
#
# Course: EL2805 - Reinforcement Learning - Lab 2 Problem 1
# Code author: [Alessio Russo - alessior@kth.se]
# Last update: 6th October 2020, by alessior@kth.se
#

# Load packages
from collections import namedtuple
import numpy as np
import gym
import matplotlib.pyplot as plt
from tqdm import trange
from DQN_agent import RandomAgent
from DQN_neural_networks import MyNetwork
from ExperienceBuffer import ExperienceReplayBuffer
from torch import nn
import torch

def running_average(x, N):
    ''' Function used to compute the running average
        of the last N elements of a vector x
    '''
    if len(x) >= N:
        y = np.copy(x)
        y[N-1:] = np.convolve(x, np.ones((N, )) / N, mode='valid')
    else:
        y = np.zeros_like(x)
    return y

# Import and initialize the discrete Lunar Laner Environment
env = gym.make('LunarLander-v2')
env.reset()

# Parameters
#N_episodes = 600                             # Number of episodes
#discount_factor = 0.95                       # Value of the discount factor
#n_ep_running_average = 50                    # Running average of 50 episodes
n_actions = env.action_space.n               # Number of available actions
dim_state = len(env.observation_space.low)   # State dimensionality
#batch_size = 32
#buffer_length = 16384                        # Max length of Experience Replay Buffer
#update_freq = 256
#learning_rate = 10e-4
#epsilon_max = 0.99
#epsilon_min = 0.05
#reward_threshold = 100

def run_DQN_lander(N_episodes=900, 
                    discount_factor=0.99, 
                    n_ep_running_average=50, 
                    n_actions=n_actions,
                    dim_state=dim_state,
                    batch_size=64,
                    buffer_length=16384,
                    update_freq=256,
                    learning_rate=10e-4,
                    epsilon_max=0.99,
                    epsilon_min=0.05,
                    reward_threshold=100,
                    parameter_changed='PutParameterAndValueHere'):
    # We will use these variables to compute the average episodic reward and
    # the average number of steps per episode
    episode_reward_list = []       # this list contains the total reward per episode
    episode_number_of_steps = []   # this list contains the number of steps per episode
    
    # Random agent initialization
    random_agent = RandomAgent(n_actions)
    
    # Experience and experience buffer initialization
    Experience = namedtuple('Experience',
                            ['state', 'action', 'reward', 'next_state', 'done'])
    
    buffer = ExperienceReplayBuffer(buffer_length)
    
    # Reset enviroment data and initialize variables
    state = env.reset()
    for i in range(buffer_length):
        # Take a random action
        action = random_agent.forward(state)
        # Get next state and reward.  The done variable
        # will be True if you reached the goal position,
        # False otherwise
        next_state, reward, done, _ = env.step(action)
        
        # Put experience in buffer
        exp = Experience(state, action, reward, next_state, done)
        buffer.append(exp)
    
        if done:
            state = env.reset()
        else:
            state = next_state
    
    ### Training process
    
    # trange is an alternative to range in python, from the tqdm library
    # It shows a nice progression bar that you can update with useful information
    EPISODES = trange(N_episodes, desc='Episode: ', leave=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device("cpu")
    
    # TO DO: CHECK IF STATE SPACE IS CORRECTLY DEFINED
    target_network = MyNetwork(input_size = dim_state, output_size = n_actions, device=device).to(device)
    network = MyNetwork(input_size = dim_state, output_size = n_actions, device=device).to(device)
    
    target_network.eval()
    network.train()
    
    optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)
    
    for i in EPISODES:
        # Reset enviroment data and initialize variables
        done = False
        state = env.reset()
        total_episode_reward = 0.
        t = 0
    
        # Define epsilon
        epsilon = np.max([epsilon_min, epsilon_max-(epsilon_max - epsilon_min)*(i-1)/(N_episodes*0.9-1)])
    
        while not done:
    
            target_network.eval()
            network.train()
    
            if i >= N_episodes - 20:
                env.render()
    
            # Put state in a tensor for the NN
            state_tensor = torch.tensor([state],
                                        requires_grad=False,
                                        dtype=torch.float32).to(device)
    
            # Take epsilon greedy action
            if torch.rand((1,)) < epsilon:
                action = env.action_space.sample()
            else:
                values = network(state_tensor)
                action = values.max(1)[1].item()
    
            # Get next state and reward.  The done variable
            # will be True if you reached the goal position,
            # False otherwise
            next_state, reward, done, _ = env.step(action)
    
            # Append experience to Buffer
            exp = Experience(state, action, reward, next_state, done)
            buffer.append(exp)
    
            # Update episode reward
            total_episode_reward += reward
    
            # Sample experiences from the buffer 
            states, actions, rewards, next_states, dones = buffer.sample_batch(batch_size)
    
            # Training process, set gradients to 0
            optimizer.zero_grad()
    
            NN_outputs = network(torch.tensor(states,
                                            requires_grad=True,
                                            dtype=torch.float32).to(device))
    
            NN_outputs = NN_outputs[torch.arange(batch_size), torch.tensor(actions).to(device)]
    
            # Compute output of the network given the states batch
            target_NN_outputs = target_network(torch.tensor(next_states,
                                                requires_grad=False,
                                                dtype=torch.float32).to(device)).detach()
    
            # y = reward +  discount_factor* (1 - dones) * target_q_values.
            target_NN_outputs = torch.tensor(rewards, dtype=torch.float32).to(device) + ~torch.tensor(dones).to(device) * (discount_factor * target_NN_outputs.max(1).values)
    
            # Compute loss function
            loss = nn.functional.mse_loss(
                            target_NN_outputs,
                            NN_outputs)
    
            # Compute gradient
            loss.backward()
    
            # Clip gradient norm to 1
            nn.utils.clip_grad_norm_(network.parameters(), max_norm=1.)
    
            # Perform backward pass (backpropagation)
            optimizer.step()
    
            if t % update_freq == 0:
                target_network.load_state_dict(network.state_dict())
    
            # Update state for next iteration
            state = next_state
            t += 1
    
        # Append episode reward and total number of steps
        episode_reward_list.append(total_episode_reward)
        episode_number_of_steps.append(t)
    
        # Close environment
        env.close()
    
        # Saving the network when the reward threshold is reached and stopping the training
        if running_average(episode_reward_list, n_ep_running_average)[-1] > reward_threshold:
            torch.save(target_network.cpu(), 'neural-network-1_{}.pth'.format(parameter_changed))
            break
    
        # Updates the tqdm update bar with fresh information
        # (episode number, total reward of the last episode, total number of Steps
        # of the last episode, average reward, average number of steps)
        EPISODES.set_description(
            "Episode {} - Reward/Steps: {:.1f}/{} - Avg. Reward/Steps: {:.1f}/{}".format(
            i, total_episode_reward, t,
            running_average(episode_reward_list, n_ep_running_average)[-1],
            running_average(episode_number_of_steps, n_ep_running_average)[-1]))
    
    #torch.save(target_network.cpu(), 'neural-network-1.pth')
    
    # Plot Rewards and steps
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 9))
    ax[0].plot([k for k in range(1, i+2)], episode_reward_list, label='Episode reward')
    ax[0].plot([k for k in range(1, i+2)], running_average(
        episode_reward_list, n_ep_running_average), label='Avg. episode reward')
    ax[0].set_xlabel('Episodes')
    ax[0].set_ylabel('Total reward')
    ax[0].set_title('Total Reward vs Episodes')
    ax[0].legend()
    ax[0].grid(alpha=0.3)
    
    ax[1].plot([k for k in range(1, i+2)], episode_number_of_steps, label='Steps per episode')
    ax[1].plot([k for k in range(1, i+2)], running_average(
        episode_number_of_steps, n_ep_running_average), label='Avg. number of steps per episode')
    ax[1].set_xlabel('Episodes')
    ax[1].set_ylabel('Total number of steps')
    ax[1].set_title('Total number of steps vs Episodes')
    ax[1].legend()
    ax[1].grid(alpha=0.3)
    plt.savefig('reward_steps_plot_{}.png'.format(parameter_changed))

    plt.show()
    plt.clf()

run_DQN_lander(parameter_changed='best_params')