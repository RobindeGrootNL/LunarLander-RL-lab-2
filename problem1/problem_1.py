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
import torch
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
N_episodes = 300                             # Number of episodes
discount_factor = 0.95                       # Value of the discount factor
n_ep_running_average = 50                    # Running average of 50 episodes
n_actions = env.action_space.n               # Number of available actions
dim_state = len(env.observation_space.high)  # State dimensionality
batch_size = 64
buffer_length = 10000                        # Max length of Experience Replay Buffer
update_freq = 170
learning_rate = 10e-4
epsilon_max = 0.99
epsilon_min = 0.05

print('n_actions: ', n_actions)
print('dim_state: ', dim_state)

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
for i in range(buffer_length):
    state = env.reset()
    # Take a random action
    action = random_agent.forward(state)
    # Get next state and reward.  The done variable
    # will be True if you reached the goal position,
    # False otherwise
    next_state, reward, done, _ = env.step(action)
    #print("reward: ", reward)
    # Put experience in buffer
    exp = Experience(state, action, reward, next_state, done)
    buffer.append(exp)
    # Update state for next iteration
    #state = next_state

### Training process

# trange is an alternative to range in python, from the tqdm library
# It shows a nice progression bar that you can update with useful information
EPISODES = trange(N_episodes, desc='Episode: ', leave=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device("cpu")

# TO DO: CHECK IF STATE SPACE IS CORRECTLY DEFINED
target_network = MyNetwork(input_size = dim_state, output_size = n_actions, device=device).to(device)
network = MyNetwork(input_size = dim_state, output_size = n_actions, device=device).to(device)

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

        if i >= 280:
            env.render()

        # Put state in a tensor for the NN
        state_tensor = torch.tensor([state],
                                    requires_grad=False,
                                    dtype=torch.float32).to(device)

        values = network(state_tensor)
        #print('values dimensions: ', values.shape)
        #print('values', values)

        # Take epsilon greedy action
        p = np.random.random()
        if p < epsilon:
            #print('random action')
            action = env.action_space.sample()
        else:
            #print('greedy action')
            action = values.max(1)[1].item()
        #print('action', action)

        # Get next state and reward.  The done variable
        # will be True if you reached the goal position,
        # False otherwise
        next_state, reward, done, _ = env.step(action)

        # Append experience to Buffer
        exp = Experience(state, action, reward, next_state, done)

        #print('Experience: ', exp)

        buffer.append(exp)

        # Update episode reward
        #print("action: ", action)
        #print("reward: ", reward)
        total_episode_reward += reward

        # Sample experiences from the buffer 
        # Sample a batch of 3 elements
        states, actions, rewards, next_states, dones = buffer.sample_batch(batch_size)

        # Training process, set gradients to 0
        optimizer.zero_grad()

        # Compute output of the network given the states batch
        # TO DO: CHECK IF requires_grad SHOULD BE TRUE OR FALSE HERE
        target_NN_outputs = target_network(torch.tensor(next_states,
                                            requires_grad=False,
                                            dtype=torch.float32).to(device))

        #print('target_NN_outputs shape: ', target_NN_outputs.shape)
        #print('target_NN_outputs[1] shape', target_NN_outputs[1].shape)

        # target_values = [0] * batch_size

        # TO DO: MAKE THIS INTO ONE LINE
        # y = reward +  discount_factor* (1 - dones) * target_q_values.
        target_values = rewards + discount_factor * (np.ones(batch_size)-dones) * target_NN_outputs.max().item()

        #for j in range(batch_size):
        #    if dones[j] == True:
        #        target_values[j] = rewards[j]
        #    else:
        #        #print(target_NN_outputs[i])
        #        #print(target_NN_outputs[i].max().item())
        #        target_values[j] = rewards[j] + discount_factor * target_NN_outputs[j].max().item()

        NN_outputs = network(torch.tensor(states,
                                        requires_grad=False,
                                        dtype=torch.float32).to(device))

        NN_values = [0] * batch_size

        for k in range(batch_size):
            NN_values[k] = NN_outputs[k][actions[k]]

        #print('type NN_values: ', type(NN_values))

        NN_values = torch.tensor(NN_values, requires_grad=True, dtype=torch.float32).to(device)

        target_values = torch.tensor(target_values, requires_grad=False, dtype=torch.float32).to(device)

        #print(target_values)
        #print(NN_values)

        # Compute loss function
        loss = nn.functional.mse_loss(
                        target_values,
                        NN_values)

        # Compute gradient
        loss.backward()

        # Clip gradient norm to 1
        nn.utils.clip_grad_norm_(network.parameters(), max_norm=1.)

        # Perform backward pass (backpropagation)
        optimizer.step()

        # TO DO: UPDATE TARGET NETWORK
        if update_freq % (t+1) == 0:
            #print("Episode number: {}. Updating target model".format(i))
            target_network.load_state_dict(network.state_dict())

        # Update state for next iteration
        state = next_state
        t += 1

    # Append episode reward and total number of steps
    episode_reward_list.append(total_episode_reward)
    episode_number_of_steps.append(t)

    # Close environment
    env.close()


    # TO DO: IMPLEMENT EARLY STOPPING WHEN PASSED THE THRESHOLD


    # Updates the tqdm update bar with fresh information
    # (episode number, total reward of the last episode, total number of Steps
    # of the last episode, average reward, average number of steps)
    EPISODES.set_description(
        "Episode {} - Reward/Steps: {:.1f}/{} - Avg. Reward/Steps: {:.1f}/{}".format(
        i, total_episode_reward, t,
        running_average(episode_reward_list, n_ep_running_average)[-1],
        running_average(episode_number_of_steps, n_ep_running_average)[-1]))

torch.save(target_network.cpu(), 'neural-network-1.pth')

# Plot Rewards and steps
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 9))
ax[0].plot([i for i in range(1, N_episodes+1)], episode_reward_list, label='Episode reward')
ax[0].plot([i for i in range(1, N_episodes+1)], running_average(
    episode_reward_list, n_ep_running_average), label='Avg. episode reward')
ax[0].set_xlabel('Episodes')
ax[0].set_ylabel('Total reward')
ax[0].set_title('Total Reward vs Episodes')
ax[0].legend()
ax[0].grid(alpha=0.3)

ax[1].plot([i for i in range(1, N_episodes+1)], episode_number_of_steps, label='Steps per episode')
ax[1].plot([i for i in range(1, N_episodes+1)], running_average(
    episode_number_of_steps, n_ep_running_average), label='Avg. number of steps per episode')
ax[1].set_xlabel('Episodes')
ax[1].set_ylabel('Total number of steps')
ax[1].set_title('Total number of steps vs Episodes')
ax[1].legend()
ax[1].grid(alpha=0.3)
plt.show()
