import gym
import simple_driving
import numpy as np
import math
import torch
import random
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

##################################### Hyper parameters for DQN #######################################################################################

EPISODES = 50000                # Amount of episodes to run the training, higher the more better
LEARNING_RATE = 0.00025         # The learning rate for optimising the neural network weights
MEM_SIZE = 50000                # Maximum size of the replay memory 
GAMMA = 0.99                    # Discount factor               
REPLAY_START_SIZE = 10000       # Amount of samples to fill the replay memory with before we start learning
BATCH_SIZE = 32                 # Number of random samples from the replay memory to use for training each iteration
EPS_START = 0.1                 # Initial epsilon value for epsilon greedy action sampling
EPS_END = 0.0001                # Final epsilon value
EPS_DECAY = 4 * MEM_SIZE        # Amount of samples we decay epsilon over
MEM_RETAIN = 0.1                # Percentage of initial samples in replay memory to keep
NETWORK_UPDATE_ITERS = 5000     # Number of samples 'C' for slowly updating the target network \hat{Q}'s weights with the policy network Q's weights
FC1_DIMS = 128                  # Number of neurons in our MLP's first hidden layer
FC2_DIMS = 128                  # Number of neurons in our MLP's second hidden layer

# Global training metrics 
best_reward = 0
average_reward = 0
episode_history = []
episode_reward_history = []
np.bool = np.bool_
####################################################################################################################################################

class Network(torch.nn.Module):
    def __init__(self, env):
        super().__init__()
        self.input_shape = env.observation_space.shape
        self.action_space = env.action_space.n
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(*self.input_shape, FC1_DIMS),   # Input layer
            torch.nn.ReLU(),                                # Activation function
            torch.nn.Linear(FC1_DIMS, FC2_DIMS),            # Hidden layer
            torch.nn.ReLU(),                                # Activation function
            torch.nn.Linear(FC2_DIMS, self.action_space)    # Output layer
            )

        self.optimizer = optim.Adam(self.parameters(), lr=LEARNING_RATE)
        self.loss = nn.MSELoss()  # Loss function

    def forward(self, x):
        return self.layers(x)

# Store and retrive sampled experiences
class ReplayBuffer:
    def __init__(self, env):
        self.mem_count = 0
        self.states = np.zeros((MEM_SIZE, *env.observation_space.shape),dtype=np.float32)
        self.actions = np.zeros(MEM_SIZE, dtype=np.int64)
        self.rewards = np.zeros(MEM_SIZE, dtype=np.float32)
        self.states_ = np.zeros((MEM_SIZE, *env.observation_space.shape),dtype=np.float32)
        self.dones = np.zeros(MEM_SIZE, dtype=np.bool)

    def add(self, state, action, reward, state_, done):
    # if memory count is higher than the max memory size then overwrite previous values
        if self.mem_count < MEM_SIZE:
            mem_index = self.mem_count
        else:
            mem_index = int(self.mem_count % ((1-MEM_RETAIN) * MEM_SIZE) + (MEM_RETAIN * MEM_SIZE))  # avoid catastrophic forgetting, retain first 10% of replay buffer

        self.states[mem_index]  = state
        self.actions[mem_index] = action
        self.rewards[mem_index] = reward
        self.states_[mem_index] = state_
        self.dones[mem_index] =  1 - done

        self.mem_count += 1

    # returns random samples from the replay buffer, number is equal to BATCH_SIZE
    def sample(self):
        MEM_MAX = min(self.mem_count, MEM_SIZE)
        batch_indices = np.random.choice(MEM_MAX, BATCH_SIZE, replace=True)

        states  = self.states[batch_indices]
        actions = self.actions[batch_indices]
        rewards = self.rewards[batch_indices]
        states_ = self.states_[batch_indices]
        dones   = self.dones[batch_indices]

        return states, actions, rewards, states_, dones

class DQN_Solver:
    def __init__(self, env):
        self.memory = ReplayBuffer(env)
        self.policy_network = Network(env)  # Q
        self.target_network = Network(env)  # \hat{Q}
        self.target_network.load_state_dict(self.policy_network.state_dict())  # initially set weights of Q to \hat{Q}
        self.learn_count = 0    # keep track of the number of iterations we have learnt for

    # Epsilon greedy 
    def choose_action(self, observation):
        if self.memory.mem_count > REPLAY_START_SIZE:
            eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                math.exp(-1. * self.learn_count / EPS_DECAY)
        else:
            eps_threshold = 1.0
        if random.random() < eps_threshold:
            return np.random.choice(np.array(range(9)), p=[0.1, 0.2, 0.05, 0.15, 0.03, 0.07, 0.12, 0.1, 0.18]) 

        # Policy network Q chooses action with highest estimated Q-value so far
        state = torch.tensor(observation).float().detach()
        state = state.unsqueeze(0)
        self.policy_network.eval()  # Program only need forward pass
        with torch.no_grad():       # Don't compute gradients to save memory and computation
            q_values = self.policy_network(state)
        return torch.argmax(q_values).item()

    # Main training loop
    def learn(self):
        states, actions, rewards, states_, dones = self.memory.sample()  # Retrieve random batch of samples from the replay memory
        states = torch.tensor(states , dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        states_ = torch.tensor(states_, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.bool)
        batch_indices = np.arange(BATCH_SIZE, dtype=np.int64)

        self.policy_network.train(True)
        q_values = self.policy_network(states)                # Retrieve current q-value estimates from policy network, Q
        q_values = q_values[batch_indices, actions]           # q values for sampled actions only

        self.target_network.eval()                            # Evaluating the target network 
        with torch.no_grad():                                 
            q_values_next = self.target_network(states_)      # Target q-values for states_ for all actions (target network, \hat{Q})

        q_values_next_max = torch.max(q_values_next, dim=1)[0]  # Maximum q values for next state

        q_target = rewards + GAMMA * q_values_next_max * dones  # Target q-value

        loss = self.policy_network.loss(q_target, q_values)     # Compute the loss between target
        #Compute the gradients and update policy network Q weights
        self.policy_network.optimizer.zero_grad()
        loss.backward()
        self.policy_network.optimizer.step()
        self.learn_count += 1

        # set target network \hat{Q}'s weights to policy network Q's weights every C steps
        if  self.learn_count % NETWORK_UPDATE_ITERS == NETWORK_UPDATE_ITERS - 1:
            print("updating target network")
            self.update_target_network()

    def update_target_network(self):
        self.target_network.load_state_dict(self.policy_network.state_dict())

    def returning_epsilon(self):
        return self.exploration_rate

######################### if running locally you can just render the environment in pybullet's GUI #######################
env = gym.make("SimpleDriving-v0", apply_api_compatibility=True, renders=True, isDiscrete=True)
##########################################################################################################################

agent = DQN_Solver(env)
agent.policy_network.load_state_dict(torch.load("./policy_network.pkl"))
frames = []
state, info = env.reset()
agent.policy_network.eval()

while True:
    with torch.no_grad():
        q_values = agent.policy_network(torch.tensor(state, dtype=torch.float32))
    action = torch.argmax(q_values).item() # select action with highest predicted q-value
    state, reward, done, _, info = env.step(action)
    
    if done:
        break

env.close()
