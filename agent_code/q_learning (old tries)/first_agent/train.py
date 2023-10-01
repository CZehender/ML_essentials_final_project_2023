from collections import namedtuple, deque
import pickle
from typing import List
import sys
import events as e
from .callbacks import state_to_features
import itertools
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn.functional as F
import os
import torch.nn as nn

max_runs=100
Max_Memory = 100
ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'BOMB', 'WAIT']
Model_Name = "CoinCollector1"
Batch_Size = 32
plot_rewards=[]
plot_mean_rewards=[]
total_reward=0
record=-10000
n_games=0
action_counter=0
last_entries=32
'''
def tensor_info(tensor_list):
    num_tensors = len(tensor_list)
    
    if num_tensors > 0:
        tensor_shape = tensor_list[0].shape
        num_values = torch.prod(torch.tensor(tensor_shape)).item()
    else:
        tensor_shape = None
        num_values = 0
    
    return num_tensors, num_values, tensor_shape
'''
class MultiBranchNet(nn.Module):
    def __init__(self, input_sizes, hidden_layer_sizes, output_size):
        super(MultiBranchNet, self).__init__()

        # Define separate branches for each input tensor
        self.branches = nn.ModuleList([
            self.create_branch(input_size, hidden_layer_sizes)
            for input_size in input_sizes
        ])

        # Final layer to merge information from all branches
        self.final_layer = nn.Linear(len(input_sizes) * hidden_layer_sizes[2], output_size)

    def create_branch(self, input_size, hidden_layer_sizes):
        layers = []
        prev_size = input_size
        for hidden_size in hidden_layer_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size
        return nn.Sequential(*layers)

    def forward(self, inputs):
        branch_outputs = []
        for i, branch in enumerate(self.branches):
            branch_output = branch(inputs[i])
            branch_outputs.append(branch_output)
        #print("branch_outputs: ", tensor_info(branch_outputs), branch_outputs)
        combined_output = torch.cat(branch_outputs, dim=0)
        #print("combined_output: ", tensor_info(combined_output), combined_output)
        final_output = self.final_layer(combined_output)
        #print("final_output: ", tensor_info(final_output), final_output)
        return final_output.squeeze()


def action_converter(self_action: str):
    if self_action =='UP':
        return 0
    elif self_action=='RIGHT':
        return 1 
    elif self_action=='DOWN':
        return 2
    elif self_action=='LEFT':
        return 3
    elif self_action=='BOMB':
        return 4
    elif self_action=='WAIT':
        return 5

def setup_training(self):
    self.game_reward=0
    self.lr=0.01
    self.gamma=0.9 #discount rate zwischen 0 und 1
    self.input_sizes = [4, 4, 4, 4, 4]  # Size of each input np.array
    self.hidden_layer_sizes = [16, 32, 64]  # Adjust hidden layer size
    self.output_size = len(ACTIONS)  # Number of output values
    self.model = MultiBranchNet(self.input_sizes, self.hidden_layer_sizes, self.output_size)
    self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
    self.criterion= nn.MSELoss()
    self.memory=deque(maxlen=Max_Memory)

    self.logger.debug("Self setup done")

def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    """
    Called once per step to allow intermediate rewards based on game events.

    When this method is called, self.events will contain a list of all game
    events relevant to your agent that occurred during the previous step. Consult
    settings.py to see what events are tracked. You can hand out rewards to your
    agent based on these events and your knowledge of the (new) game state.

    This is *one* of the places where you could update your agent.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    :param old_game_state: The state that was passed to the last call of `act`.
    :param self_action: The action that you took.
    :param new_game_state: The state the agent is in now.
    :param events: The events that occurred when going from  `old_game_state` to `new_game_state`
    """
    global Batch_Size

    # Convert parameters states to input format
    old_state = state_to_features(old_game_state)
    action = action_converter(self_action)
    reward = reward_from_events(self, events)
    new_state = state_to_features(new_game_state)


    # Store the experience in memory
    self.memory.append((old_state, action, reward, new_state))

    if len(self.memory) > Batch_Size:
        mini_sample = random.sample(self.memory, Batch_Size)
    else:
        mini_sample= self.memory

    train(self, mini_sample)


    global action_counter
    action_counter += 1
    self.game_reward += reward_from_events(self, events)
    

def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.
    This replaces game_events_occurred in this round.

    This is similar to game_events_occurred. self.events will contain all events that
    occurred during your agent's final step.

    This is *one* of the places where you could update your agent.
    This is also a good place to store an agent that you updated.

    :param self: The same object that is passed to all of your callbacks.
    """
    global max_runs
    global plot_rewards
    global plot_mean_rewards
    global total_reward
    global record
    global n_games
    global action_counter
    global Batch_Size
    global last_entries

    #Bestimme den finalen reward
    action_counter +=1
    final_reward=  self.game_reward/action_counter*2
    self.game_reward +=reward_from_events(self, events)+final_reward
    self.memory.append((state_to_features(last_game_state), action_converter(last_action), reward_from_events(self, events), None))


    n_games += 1
    if self.game_reward > record:
        record = self.game_reward
    #print("Game", n_games, "Reward", self.game_reward, "Record", record)
    total_reward += self.game_reward
    plot_rewards.append(self.game_reward)
    mean_reward = total_reward/n_games        
    plot_mean_rewards.append(mean_reward) 


    
    # Save output to a file
    model_folder_path = './' + Model_Name
    if not os.path.exists(model_folder_path):
        os.makedirs(model_folder_path)

    output_file_path = os.path.join(model_folder_path, "output.txt")
    with open(output_file_path, "a") as output_file:
        output_file.write(f"Game {n_games}, Final reward {final_reward}, Action Counter {action_counter}, plot_mean_rewards{mean_reward},  Reward {self.game_reward}, Record {record}\n")



    # Modify the last `last_entries` entries in memory
    for i in range(-1, -min(last_entries, len(self.memory)) - 1, -1):
        old_state, action, reward, new_state = self.memory[i]
        self.memory[i] = (old_state, action, reward + final_reward, new_state)

    # Perform experience replay on those last_entries to learn include the final result.
    if len(self.memory) > last_entries:
        if last_entries <= Batch_Size:
            mini_sample = list(itertools.islice(self.memory, len(self.memory)-last_entries, len(self.memory)))
        else:
            mini_sample = random.sample(list(itertools.islice(self.memory, len(self.memory)-last_entries, len(self.memory))), Batch_Size)
    else:
            mini_sample= self.memory

    train(self, mini_sample)
    
    


    action_counter=0
    self.game_reward=0
    self.memory = deque(maxlen=Max_Memory)
    self.logger.info("Everything set back for new game.")


    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')

    # Store the model
    torch.save(self.model, os.path.join(model_folder_path, Model_Name + ".pt"))
    self.logger.info("Model saved to " + Model_Name + ".pt")

    if n_games >=max_runs:
        # Plotting
        plt.figure(figsize=(8, 6))
        plt.title("Training...")
        plt.xlabel("Number of Games")
        plt.ylabel("Score")
        plt.plot(plot_rewards, label="Game Reward")
        plt.plot(plot_mean_rewards, label="Mean Reward")
        plt.ylim(ymin=0)
        plt.text(len(plot_rewards) - 1, plot_rewards[-1], str(plot_rewards[-1]))
        plt.text(len(plot_mean_rewards) - 1, plot_mean_rewards[-1], str(plot_mean_rewards[-1]))
        plt.legend()

        # Save the plot to an image file
        plt.savefig(os.path.join(model_folder_path, "plot.png"))
        print("Model got trained with "+ max_runs + " games. System exit now")
        
        sys.exit()


def reward_from_events(self, events: List[str]) -> float:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        e.COIN_COLLECTED: 5,
        e.KILLED_SELF: -5,
        e.INVALID_ACTION: -0.5,
        e.SURVIVED_ROUND: 0.1
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum


def train(self, mini_sample):
    # Unpack mini_sample
    old_states_batch, actions_batch, rewards_batch, new_states_batch = zip(*mini_sample)
    
    # Filter out None values from new_states_batch
    new_states_batch = [states for states in new_states_batch if states is not None]
    # Convert list of arrays of tensors to list of tensors
    old_states = [torch.stack([state.clone().detach() for state in states]) for states in old_states_batch]
    new_states = [torch.stack([state.clone().detach() for state in states]) for states in new_states_batch]
    
    rewards = torch.tensor(rewards_batch, dtype=torch.float)
    actions = torch.tensor(actions_batch, dtype=torch.int)
    self.logger.debug("States and rewards translated to tensors.")

    for old_state, action, reward, new_state in zip(old_states, actions, rewards, new_states):
        #print("old_state", old_state)
        # Predict Q-values for both current and next states
        old_q_values = self.model(old_state)
        new_q_values = self.model(new_state)
        self.logger.debug("Q values calculated.")

        # Find the index corresponding to the chosen action
        action_index = int(action)

        # Update the Q-value for the chosen action using the Bellman equation
        if new_state is not None:
            old_q_values[action_index] = reward + self.gamma * torch.max(new_q_values)

        # Calculate loss and perform optimization step
        self.optimizer.zero_grad()
        loss = self.criterion(self.model(old_state), old_q_values)
        loss.backward()
        self.optimizer.step()

    self.logger.debug("Model trained for this step")

