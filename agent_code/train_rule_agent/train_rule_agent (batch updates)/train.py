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
import torch.nn.init as init
import os
import torch.nn as nn
import keyboard
import subprocess
import math

max_runs=20000
Max_Memory = 400
ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'BOMB', 'WAIT']
Model_Name = "Coin"
Batch_Size = 16
last_entries=400
normalization_length=250
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
class SimpleNet(nn.Module):
    def __init__(self, input_sizes, hidden_layer_sizes, output_size):
        super(SimpleNet, self).__init__()

        self.hidden_layers=nn.Sequential(
            nn.Linear(input_sizes, hidden_layer_sizes), 
            nn.ReLU(), 
            nn.Linear(hidden_layer_sizes, output_size))

    def forward(self, input_tensor):
        # Create a mask to identify NaN values
        nan_mask = torch.isnan(input_tensor)
        zero_mask = torch.eq(input_tensor, 0.0)
        #nan_mask = nan_mask.unsqueeze(1).expand_as(inputs[i])

        # Set NaN values to 0 using the mask
        masked_input = input_tensor.clone()
        masked_input[nan_mask] = 20.0
        masked_input[zero_mask] = 0.01
        
        final_output=self.hidden_layers(masked_input)
        #print("final_output: ", tensor_info(final_output), final_output, final_output.squeeze())
        return final_output.squeeze()
    
    def rand_weights(self):
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                init.xavier_uniform_(layer.weight)
                init.constant_(layer.bias, random.uniform(0, 0.2))


def train(self, mini_sample):
    #adapt the reward normalization to the mean and stabw of the last 
    #print(self.normalizer)
    if len(self.normalizer) !=0:
        mean=np.mean(np.array(self.normalizer))
        std=np.std(np.array(self.normalizer))
    else:
        mean=0
        std=1
    old_states_batch, actions_batch, rewards_batch, new_states_batch = zip(*mini_sample)
   
    # Filter out None values from new_states_batch
    new_states_batch = [states for states in new_states_batch if states is not None]
    # Convert list of arrays of tensors to list of tensors
    old_states = [torch.stack([state.clone().detach() for state in states]) for states in old_states_batch]
    new_states = [torch.stack([state.clone().detach() for state in states]) for states in new_states_batch]
        
    rewards = torch.tensor(rewards_batch, dtype=torch.float)
    actions = torch.tensor(actions_batch, dtype=torch.int)
    self.logger.debug("States and rewards translated to tensors.")
    

    intermediate_loss=0
    for old_state, action, reward, new_state in zip(old_states, actions, rewards, new_states):
        norm_reward= (reward-mean)/(std+0.000000001)
        #print("old_state", old_state)
        # Predict Q-values for both current and next states
        old_q_values = self.model(old_state)
        new_q_values = self.model(new_state)
        #print("old_q_values", old_q_values)
        #print("new_q_values", new_q_values)
        self.logger.debug("Q values calculated.")

        # Find the index corresponding to the chosen action
        action_index = int(action)

        # Update the Q-value for the chosen action using the Bellman equation
        if new_state is not None:
            old_q_target=old_q_values.clone()
            old_q_target[action_index] = norm_reward + self.gamma * torch.max(new_q_values)


        # Calculate loss and perform optimization step
        loss = self.criterion(old_q_values, old_q_target)
        intermediate_loss += loss
        #print(intermediate_loss)
    
    self.optimizer.zero_grad()
    final_loss=intermediate_loss/len(mini_sample)
    final_loss.backward()
    self.optimizer.step()

        
    self.game_loss=((self.game_loss[1]*self.game_loss[0]+intermediate_loss)/(self.game_loss[1]+len(mini_sample)),self.game_loss[1]+len(mini_sample))
    #print(self.game_loss)
    self.logger.debug("Model trained for this step")


def action_converter(self_action: str):
    action_map = {
        'UP': 0,
        'RIGHT': 1,
        'DOWN': 2,
        'LEFT': 3,
        'BOMB': 4,
        'WAIT': 5
    }
    return action_map.get(self_action, None)

def setup_training(self):
    global Model_Name

    self.plot_loss=[]
    self.plot_mean_loss=[]
    self.plot_rewards=[]
    self.plot_mean_rewards=[]
    self.total_reward=0
    self.total_loss=0
    self.record=float('nan')
    self.n_games=0
    self.action_counter=0

    self.game_reward=0
    self.lr=0.1
    self.gamma=0.0 #discount rate zwischen 0 und 1
    self.input_sizes = 6  # Size of each input np.array
    self.hidden_layer_sizes = 2*self.input_sizes  # Adjust hidden layer size
    self.output_size = len(ACTIONS)  # Number of output values
    self.model = SimpleNet(self.input_sizes, self.hidden_layer_sizes, self.output_size)
    self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
    self.criterion= nn.SmoothL1Loss()
    self.memory=deque(maxlen=Max_Memory)
    self.way=deque(maxlen=Max_Memory)
    self.normalizer=deque(maxlen=normalization_length)
    self.interstep_counter=0
    self.game_loss=(0,0)
    self.command=subprocess.list2cmdline(subprocess.sys.argv)
    old_command = None
    check_path=os.path.join('./' + Model_Name, "Checkpoint" + ".pt")
    if os.path.isfile(check_path):
        with open(check_path, "rb") as file:
            checkpoint=torch.load(file)
            self.model.load_state_dict(checkpoint['model_state'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state'])
            old_command=checkpoint['command']
            if old_command==self.command:
                self.total_reward= checkpoint['total_reward']
                self.normalizer= checkpoint['normalizer']
                self.record=checkpoint['record']
                self.total_loss= checkpoint['total_loss'].detach()
                self.plot_rewards= checkpoint['plot_rewards']
                self.plot_loss= checkpoint['plot_loss']
                self.n_games= checkpoint['n_games']    
                self.plot_mean_rewards= checkpoint['plot_mean_rewards']
                self.plot_mean_loss= checkpoint['plot_mean_loss']

    # Save output to a file
    model_folder_path = './' + Model_Name
    if not os.path.exists(model_folder_path):
        os.makedirs(model_folder_path)
        self.model.rand_weights()
        print("Weights and biases initialized")

    if not old_command==self.command:
        output_file_path = os.path.join(model_folder_path, "output.txt")
        with open(output_file_path, "a") as output_file:
            output_file.write(f"Command {self.command}\n")



def save(self):
    global Model_Name
    model_folder_path = './' + Model_Name
    torch.save(self.model, os.path.join(model_folder_path, Model_Name + ".pt"))
    torch.save({'model_state': self.model.state_dict(), 
                'optimizer_state': self.optimizer.state_dict(),
                'command': self.command,
                'normalizer': self.normalizer,
                'record': self.record,
                'total_reward': self.total_reward,
                'total_loss':self.total_loss,
                'plot_rewards':self.plot_rewards,
                'plot_loss':self.plot_loss,
                'n_games':self.n_games,      
                'plot_mean_rewards':self.plot_mean_rewards,
                'plot_mean_loss':self.plot_mean_loss
                }, os.path.join(model_folder_path, "Checkpoint" + ".pt"))

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
    global Model_Name
    global Batch_Size

    self.action_counter += 1

    # Convert parameters states to input format
    old_state = state_to_features(old_game_state)
    action = action_converter(self_action)
    #print("Actoin", action)
    reward = 0
    new_state = state_to_features(new_game_state)

    # Store the experience in memory
    self.memory.append((old_state, action, reward, new_state))
    self.way.append((old_game_state['self'][3], new_game_state['self'][3]))

    #e.INVALID_ACTION or dangerous action: ['UP', 'RIGHT', 'DOWN', 'LEFT', 'BOMB', 'WAIT']
    if (action == 0 and old_state[0]==0 ) or (action == 1 and old_state[1]==0) or (action == 2 and old_state[2]==0) or (action == 3 and old_state[3]==0)  or (action == 4 and old_state[4]==0) or (action == 5 and old_state[5]==0) or (action == 6 and old_state[6]==0):
        add_reward=-100
    else:
        add_reward=100
    self.game_reward += add_reward
    old_state, action, reward, new_state = self.memory[len(self.memory) - 1]
    self.memory[len(self.memory) - 1] = (old_state, action, reward + add_reward, new_state)


    self.normalizer.append(self.memory[len(self.memory)-1][2])


    if len(self.memory) > Batch_Size:
        mini_sample = random.sample(self.memory, Batch_Size)
    else:
        mini_sample= self.memory
        
    train(self, mini_sample)

    #plot
    if keyboard.is_pressed('q'):
        model_folder_path = './' + Model_Name
        save(self)
        plot(self, model_folder_path)
    

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
    global Model_Name
    global max_runs
    global Batch_Size
    global last_entries


    self.action_counter +=1
    self.n_games += 1



    if len(self.memory) > Batch_Size:
        mini_sample = random.sample(self.memory, Batch_Size)
    else:
        mini_sample= self.memory
    

    
    if self.game_reward >= self.record or math.isnan(self.record):
        self.record = self.game_reward
    #print("Game", n_games, "Reward", self.game_reward, "Record", record)
    self.total_reward += self.game_reward
    self.total_loss += self.game_loss[0]
    self.plot_rewards.append(self.game_reward)
    self.plot_loss.append(self.game_loss[0])
    mean_reward = self.total_reward/self.n_games
    mean_loss=self.total_loss/self.n_games        
    self.plot_mean_rewards.append(mean_reward)
    self.plot_mean_loss.append(mean_loss)    
    
    model_folder_path = './' + Model_Name
    output_file_path = os.path.join(model_folder_path, "output.txt")
    with open(output_file_path, "a") as output_file:
        output_file.write(f"Game {self.n_games}, Game Loss {self.game_loss[0]}, Mean Loss {mean_loss}, Actions {self.action_counter}, Mean reward {mean_reward}, Reward {self.game_reward}, Record {self.record}\n")

    # Store the model
    save(self)
    self.logger.info("Model saved to " + Model_Name + ".pt")

    #plot
    if self.n_games >=max_runs or keyboard.is_pressed('q'):
        plot(self, model_folder_path)


    #reset everything for the next round
    self.game_loss=(0,0)
    self.action_counter=0
    self.interstep_counter=0
    self.game_reward=0
    self.memory = deque(maxlen=Max_Memory)
    self.way = deque(maxlen=Max_Memory)
    self.logger.info("Everything set back for new game.")


    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')



def plot(self, model_folder_path: str):
    # Plotting rewards
    plt.figure(figsize=(8, 6))
    plt.title("Reward During Training")
    plt.xlabel("Number of Games")
    plt.ylabel("Score")
    plt.plot(self.plot_rewards, label="Game Reward")
    plt.plot(self.plot_mean_rewards, label="Mean Reward")
    
    # Determine y-axis limits for both positive and negative values
    y_min = min(min(self.plot_rewards), min(self.plot_mean_rewards))
    y_max = max(max(self.plot_rewards), max(self.plot_mean_rewards))
    plt.ylim(y_min, y_max)
    
    plt.text(len(self.plot_rewards) - 1, self.plot_rewards[-1], str(self.plot_rewards[-1]))
    plt.text(len(self.plot_mean_rewards) - 1, self.plot_mean_rewards[-1], str(self.plot_mean_rewards[-1]))
    plt.legend()

    # Save the reward plot to an image file
    plt.savefig(os.path.join(model_folder_path, self.command + "-Reward.png"))

    # Clear the current figure before plotting the next one
    plt.clf()

    # Convert plot_loss tensor values to NumPy array
    plot_loss_np = [loss_item.detach().numpy() for loss_item in self.plot_loss]
    plot_mean_loss_np = [loss_item.detach().numpy() for loss_item in self.plot_mean_loss]

    # Plot the loss data
    plt.figure(figsize=(8, 6))
    plt.title("Loss During Training")
    plt.xlabel("Game number")
    plt.ylabel("Loss")
    plt.plot(plot_loss_np, label="Mean loss per game")
    plt.plot(plot_mean_loss_np, label="Mean loss all games")
    plt.legend()

    # Save the loss plot to an image file
    plt.savefig(os.path.join(model_folder_path, self.command + "-Loss.png"))
    
    print("Model got trained with " + str(self.n_games) + " games. System exit now")
    sys.exit()
