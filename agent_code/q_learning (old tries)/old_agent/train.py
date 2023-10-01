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
import keyboard

max_runs=20000
Max_Memory = 400
ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'BOMB', 'WAIT']
Model_Name = "Coin"
Batch_Size = 16
plot_loss=[]
plot_mean_loss=[]
plot_rewards=[]
plot_mean_rewards=[]
total_reward=0
total_loss=0
record=-10000
n_games=0
action_counter=0
last_entries=400
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
    self.game_reward=0
    self.lr=0.02
    self.gamma=0.4 #discount rate zwischen 0 und 1
    self.input_sizes = [6, 6, 6, 6, 6]  # Size of each input np.array
    self.hidden_layer_sizes = [16, 32, 64]  # Adjust hidden layer size
    self.output_size = len(ACTIONS)  # Number of output values
    self.model = MultiBranchNet(self.input_sizes, self.hidden_layer_sizes, self.output_size)
    self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
    self.criterion= nn.MSELoss()
    self.memory=deque(maxlen=Max_Memory)
    self.interstep_counter=0
    self.game_loss=(0,0)
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
    global Model_Name
    global plot_loss
    global plot_mean_loss
    global plot_rewards
    global plot_mean_rewards
    global n_games
    global Batch_Size
    global action_counter
    action_counter += 1
    self.game_reward += reward_from_events(self, events)

    # Convert parameters states to input format
    old_state = state_to_features(old_game_state)
    action = action_converter(self_action)
    reward = reward_from_events(self, events)
    new_state = state_to_features(new_game_state)

    # Store the experience in memory
    self.memory.append((old_state, action, reward, new_state))


    if e.COIN_COLLECTED in events:
        train_COIN_COLLECTED (self, new_game_state, old_state, new_state)
    if e.BOMB_EXPLODED in events:
        train_BOMB_EXPLODED (self, new_game_state, old_state, new_state)
    if e.BOMB_DROPPED in events:
        train_BOMB_DROPPED (self, new_game_state, old_state, new_state)
    if e.KILLED_OPPONENT in events:
        train_KILLED_OPPONENT (self, new_game_state, old_state, new_state)
    if e.CRATE_DESTROYED in events:
        train_CRATE_DESTROYED (self, new_game_state, old_state, new_state)
    if e.COIN_FOUND in events:
        train_COIN_FOUND (self, new_game_state, old_state, new_state)
    
    if len(self.memory) > Batch_Size:
        mini_sample = random.sample(self.memory, Batch_Size)
    else:
        mini_sample= self.memory
        
    train(self, mini_sample)

    #plot
    if keyboard.is_pressed('q'):
        model_folder_path = './' + Model_Name
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        torch.save(self.model, os.path.join(model_folder_path, Model_Name + ".pt"))
        plot(plot_rewards, plot_mean_rewards, plot_loss, plot_mean_loss, model_folder_path, n_games)
    

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
    global plot_rewards
    global plot_mean_rewards
    global total_reward
    global total_loss
    global record
    global n_games
    global action_counter
    global Batch_Size
    global last_entries
    global plot_loss
    global plot_mean_loss

    action_counter +=1
    self.game_reward +=reward_from_events(self, events)
    self.memory.append((state_to_features(last_game_state), action_converter(last_action), reward_from_events(self, events), None))
    n_games += 1
    
    
    #Bestimme den finalen reward
    if (e.KILLED_SELF in events) or (e.GOT_KILLED in events):
        if e.KILLED_SELF in events:
            add_reward=20
            minus_reward=-20
            last=4
        else:
            add_reward=32
            minus_reward=-32
            last=3
        # Modify the last "last" entries in memory
        for i in range(len(self.memory) - min(last, len(self.memory)-1)-1, len(self.memory) - 2, 1):
            old_state, action, reward, new_state = self.memory[i]
            if np.sqrt((old_state[2][0])**2+(old_state[2][1])**2)<np.sqrt((new_state[2][0])**2+(new_state[2][1])**2):
                self.memory[i] = (old_state, action, reward + add_reward, new_state)
                self.game_reward += add_reward
            if np.sqrt((old_state[2][0])**2+(old_state[2][1])**2)>=np.sqrt((new_state[2][0])**2+(new_state[2][1])**2):
                self.memory[i] = (old_state, action, reward + minus_reward, new_state)
                self.game_reward += minus_reward
        if len(self.memory) > last:
            batch = list(itertools.islice(self.memory, len(self.memory)-last, len(self.memory)))
        else:
            batch= self.memory
        train(self, batch)        

    final_reward = (last_game_state['self'][1])/2
    # Modify the last `last_entries` entries in memory
    for i in range(-1, -min(last_entries, len(self.memory)) - 1, -1):
        old_state, action, reward, new_state = self.memory[i]
        if reward>=0:
            self.memory[i] = (old_state, action, reward + final_reward, new_state)
            self.game_reward += final_reward
    if len(self.memory) > last_entries:
        if last_entries <= Batch_Size:
            mini_sample = list(itertools.islice(self.memory, len(self.memory)-last_entries, len(self.memory)))
        else:
            mini_sample = random.sample(list(itertools.islice(self.memory, len(self.memory)-last_entries, len(self.memory))), Batch_Size)
    else:
        mini_sample= self.memory
    train(self, mini_sample)
    

    
    if self.game_reward > record:
        record = self.game_reward
    #print("Game", n_games, "Reward", self.game_reward, "Record", record)
    total_reward += self.game_reward
    total_loss += self.game_loss[0]
    plot_rewards.append(self.game_reward)
    plot_loss.append(self.game_loss[0])
    mean_reward = total_reward/n_games
    mean_loss=total_loss/n_games        
    plot_mean_rewards.append(mean_reward)
    plot_mean_loss.append(mean_loss)    
    
    # Save output to a file
    model_folder_path = './' + Model_Name
    if not os.path.exists(model_folder_path):
        os.makedirs(model_folder_path)

    output_file_path = os.path.join(model_folder_path, "output.txt")
    with open(output_file_path, "a") as output_file:
        output_file.write(f"Game {n_games}, Game Loss {self.game_loss[0]}, Mean Loss {mean_loss}, Actions {action_counter}, Mean reward {mean_reward}, Final reward {final_reward},  Reward {self.game_reward}, Record {record}\n")

    # Store the model
    torch.save(self.model, os.path.join(model_folder_path, Model_Name + ".pt"))
    self.logger.info("Model saved to " + Model_Name + ".pt")

    #plot
    if n_games >=max_runs or keyboard.is_pressed('q'):
        plot(plot_rewards, plot_mean_rewards, plot_loss, plot_mean_loss, model_folder_path, n_games)


    #reset everything for the next round
    self.game_loss=(0,0)
    action_counter=0
    self.interstep_counter=0
    self.game_reward=0
    self.memory = deque(maxlen=Max_Memory)
    self.logger.info("Everything set back for new game.")


    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')


def reward_from_events(self, events: List[str]) -> float:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        e.INVALID_ACTION: -8,
        e.COIN_COLLECTED: 20,
        #e.WAITED: -0.5,
        e.BOMB_DROPPED: -0.5
        #e.COIN_FOUND: 12,
        #e.CRATE_DESTROYED: 8
        #e.KILLED_SELF: -20,
        #e.GOT_KILLED: -32
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

    intermediate_loss=0
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
        intermediate_loss += loss
        #print(intermediate_loss)
        loss.backward()
        self.optimizer.step()
    self.game_loss=((self.game_loss[1]*self.game_loss[0]+intermediate_loss)/(self.game_loss[1]+len(mini_sample)),self.game_loss[1]+len(mini_sample))
    #print(self.game_loss)
    self.logger.debug("Model trained for this step")



def plot(plot_rewards: list, plot_mean_rewards: list, plot_loss: list, plot_mean_loss: list, model_folder_path: str, n_games: int):
    # Plotting rewards
    plt.figure(figsize=(8, 6))
    plt.title("Training...")
    plt.xlabel("Number of Games")
    plt.ylabel("Score")
    plt.plot(plot_rewards, label="Game Reward")
    plt.plot(plot_mean_rewards, label="Mean Reward")
    
    # Determine y-axis limits for both positive and negative values
    y_min = min(min(plot_rewards), min(plot_mean_rewards))
    y_max = max(max(plot_rewards), max(plot_mean_rewards))
    plt.ylim(y_min, y_max)
    
    plt.text(len(plot_rewards) - 1, plot_rewards[-1], str(plot_rewards[-1]))
    plt.text(len(plot_mean_rewards) - 1, plot_mean_rewards[-1], str(plot_mean_rewards[-1]))
    plt.legend()

    # Save the reward plot to an image file
    plt.savefig(os.path.join(model_folder_path, "plot_reward.png"))

    # Clear the current figure before plotting the next one
    plt.clf()

    # Convert plot_loss tensor values to NumPy array
    plot_loss_np = [loss_item.detach().numpy() for loss_item in plot_loss]
    plot_mean_loss_np = [loss_item.detach().numpy() for loss_item in plot_mean_loss]

    # Plot the loss data
    plt.figure(figsize=(8, 6))
    plt.title("Loss During Training")
    plt.xlabel("Game number")
    plt.ylabel("Loss")
    plt.plot(plot_loss_np, label="Mean loss per game")
    plt.plot(plot_mean_loss_np, label="Mean loss all games")
    plt.legend()

    # Save the loss plot to an image file
    plt.savefig(os.path.join(model_folder_path, "plot_loss.png"))
    
    print("Model got trained with " + str(n_games) + " games. System exit now")
    sys.exit()

'''
def check_event(self, events: List[str], name: str) -> bool:
    return name in events
'''

def train_COIN_COLLECTED (self, new_game_state: dict, old_state: list, new_state: list):
    last=(new_game_state['step'])-self.interstep_counter
    #print("Last:", last)
    self.interstep_counter=(new_game_state['step'])
    #print("interstep_counter:", self.interstep_counter)
    x_coll_coin=old_state[0][0]-old_state[1][0]
    y_coll_coin=old_state[0][1]-old_state[1][1]
    add_reward=new_game_state['self'][1]/(self.interstep_counter)*20
    
    # Modify the last "last" entries in memory
    for i in range(-1, -min(last, len(self.memory)) - 1, -1):
        old_state, action, reward, new_state = self.memory[i]
        if np.sqrt((old_state[0][0]-x_coll_coin)**2+(old_state[0][1]-y_coll_coin)**2)>np.sqrt((new_state[0][0]-x_coll_coin)**2+(new_state[0][1]-y_coll_coin)**2):
            self.memory[i] = (old_state, action, reward + add_reward, new_state)
            self.game_reward += add_reward
    #print("Memory length:", len(self.memory))
    if len(self.memory) > last:
        batch = list(itertools.islice(self.memory, len(self.memory)-last, len(self.memory)))
    else:
        batch= self.memory
    #print("Length_batch:", batch)
    train(self, batch)


def train_BOMB_EXPLODED (self, new_game_state: dict, old_state: list, new_state: list):
    last=4
    #print("interstep_counter:", self.interstep_counter)
    x_own_bomb=old_state[0][0]-old_state[2][0]
    y_own_bomb=old_state[0][1]-old_state[2][1]
    add_reward=20

    # Modify the last "last" entries in memory
    for i in range(-1, -min(last, len(self.memory)) - 1, -1):
        old_state, action, reward, new_state = self.memory[i]
        if reward>=0:
        #if np.sqrt((old_state[0][0]-x_own_bomb)**2+(old_state[0][1]-y_own_bomb)**2)<np.sqrt((new_state[0][0]-x_own_bomb)**2+(new_state[0][1]-y_own_bomb)**2):
            self.memory[i] = (old_state, action, reward + add_reward, new_state)
            self.game_reward += add_reward
    #print("Memory length:", len(self.memory))
    if len(self.memory) > last:
        batch = list(itertools.islice(self.memory, len(self.memory)-last, len(self.memory)))
    else:
        batch= self.memory
    #print("Length_batch:", batch)
    train(self, batch)


def train_KILLED_OPPONENT (self, new_game_state: dict, old_state: list, new_state: list):
    last=8
    add_reward=24

    # Modify the last "last" entries in memory
    for i in range(-1, -min(last, len(self.memory)) - 1, -1):
        old_state, action, reward, new_state = self.memory[i]
        if reward>=0:
            self.memory[i] = (old_state, action, reward + add_reward, new_state)
            self.game_reward += add_reward
            if action==4:
                self.memory[i] = (old_state, action, reward + add_reward+6, new_state)
                self.game_reward += add_reward + 6
    for i in range(len(self.memory) - min(last, len(self.memory)-1)-1, len(self.memory) - 5, 1):
        old_state, action, reward, new_state = self.memory[i]
        if reward>=0:
            self.memory[i] = (old_state, action, reward + 6, new_state)
            self.game_reward += 6
    #print("Memory length:", len(self.memory))
    if len(self.memory) > last:
        batch = list(itertools.islice(self.memory, len(self.memory)-last, len(self.memory)))
    else:
        batch= self.memory
    #print("Length_batch:", batch)
    train(self, batch)
    

def train_BOMB_DROPPED (self, new_game_state: dict, old_state: list, new_state: list):
    if (old_state[0][0]==4.) or (old_state[0][1]==4.) or (old_state[0][2]==4.) or (old_state[0][3]==4.0):
        last=5
        add_reward=12

        # Modify the last "last" entries in memory
        for i in range(-1, -min(last, len(self.memory)) - 1, -1):
            old_state, action, reward, new_state = self.memory[i]
            if np.sqrt((old_state[3][0])**2+(old_state[3][1])**2)>np.sqrt((new_state[3][0])**2+(new_state[3][1])**2):
                self.memory[i] = (old_state, action, reward + add_reward, new_state)
                self.game_reward += add_reward
        if len(self.memory) > last:
            batch = list(itertools.islice(self.memory, len(self.memory)-last, len(self.memory)))
        else:
            batch= self.memory
        train(self, batch)


def train_COIN_FOUND (self, new_game_state: dict, old_state: list, new_state: list):
    add_reward=12
    self.game_reward += add_reward
    old_state, action, reward, new_state = self.memory[len(self.memory) - 5]
    self.memory[len(self.memory) - 5] = (old_state, action, reward + add_reward, new_state)
    train(self, self.memory[len(self.memory) - 5])


def train_CRATE_DESTROYED (self, new_game_state: dict, old_state: list, new_state: list):
    add_reward=8
    self.game_reward += add_reward
    old_state, action, reward, new_state = self.memory[len(self.memory) - 5]
    self.memory[len(self.memory) - 5] = (old_state, action, reward + add_reward, new_state)
    train(self, self.memory[len(self.memory) - 5])

'''
def check_explosion_kill (self, last_game_state: dict):
    return (last_game_state['explosion_map'][last_game_state['self'][3][0]][last_game_state['self'][3][1]]==1)
'''