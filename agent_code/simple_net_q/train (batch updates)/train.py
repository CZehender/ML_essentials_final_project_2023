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
scaling_factor=1
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
            nn.Linear(input_sizes, hidden_layer_sizes[0]), 
            nn.ReLU(), 
            nn.Linear(hidden_layer_sizes[0], hidden_layer_sizes[1]),
            nn.ReLU(), 
            nn.Linear(hidden_layer_sizes[1], output_size))

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
    global scaling_factor
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
        norm_reward= (reward-mean)/((std+0.000000001)*scaling_factor)
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
    self.gamma=0.15 #discount rate zwischen 0 und 1
    self.input_sizes = 28  # Size of each input np.array
    self.hidden_layer_sizes = [48, 28]  # Adjust hidden layer size
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
    self.game_reward += reward_from_events(self, events)

    # Convert parameters states to input format
    old_state = state_to_features(old_game_state)
    action = action_converter(self_action)
    #print(" Action: ", self_action)
    reward = reward_from_events(self, events)
    new_state = state_to_features(new_game_state)

    # Store the experience in memory
    self.memory.append((old_state, action, reward, new_state))
    self.normalizer.append(reward)
    self.way.append((old_game_state['self'][3], new_game_state['self'][3]))

    #e.INVALID_ACTION or dangerous action: #left, #right, #down, #up, #wait, #bomb 'UP': 0,'RIGHT': 1,'DOWN': 2,'LEFT': 3,'BOMB': 4,'WAIT': 5
    if (action == 2 and old_state[2]!=0 ) or (action == 1 and old_state[1]!=0) or (action == 0 and old_state[3]!=0) or (action == 3 and old_state[0]!=0)  or ((action == 5 or action == 4) and old_state[4]!=0) or (action == 4 and old_state[5]!=0):
        add_reward=-100
        '''print(" Bad action: ")
        if (action == 2 and old_state[2]!=0):
            print("Down") 
        if (action == 1 and old_state[1]!=0):
            print ("Right") 
        if (action == 0 and old_state[3]!=0):
            print("Up") 
        if (action == 3 and old_state[0]!=0):
            print("Left")
        if ((action == 5 or action == 4) and old_state[4]!=0):
            print("Bomb and Wait") 
        if (action == 4 and old_state[5]!=0):
            print("Bomb")'''
    
    else:
        add_reward=100
    self.game_reward += add_reward
    old_state, action, reward, new_state = self.memory[len(self.memory) - 1]
    self.memory[len(self.memory) - 1] = (old_state, action, reward + add_reward, new_state)


    if e.COIN_COLLECTED in events:
        train_COIN_COLLECTED (self, new_game_state, old_state, new_state)
    if e.BOMB_EXPLODED in events:
        train_BOMB_EXPLODED (self, old_game_state, new_game_state, old_state, new_state)
    if e.BOMB_DROPPED in events:
        train_BOMB_DROPPED (self, old_game_state, new_game_state, old_state, new_state)
    if e.KILLED_OPPONENT in events:
        train_KILLED_OPPONENT (self, new_game_state, old_state, new_state)
    if e.CRATE_DESTROYED in events:
        train_CRATE_DESTROYED (self, new_game_state, old_state, new_state)
    if e.COIN_FOUND in events:
        train_COIN_FOUND (self, new_game_state, old_state, new_state)
    
    #print("Memory reward: ", self.memory[len(self.memory)-1][2])

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
    self.game_reward +=reward_from_events(self, events)
    #self.memory.append((state_to_features(last_game_state), action_converter(last_action), reward_from_events(self, events), None))
    self.memory.append((state_to_features(last_game_state), action_converter(last_action), reward_from_events(self, events), state_to_features(None)))
    self.normalizer.append(reward_from_events(self, events))
    self.n_games += 1
    
    if check_explosion_kill(self, last_game_state, last_action):
        #print("explosion kill")
        minus_reward= -40
        self.game_reward += minus_reward
        old_state, action, reward, new_state = self.memory[len(self.memory) - 1]
        self.memory[len(self.memory) - 1] = (old_state, action, reward + minus_reward, new_state)
        #print("Last entry: ", self.memory[len(self.memory) - 1])
        #print("Length memory: ", len(self.memory))
        batch = list(itertools.islice(self.memory, len(self.memory)-1, len(self.memory)))
        #print(batch)
        train(self, batch)
        
    
    #Bestimme den finalen reward
    if (e.KILLED_SELF in events) or (e.GOT_KILLED in events) and not check_explosion_kill(self, last_game_state, last_action):
        #print("no explosion kill")
        if e.KILLED_SELF in events:
            add_reward=20
            minus_reward=-40
            last=4
        else:
            add_reward=32
            minus_reward=-52
            last=4
        # Modify the last "last" entries in memory
        for i in range(len(self.memory) - min(last, len(self.memory)), len(self.memory), 1):
            old_state, action, reward, new_state = self.memory[i]
            if (((np.sqrt((old_state[13])**2+(old_state[14])**2)<np.sqrt((new_state[13])**2+(new_state[14])**2) or new_state[4]==0) and old_state[4]==1) or (old_state[4]==0 and new_state[4]==0)):
                self.memory[i] = (old_state, action, reward + add_reward, new_state)
                self.game_reward += add_reward
            if (old_state[4]==1 and (np.sqrt((old_state[13])**2+(old_state[14])**2)>np.sqrt((new_state[13])**2+(new_state[14])**2) and not new_state[4]==0)) or (old_state[4]==0 and new_state[4]==1):
                self.memory[i] = (old_state, action, reward + minus_reward, new_state)
                self.game_reward += minus_reward
        if len(self.memory) > last:
            batch = list(itertools.islice(self.memory, len(self.memory)-last, len(self.memory)))
        else:
            batch= self.memory
        train(self, batch)        

    final_reward = (last_game_state['self'][1])/2
    #final_reward = self.action_counter
    # Modify the last `last_entries` entries in memory
    for i in range(len(self.memory) - min(last_entries, len(self.memory)), len(self.memory), 1):
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
        output_file.write(f"Game {self.n_games}, Game Loss {self.game_loss[0]}, Mean Loss {mean_loss}, Actions {self.action_counter}, Mean reward {mean_reward}, Final reward {final_reward},  Reward {self.game_reward}, Record {self.record}\n")

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


def reward_from_events(self, events: List[str]) -> float:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        #e.INVALID_ACTION: -80,
        e.COIN_COLLECTED: 20
        #e.WAITED: -0.5,
        #e.BOMB_DROPPED: -0.5
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

'''
def check_event(self, events: List[str], name: str) -> bool:
    return name in events
'''

def train_COIN_COLLECTED (self, new_game_state: dict, old_state: list, new_state: list):
    last=(new_game_state['step'])-self.interstep_counter
    #print("Last:", last)
    self.interstep_counter=(new_game_state['step'])
    #print("interstep_counter:", self.interstep_counter)
    x_coll_coin=new_game_state['self'][3][0]
    y_coll_coin=new_game_state['self'][3][1]
    add_reward=new_game_state['self'][1]/(self.interstep_counter)*40
    
    # Modify the last "last" entries in memory
    for i in range(len(self.memory) - min(last, len(self.memory)), len(self.memory), 1):
        old_state, action, reward, new_state = self.memory[i]
        old, new = self.way[i]
        if np.sqrt((old[0]-x_coll_coin)**2+(old[1]-y_coll_coin)**2)>np.sqrt((new[0]-x_coll_coin)**2+(new[1]-y_coll_coin)**2):
            self.memory[i] = (old_state, action, reward + add_reward, new_state)
            self.game_reward += add_reward
        if np.sqrt((old[0]-x_coll_coin)**2+(old[1]-y_coll_coin)**2)<np.sqrt((new[0]-x_coll_coin)**2+(new[1]-y_coll_coin)**2):
            self.memory[i] = (old_state, action, reward + add_reward, new_state)
            self.game_reward -= add_reward/2
    #print("Memory length:", len(self.memory))



def train_BOMB_EXPLODED (self, old_game_state: dict, new_game_state: dict, old_state: list, new_state: list):
    for i, ((x, y), c) in enumerate(old_game_state['bombs']):
        if c==0:
            bomb_x=x
            bomb_y=y
        #check if 4 tiles away (thats the minimum distance for an bomb to be relevant)                
        if np.sqrt((old_game_state ['self'][3][0]-bomb_x)**2+(old_game_state ['self'][3][1]-bomb_y)**2)<=4:
            last=4
            add_reward=42

            # Modify the last "last" entries in memory
            for i in range(len(self.memory) - min(last, len(self.memory)), len(self.memory), 1):
                old_state, action, reward, new_state = self.memory[i]
                #if old_state[4]==0 and new_state[4]==0:
                #if reward>=0:
                if (old_state[4]==1 and np.sqrt((old_state[13])**2+(old_state[14])**2)<np.sqrt((new_state[13])**2+(new_state[14])**2)) or (old_state[4]==1 and new_state[4]==0):
                    self.memory[i] = (old_state, action, reward + add_reward, new_state)
                    self.game_reward += add_reward
            #print("Memory length:", len(self.memory))



def train_KILLED_OPPONENT (self, new_game_state: dict, old_state: list, new_state: list):
    last=8
    add_reward=24

    # Modify the last "last" entries in memory
    for i in range(len(self.memory) - min(last, len(self.memory)), len(self.memory), 1):
        old_state, action, reward, new_state = self.memory[i]
        if not (old_state[0][4]==1 and (action== 4 or action== 5)):
            self.memory[i] = (old_state, action, reward + add_reward, new_state)
            self.game_reward += add_reward
            if action==4:
                self.memory[i] = (old_state, action, reward + add_reward+6, new_state)
                self.game_reward += add_reward + 6
    for i in range(len(self.memory) - min(last, len(self.memory)), len(self.memory) - 5, 1):
        old_state, action, reward, new_state = self.memory[i]
        if not (old_state[0][4]==1 and (action== 4 or action== 5)):
            self.memory[i] = (old_state, action, reward + 6, new_state)
            self.game_reward += 6
    #print("Memory length:", len(self.memory))

    

def train_BOMB_DROPPED (self, old_game_state: dict, new_game_state: dict, old_state: list, new_state: list):
    counter=0
    add_reward=32
    for item in old_game_state['others']:
                (x_, y_) = item[3]
                if -3<=(x_-old_game_state['self'][3][0])<=3 and -3<=(y_-old_game_state ['self'][3][1])<=3:
                    counter+=1
                    add_reward+=6*(5-np.sqrt((x_-old_game_state['self'][3][0])**2+(y_-old_game_state['self'][3][1])**2))
    if counter>0:
        last=5
        # Modify the last "last" entries in memory
        for i in range(len(self.memory) - min(last, len(self.memory)), len(self.memory), 1):
            old_state, action, reward, new_state = self.memory[i]
            if np.sqrt((old_state[20])**2+(old_state[21])**2)>np.sqrt((new_state[20])**2+(new_state[21])**2) or np.sqrt((old_state[22])**2+(old_state[23])**2)>np.sqrt((new_state[22])**2+(new_state[23])**2):
                self.memory[i] = (old_state, action, reward + add_reward, new_state)
                self.game_reward += add_reward



def train_COIN_FOUND (self, new_game_state: dict, old_state: list, new_state: list):
    add_reward=12
    old_state, action, reward, new_state = self.memory[len(self.memory) - 5]
    self.memory[len(self.memory) - 5] = (old_state, action, reward + add_reward*2, new_state)
    self.game_reward += add_reward*2
    for i in range(len(self.memory) - min(9, len(self.memory)), len(self.memory) -5, 1):
            old_state, action, reward, new_state = self.memory[i]
            if np.sqrt((old_state[7])**2+(old_state[8])**2)>np.sqrt((new_state[7])**2+(new_state[8])**2):
                self.memory[i] = (old_state, action, reward + add_reward, new_state)
                self.game_reward += add_reward



def train_CRATE_DESTROYED (self, new_game_state: dict, old_state: list, new_state: list):
    add_reward=24
    old_state, action, reward, new_state = self.memory[len(self.memory) - 5]
    self.memory[len(self.memory) - 5] = (old_state, action, reward + add_reward*2, new_state)
    self.game_reward += add_reward*2
    for i in range(len(self.memory) - min(9, len(self.memory)), len(self.memory) -5, 1):
            old_state, action, reward, new_state = self.memory[i]
            if np.sqrt((old_state[7])**2+(old_state[8])**2)>np.sqrt((new_state[7])**2+(new_state[8])**2):
                self.memory[i] = (old_state, action, reward + add_reward, new_state)
                self.game_reward += add_reward


def check_explosion_kill (self, last_game_state: dict, last_action: str):
    if last_action == 'UP':
        x=0
        y=1
    elif last_action == 'RIGHT':
        x=1
        y=0
    elif last_action == 'DOWN':
        x=0
        y=-1
    elif last_action == 'LEFT':
        x=-1
        y=0
    else:
        return False
    if last_game_state['explosion_map'][last_game_state['self'][3][0]+x][last_game_state['self'][3][1]+y]==1:
        return True
    else:
        return False
