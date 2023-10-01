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

#Copy of the rule_based_agent:
from collections import deque
from random import shuffle

import numpy as np

import settings as s

max_runs=20000
Max_Memory = 400
ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'BOMB', 'WAIT']
Model_Name = "Coin"
Batch_Size = 16
last_entries=400
normalization_length=250
scaling_factor=0.01
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
    def __init__(self, input_sizes, hidden_layer_sizes, final_hidden_layer_size, output_size, slope):
        super(MultiBranchNet, self).__init__()

        # Define separate branches for each input tensor
        self.branches = nn.ModuleList([
            self.create_branch(input_size, hidden_layer_sizes, slope)
            for input_size in input_sizes
        ])

        #Add a hidden layer to the final layer:
        self.final_layers=nn.Sequential(
            nn.Linear(len(input_sizes) * hidden_layer_sizes[len(hidden_layer_sizes)-1], final_hidden_layer_size), 
            nn.LeakyReLU(slope), 
            nn.Linear(final_hidden_layer_size, output_size))

    def create_branch(self, input_size, hidden_layer_sizes, slope):
        layers = []
        prev_size = input_size
        for hidden_size in hidden_layer_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.LeakyReLU(slope))
            prev_size = hidden_size
        return nn.Sequential(*layers)

    def forward(self, inputs):
        branch_outputs = []
        for i, branch in enumerate(self.branches):
            input_tensor = inputs[i]
            # Create a mask to identify NaN values
            nan_mask = torch.isnan(input_tensor)
            zero_mask = torch.eq(input_tensor, 0.0)
            #nan_mask = nan_mask.unsqueeze(1).expand_as(inputs[i])

            # Set NaN values to 0 using the mask
            masked_input = input_tensor.clone()
            masked_input[nan_mask] = 0.0
            masked_input[zero_mask] = 0.01
            
            # Pass the masked input through the branch
            branch_output = branch(masked_input)
            branch_outputs.append(branch_output)

        #print("branch_outputs: ", tensor_info(branch_outputs), branch_outputs)
        combined_output = torch.cat(branch_outputs, dim=0)
        #print("combined_output: ", tensor_info(combined_output), combined_output)
        final_output = self.final_layers(combined_output)
        #print("final_output: ", tensor_info(final_output), final_output)
        return final_output.squeeze()
    
    def rand_weights(self):
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                init.xavier_uniform_(layer.weight)
                init.constant_(layer.bias, random.uniform(0, 0.2))



def train(self, mini_sample):
    #adapt the reward normalization to the mean and stabw of the last 
    mean=np.mean(np.array(self.normalizer))
    std=np.std(np.array(self.normalizer))
    old_states_batch, actions_batch, rewards_batch, new_states_batch = zip(*mini_sample)
   
    # Filter out None values from new_states_batch
    new_states_batch = [states for states in new_states_batch if states is not None]
    # Convert list of arrays of tensors to list of tensors
    old_states = [torch.stack([state.clone().detach() for state in states]) for states in old_states_batch]
    new_states = [torch.stack([state.clone().detach() for state in states]) for states in new_states_batch]
        
    rewards = torch.tensor(rewards_batch, dtype=torch.float)
    actions = torch.tensor(actions_batch, dtype=torch.int)
    self.logger.debug("States and rewards translated to tensors.")
    
    actor_loss_list=[]
    critic_loss_list=[]
    entropy_loss_list=[]
    
    for old_state, action, reward, new_state in zip(old_states, actions, rewards, new_states):
        norm_reward= (reward-mean)/(std+0.000000001)
        #print("norm_reward: ", norm_reward)
        action_probs = self.model(old_state)
        value=self.target_model(old_state)
        #print("action_probs: ", action_probs, "value: ", value)
        action_probs = F.softmax(action_probs, dim=-1)
        #print("action_probs: ", action_probs)
        action_log_probs = F.log_softmax(action_probs, dim=-1)
        #print("action_log_probs: ", action_log_probs)

        new_action_probs= self.model(new_state)
        new_value= self.target_model(new_state)
        new_action_probs = F.softmax(new_action_probs, dim=-1)
        new_action_log_probs = F.log_softmax(new_action_probs, dim=-1)

        #print("action: ", action)
        action_log_probs = action_log_probs[int(action)]
        #print("action_log_probs: ", action_log_probs)

        #print("reward: ", reward, "value :", value)
        ratios = torch.exp(action_log_probs - new_action_log_probs)
        advantage = norm_reward - value
        #print("advantage: ", advantage)
        #print("ratios: ", ratios)

        surrogate1 = ratios * advantage
        surrogate2 = torch.clamp(ratios, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantage
        #print(surrogate1, surrogate2)

        actor_loss = -torch.min(surrogate1, surrogate2).mean()
        actor_loss_list.append(actor_loss)
        critic_loss = self.criterion(value, norm_reward)
        critic_loss_list.append(critic_loss)
        entropy_loss = -(action_probs * action_log_probs).sum(-1).mean()
        entropy_loss_list.append(entropy_loss)
        #print("actor_loss: ", actor_loss, "critic_loss: ", critic_loss, "entropy_loss: ", entropy_loss)

    loss = sum(actor_loss_list)/len(actor_loss_list) + self.value_coeff * sum(critic_loss_list)/len(critic_loss_list) - self.entropy_coeff * sum(entropy_loss_list)/len(entropy_loss_list)
    #print(intermediate_loss)
    self.optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_value)
    nn.utils.clip_grad_norm_(self.target_model.parameters(), self.clip_value)
    self.optimizer.step()
        
    self.game_loss=((self.game_loss[1]*self.game_loss[0]+loss*len(mini_sample))/(self.game_loss[1]+len(mini_sample)),self.game_loss[1]+len(mini_sample))
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


    #setup rule_based_agent:
    self.logger.debug('Successfully entered setup code')
    np.random.seed()
    # Fixed length FIFO queues to avoid repeating the same actions
    self.bomb_history = deque([], 5)
    self.coordinate_history = deque([], 20)
    # While this timer is positive, agent will not hunt/attack opponents
    self.ignore_others_timer = 0
    self.current_round = 0


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
    self.lr=0.01
    self.clip_epsilon = 0.2
    self.value_coeff = 0.5
    self.entropy_coeff = 0.01
    self.clip_value=1.0
    self.input_sizes = [7, 7, 7, 7]  # Size of each input np.array
    self.hidden_layer_sizes = [14]  # Adjust hidden layer size
    self.output_size = len(ACTIONS)  # Number of output values
    self.slope=0.01
    self.final_hidden_layer_size=self.output_size*self.hidden_layer_sizes[len(self.hidden_layer_sizes)-1]*2
    self.model = MultiBranchNet(self.input_sizes, self.hidden_layer_sizes, self.final_hidden_layer_size, self.output_size, self.slope)
    self.target_model = MultiBranchNet(self.input_sizes, self.hidden_layer_sizes, self.final_hidden_layer_size, 1, self.slope)
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
            self.target_model.load_state_dict(checkpoint['target_model_state'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state'])
            old_command=checkpoint['command']
            if old_command==self.command:
                self.memory=checkpoint['memory']
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
    torch.save(self.target_model, os.path.join(model_folder_path, Model_Name + "_target" + ".pt"))
    torch.save({'model_state': self.model.state_dict(), 
                'target_model_state': self.target_model.state_dict(), 
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
                'plot_mean_loss':self.plot_mean_loss,
                'memory':self.memory
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


    #Good or bad action
    expert_action=act(self, old_game_state)
    #print("Pre:", self_action, expert_action)
    if self_action==expert_action:
        #print("  ", self_action, expert_action, "+100")
        add_reward=+100
    else:
        add_reward=-100
        #print("  ", self_action, expert_action, "-100")
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


#Copy of the rule_based_agent:

def look_for_targets(free_space, start, targets, logger=None):
    """Find direction of closest target that can be reached via free tiles.

    Performs a breadth-first search of the reachable free tiles until a target is encountered.
    If no target can be reached, the path that takes the agent closest to any target is chosen.

    Args:
        free_space: Boolean numpy array. True for free tiles and False for obstacles.
        start: the coordinate from which to begin the search.
        targets: list or array holding the coordinates of all target tiles.
        logger: optional logger object for debugging.
    Returns:
        coordinate of first step towards closest target or towards tile closest to any target.
    """
    if len(targets) == 0: return None

    frontier = [start]
    parent_dict = {start: start}
    dist_so_far = {start: 0}
    best = start
    best_dist = np.sum(np.abs(np.subtract(targets, start)), axis=1).min()

    while len(frontier) > 0:
        current = frontier.pop(0)
        # Find distance from current position to all targets, track closest
        d = np.sum(np.abs(np.subtract(targets, current)), axis=1).min()
        if d + dist_so_far[current] <= best_dist:
            best = current
            best_dist = d + dist_so_far[current]
        if d == 0:
            # Found path to a target's exact position, mission accomplished!
            best = current
            break
        # Add unexplored free neighboring tiles to the queue in a random order
        x, y = current
        neighbors = [(x, y) for (x, y) in [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)] if free_space[x, y]]
        shuffle(neighbors)
        for neighbor in neighbors:
            if neighbor not in parent_dict:
                frontier.append(neighbor)
                parent_dict[neighbor] = current
                dist_so_far[neighbor] = dist_so_far[current] + 1
    if logger: logger.debug(f'Suitable target found at {best}')
    # Determine the first step towards the best found target tile
    current = best
    while True:
        if parent_dict[current] == start: return current
        current = parent_dict[current]

def reset_self(self):
    self.bomb_history = deque([], 5)
    self.coordinate_history = deque([], 20)
    # While this timer is positive, agent will not hunt/attack opponents
    self.ignore_others_timer = 0


def act(self, game_state):
    """
    Called each game step to determine the agent's next action.

    You can find out about the state of the game environment via game_state,
    which is a dictionary. Consult 'get_state_for_agent' in environment.py to see
    what it contains.
    """
    self.logger.info('Picking action according to rule set')
    # Check if we are in a different round
    if game_state["round"] != self.current_round:
        reset_self(self)
        self.current_round = game_state["round"]
    # Gather information about the game state
    arena = game_state['field']
    _, score, bombs_left, (x, y) = game_state['self']
    bombs = game_state['bombs']
    bomb_xys = [xy for (xy, t) in bombs]
    others = [xy for (n, s, b, xy) in game_state['others']]
    coins = game_state['coins']
    bomb_map = np.ones(arena.shape) * 5
    for (xb, yb), t in bombs:
        for (i, j) in [(xb + h, yb) for h in range(-3, 4)] + [(xb, yb + h) for h in range(-3, 4)]:
            if (0 < i < bomb_map.shape[0]) and (0 < j < bomb_map.shape[1]):
                bomb_map[i, j] = min(bomb_map[i, j], t)

    # If agent has been in the same location three times recently, it's a loop
    if self.coordinate_history.count((x, y)) > 2:
        self.ignore_others_timer = 5
    else:
        self.ignore_others_timer -= 1
    self.coordinate_history.append((x, y))

    # Check which moves make sense at all
    directions = [(x, y), (x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
    valid_tiles, valid_actions = [], []
    for d in directions:
        if ((arena[d] == 0) and
                (game_state['explosion_map'][d] < 1) and
                (bomb_map[d] > 0) and
                (not d in others) and
                (not d in bomb_xys)):
            valid_tiles.append(d)
    if (x - 1, y) in valid_tiles: valid_actions.append('LEFT')
    if (x + 1, y) in valid_tiles: valid_actions.append('RIGHT')
    if (x, y - 1) in valid_tiles: valid_actions.append('UP')
    if (x, y + 1) in valid_tiles: valid_actions.append('DOWN')
    if (x, y) in valid_tiles: valid_actions.append('WAIT')
    # Disallow the BOMB action if agent dropped a bomb in the same spot recently
    if (bombs_left > 0) and (x, y) not in self.bomb_history: valid_actions.append('BOMB')
    self.logger.debug(f'Valid actions: {valid_actions}')

    # Collect basic action proposals in a queue
    # Later on, the last added action that is also valid will be chosen
    action_ideas = ['UP', 'DOWN', 'LEFT', 'RIGHT']
    shuffle(action_ideas)

    # Compile a list of 'targets' the agent should head towards
    cols = range(1, arena.shape[0] - 1)
    rows = range(1, arena.shape[0] - 1)
    dead_ends = [(x, y) for x in cols for y in rows if (arena[x, y] == 0)
                 and ([arena[x + 1, y], arena[x - 1, y], arena[x, y + 1], arena[x, y - 1]].count(0) == 1)]
    crates = [(x, y) for x in cols for y in rows if (arena[x, y] == 1)]
    targets = coins + dead_ends + crates
    # Add other agents as targets if in hunting mode or no crates/coins left
    if self.ignore_others_timer <= 0 or (len(crates) + len(coins) == 0):
        targets.extend(others)

    # Exclude targets that are currently occupied by a bomb
    targets = [targets[i] for i in range(len(targets)) if targets[i] not in bomb_xys]

    # Take a step towards the most immediately interesting target
    free_space = arena == 0
    if self.ignore_others_timer > 0:
        for o in others:
            free_space[o] = False
    d = look_for_targets(free_space, (x, y), targets, self.logger)
    if d == (x, y - 1): action_ideas.append('UP')
    if d == (x, y + 1): action_ideas.append('DOWN')
    if d == (x - 1, y): action_ideas.append('LEFT')
    if d == (x + 1, y): action_ideas.append('RIGHT')
    if d is None:
        self.logger.debug('All targets gone, nothing to do anymore')
        action_ideas.append('WAIT')

    # Add proposal to drop a bomb if at dead end
    if (x, y) in dead_ends:
        action_ideas.append('BOMB')
    # Add proposal to drop a bomb if touching an opponent
    if len(others) > 0:
        if (min(abs(xy[0] - x) + abs(xy[1] - y) for xy in others)) <= 1:
            action_ideas.append('BOMB')
    # Add proposal to drop a bomb if arrived at target and touching crate
    if d == (x, y) and ([arena[x + 1, y], arena[x - 1, y], arena[x, y + 1], arena[x, y - 1]].count(1) > 0):
        action_ideas.append('BOMB')

    # Add proposal to run away from any nearby bomb about to blow
    for (xb, yb), t in bombs:
        if (xb == x) and (abs(yb - y) < 4):
            # Run away
            if (yb > y): action_ideas.append('UP')
            if (yb < y): action_ideas.append('DOWN')
            # If possible, turn a corner
            action_ideas.append('LEFT')
            action_ideas.append('RIGHT')
        if (yb == y) and (abs(xb - x) < 4):
            # Run away
            if (xb > x): action_ideas.append('LEFT')
            if (xb < x): action_ideas.append('RIGHT')
            # If possible, turn a corner
            action_ideas.append('UP')
            action_ideas.append('DOWN')
    # Try random direction if directly on top of a bomb
    for (xb, yb), t in bombs:
        if xb == x and yb == y:
            action_ideas.extend(action_ideas[:4])

    # Pick last action added to the proposals list that is also valid
    while len(action_ideas) > 0:
        a = action_ideas.pop()
        if a in valid_actions:
            # Keep track of chosen action for cycle detection
            if a == 'BOMB':
                self.bomb_history.append((x, y))

            return a