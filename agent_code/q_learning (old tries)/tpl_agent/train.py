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
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

max_runs=10
Max_Memory = 10000
ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'BOMB', 'WAIT']
Model_Name = "CoinCollector1"
Batch_Size = 5
plot_rewards=[]
plot_mean_rewards=[]
total_reward=0
record=0
n_games=0
action_counter=0

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

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
    self.model = NeuralNet(2, 256, len(ACTIONS)) #input, hidden, output
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
    global action_counter
    action_counter += 1
    self.game_reward += reward_from_events(self, events)
    self.memory.append((state_to_features(old_game_state), action_converter(self_action), state_to_features(new_game_state), reward_from_events(self, events)))


    old_state = torch.unsqueeze(torch.tensor(state_to_features(old_game_state), dtype=torch.float), 0)
    new_state = torch.unsqueeze(torch.tensor(state_to_features(new_game_state), dtype=torch.float), 0)
    self_action = torch.unsqueeze(torch.tensor(action_converter(self_action), dtype=torch.float), 0)
    reward = torch.unsqueeze(torch.tensor(reward_from_events(self, events), dtype=torch.float), 0)

    self.logger.debug("States and targets translated to tensors.")

    pred = self.model(old_state)
    target = pred.clone()
    for idx in range(len(reward)):
        Q_new = reward[idx]
        Q_new = reward[idx]+self.gamma*torch.max(self.model(new_state[idx]))
        target[idx][torch.argmax(self_action).item()] = Q_new

    self.logger.debug("Q value calculated.")

    self.optimizer.zero_grad()
    loss = self.criterion(target, pred)
    loss.backward()

    self.optimizer.step()

    self.logger.debug("Model trained for this step")
    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')


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

    #Bestimme den finalen reward und schaue was dazu geführt hat gut abzuschneiden bzw. zu sterben etc.
    action_counter +=1
    final_reward=reward_from_events(self, events) - 0.2*action_counter
    self.game_reward +=final_reward
    
    n_games += 1
    if self.game_reward > record:
        record = self.game_reward
    print("Game", n_games, "Reward", self.game_reward, "Record", record)
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




    #Bestimme den finalen reward und schaue was dazu geführt hat gut abzuschneiden bzw. zu sterben etc.   
    if len(self.memory) > Batch_Size:
        sample = list(itertools.islice(self.memory, len(self.memory)-Batch_Size, len(self.memory)))
    else:
        sample = self.memory

    old_state, self_action, new_state, reward = zip(*sample)
    old_state = torch.tensor(np.array(old_state), dtype=torch.float)
    new_state = torch.tensor(np.array(new_state), dtype=torch.float)
    self_action = torch.tensor(np.array(self_action), dtype=torch.float)
    reward = torch.tensor(np.array(reward), dtype=torch.float)

    # Assign the last meaningful Batch-size entries new rewards according to the final outcome and reevaluate it
    reward[(len(self.memory)-Batch_Size):] += final_reward

    pred = self.model(old_state)
    target = pred.clone()
    for idx in range(len(reward)):
        Q_new = reward[idx]
        Q_new = reward[idx]+self.gamma*torch.max(self.model(new_state[idx]))
        target[idx][torch.argmax(self_action).item()] = Q_new

    self.logger.debug("Q value calculated.")

    self.optimizer.zero_grad()
    loss = self.criterion(target, pred)
    loss.backward()

    self.optimizer.step()

    self.logger.debug("Model trained for entire game")


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
        print("Model got trained with 10000 games. System exit now")
        
        sys.exit()


def reward_from_events(self, events: List[str]) -> float:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        e.COIN_COLLECTED: 10,
        e.KILLED_SELF: -20,
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum
