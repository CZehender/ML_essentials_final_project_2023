import os
import pickle
import random
import math
import numpy as np
import torch


ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'BOMB', 'WAIT']
Running_model="CoinCollector1"
random_prob = .1
random_vec=[.2, .2, .2, .2, 0.1, 0.1]


def setup(self):
    """
    Setup your code. This is called once when loading each agent.
    Make sure that you prepare everything such that act(...) can be called.

    When in training mode, the separate `setup_training` in train.py is called
    after this method. This separation allows you to share your trained agent
    with other students, without revealing your training code.

    In this example, our model is a set of probabilities over actions
    that are is independent of the game state.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    file_path = os.path.join("./" + Running_model, Running_model + ".pt")
    if os.path.isfile(file_path):
        print("File exists: Will run the model " + Running_model)
    else:
        print("File does not exist: ", file_path)

    if self.train or not os.path.isfile(file_path):
        self.logger.info("Setting up model from scratch.")
        weights = np.random.rand(len(ACTIONS))
        self.model = weights / weights.sum()
    else:
        self.logger.info("Loading model from saved state.")
        with open(file_path, "rb") as file:
            self.model = torch.load(file)


def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """

    #tradoff exploration and exploitation
    if self.train and random.random() < random_prob:
        self.logger.debug("Choosing action purely at random.")
        # 80%: walk in any direction. 10% wait. 10% bomb.
        return np.random.choice(ACTIONS, p=random_vec)
    else:
        c_state = torch.tensor(state_to_features(game_state), dtype=torch.float)
        prediction = self.model(c_state)       
        print("Prediction: ", prediction)    
        print ("Item: ", torch.argmax(prediction).item())  
        return ACTIONS[torch.argmax(prediction).item()]
        
    self.logger.debug("Querying model for action.")
  


def state_to_features(game_state: dict) -> np.array:
    """
    *This is not a required function, but an idea to structure your code.*

    Converts the game state to the input of your model, i.e.
    a feature vector.

    You can find out about the state of the game environment via game_state,
    which is a dictionary. Consult 'get_state_for_agent' in environment.py to see
    what it contains.

    :param game_state:  A dictionary describing the current game board.
    :return: np.array
    """
    #Your own coordinates
    x_p=game_state['self'][3][0]
    y_p=game_state['self'][3][1]

    state = [
    x_p, y_p
    ]
    return np.array(state, dtype=float)



    # This is the dict before the game begins and after it ends
    if game_state is None:
        return None


# Helpers
def calculate_gravD_coins (own_x: int, own_y: int, game_state: dict) -> np.array:
    if len(game_state['coins'])>0:
        num_obj = len(game_state['coins'])
        sum_x = 0
        sum_y = 0
        density = 0
        for index, (x, y) in enumerate(game_state['coins']):
            sum_x += x
            sum_y += y
        center_of_gravity_x = sum_x / num_obj
        center_of_gravity_y = sum_y / num_obj
        for index, (x, y) in enumerate(game_state['bombs']):
            distance = np.sqrt((center_of_gravity_x - x) ** 2 + (center_of_gravity_y - y) ** 2)
            density += distance
        density = density / num_obj
        return np.array([own_x-center_of_gravity_x, own_y-center_of_gravity_y, density, float(num_obj)])
    else:
        return np.array([0.0, 0.0, 0.0, 0.0])

def calculate_gravD_bombs(own_x: int, own_y: int, game_state: dict) -> np.array:
    if len(game_state['bombs']) > 0:
        num_obj = 0
        sum_x = 0
        sum_y = 0
        density = 0
        for index, ((x, y), c) in enumerate(game_state['bombs']): 
            sum_x += x * (4 - c)
            sum_y += y * (4 - c)
            num_obj += c
        center_of_gravity_x = sum_x / num_obj
        center_of_gravity_y = sum_y / num_obj
        for index, ((x, y), c) in enumerate(game_state['bombs']):
            distance = np.sqrt((center_of_gravity_x - x) ** 2 + (center_of_gravity_y - y) ** 2)
            density += distance
        density = density / num_obj
        return np.array([own_x - center_of_gravity_x, own_y - center_of_gravity_y, density, float(num_obj)])
    else:
        return np.array([0.0, 0.0, 0.0, 0.0])

def calculate_gravD_others (own_x: int, own_y: int, game_state: dict) -> np.array:
    if len(game_state['others'])>0:
        num_obj = len(game_state['others'])
        sum_x = 0
        sum_y = 0
        density = 0
        for index, a, b, c, (x, y) in enumerate(game_state['others']):
            sum_x += x
            sum_y += y
        center_of_gravity_x = sum_x / num_obj
        center_of_gravity_y = sum_y / num_obj
        for index, a, b, c, (x, y) in enumerate(game_state['bombs']):
            distance = np.sqrt((center_of_gravity_x - x) ** 2 + (center_of_gravity_y - y) ** 2)
            density += distance
        density = density / num_obj
        return np.array([own_x-center_of_gravity_x, own_y-center_of_gravity_y, density, float(num_obj)])
    else:
        return np.array([0.0, 0.0, 0.0, 0.0])

def calculate_gravD_crates (own_x: int, own_y: int, game_state: dict) -> np.array:
    coord = np.column_stack(np.where(game_state['field'] == 1))
    list_crates = [(x, y) for x, y in coord]
    if len(list_crates)>0:
        num_obj = len(list_crates)
        sum_x = 0
        sum_y = 0
        density = 0
        for index, (x, y) in enumerate(list_crates):
            sum_x += x
            sum_y += y
        center_of_gravity_x = sum_x / num_obj
        center_of_gravity_y = sum_y / num_obj
        for index, (x, y) in enumerate(list_crates):
            density += np.sqrt((center_of_gravity_x - x) ** 2 + (center_of_gravity_y - y) ** 2)
        density = density/num_obj
        return np.array([own_x-center_of_gravity_x, own_y-center_of_gravity_y, density, float(num_obj)])
    else:
        return np.array([0.0, 0.0, 0.0, 0.0])

def check_tile(x: int, y: int, game_state: dict) -> float:
    if game_state['field'][x][y] != 0:
        return float(game_state['field'][x][y])
    else:
        if game_state['explosion_map'][x][y] != 0:
            return 3.0
        else:
            for item in game_state['others']:
                (x_, y_) = item[3]
                if (x_, y_) == (x, y):
                    return 4.0
            for item in game_state['bombs']:
                (x_, y_) = item[0]
                if (x_, y_) == (x, y):
                    return 2.0
            for item in game_state['coins']:
                (x_, y_) = item
                if (x_, y_) == (x, y):
                    return 5.0
            return 0.0

# distance_to_center of field minus max distance(own_x, own_y):
def distance_to_wall(x: int, y: int) -> float:
    return np.sqrt(8 ** 2+8 ** 2)-np.sqrt((8-x) ** 2+(8-y) ** 2)

def feature1(x_p: int, y_p: int, game_state: dict) -> np.array:
    return np.array([check_tile(x_p-1, y_p, game_state), check_tile(x_p+1, y_p, game_state), check_tile(x_p, y_p+1, game_state), check_tile(x_p, y_p-1, game_state)])