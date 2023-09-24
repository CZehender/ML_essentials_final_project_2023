import os
import pickle
import random
import math
import numpy as np
import torch
import torch.nn as nn

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'BOMB', 'WAIT']
Running_model="Coin"
random_prob = 0.5
random_vec=[0.18, 0.18, 0.18, 0.18, 0.18, 0.1]

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

    if not os.path.isfile(file_path):
        self.logger.info("Setting up model from scratch.")
        print("Setting up model from scratch.")
        weights = np.random.rand(len(ACTIONS))
        self.model = weights / weights.sum()
    else:
        self.logger.info("Loading model from saved state.")
        print("Loading model from saved state.")
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
        c_state = state_to_features(game_state)
        prediction = self.model(c_state) 
        #print("Prediction: ", prediction)    
        #print ("Item: ", torch.argmax(prediction).item(), "Action: ", ACTIONS[torch.argmax(prediction).item()])  
        return ACTIONS[torch.argmax(prediction).item()]
              
    self.logger.debug("Querying model for action.")


def state_to_features(game_state: dict) -> list:
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
    # This is the dict before the game begins and after it ends
    if game_state is None:
        #return None
        return [torch.tensor([float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan')], dtype=torch.float),
                torch.tensor([float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan')], dtype=torch.float),
                torch.tensor([float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan')], dtype=torch.float),]
    else:
        #Your own coordinates
        x_p=game_state['self'][3][0]
        y_p=game_state['self'][3][1]

        state = [
       
        # Score branch
        calculate_gravD_coins (x_p, y_p, game_state),

        # Survival branch
        calculate_gravD_bombs (x_p, y_p, game_state),
        
        # Hunting branch
        calculate_gravD_others (x_p, y_p, game_state),
        ]
        #print("State vektor: ", state) 
        return state


def check_tile(x: int, y: int, game_state: dict) -> float:
    if game_state['field'][x][y] != 0:
        return 1 #crates=1 and stones=1
    else:
        if game_state['explosion_map'][x][y] != 0:
            return 1 #explosion=1
        if viability_check (x, y, game_state) == 1:
            return 1 #death trap=1
        else:
            for item in game_state['others']:
                (x_, y_) = item[3]
                if (x_, y_) == (x, y):
                    return 1.  #others=1
            for item in game_state['bombs']:
                (x_, y_) = item[0]
                if (x_, y_) == (x, y):
                    return 1  #bombs=1
            return 0.0  #free=0


# Helpers
def calculate_gravD_coins (own_x: int, own_y: int, game_state: dict) -> torch.tensor:
    count_obj = len(game_state['coins'])
    if count_obj>0 :
        distances = [np.sqrt((x - own_x) ** 2 + (y - own_y) ** 2) for (x, y) in game_state['coins']]
        index=distances.index(min(distances))
        closest_obj = game_state['coins'][index]
        obj_x = own_x-closest_obj[0]
        obj_y = own_y-closest_obj[1]
        if count_obj>1:
            num_obj = count_obj-1
            sum_x = 0.
            sum_y = 0.
            density = 0.
            for i, (x, y) in enumerate(game_state['coins']):
                if i != index:
                    sum_x += x
                    sum_y += y
            center_of_gravity_x = sum_x / num_obj
            center_of_gravity_y = sum_y / num_obj
            for i, (x, y) in enumerate(game_state['coins']):
                if i != index:
                    distance = np.sqrt((center_of_gravity_x - x) ** 2 + (center_of_gravity_y - y) ** 2)
                    density += distance
            density = density/num_obj
            relativ_center_of_gravity_x = own_x-center_of_gravity_x
            relativ_center_of_gravity_y = own_y-center_of_gravity_y
        else: 
            relativ_center_of_gravity_x= float('nan')
            relativ_center_of_gravity_y= float('nan')
            density= float('nan')
    else:
        obj_x = float('nan')
        obj_y = float('nan')
        relativ_center_of_gravity_x= float('nan')
        relativ_center_of_gravity_y= float('nan')
        density= float('nan')

    coord = np.column_stack(np.where(game_state['field'] == 1))
    list_crates = [(x, y) for x, y in coord]
    if len(list_crates)>0 :
        distances_c = [np.sqrt((x - own_x) ** 2 + (y - own_y) ** 2) for (x, y) in list_crates]
        index_c=distances.index(min(distances_c))
        closest_obj_c = list_crates[index_c]
        obj_x_c = own_x-closest_obj_c[0]
        obj_y_c = own_y-closest_obj_c[1]
    else:
        obj_x_c = float('nan')
        obj_y_c = float('nan')
    return torch.tensor([check_tile(own_x-1, own_y, game_state), check_tile(own_x+1, own_y, game_state), check_tile(own_x, own_y+1, game_state), check_tile(own_x, own_y-1, game_state), obj_x_c, obj_y_c, obj_x, obj_y, relativ_center_of_gravity_x, relativ_center_of_gravity_y, density], dtype=torch.float)

'''
# distance_to_center of field minus max distance(own_x, own_y):
def distance_to_wall(x: int, y: int) -> float:
    return np.sqrt(8 ** 2+8 ** 2)-np.sqrt((8-x) ** 2+(8-y) ** 2)
'''

def calculate_gravD_others (own_x: int, own_y: int, game_state: dict) -> torch.tensor:
    count_obj = len(game_state['others'])
    if count_obj>0 :
        distances = [np.sqrt((x - own_x) ** 2 + (y - own_y) ** 2) for a, b, c, (x, y) in game_state['others']]
        index=distances.index(min(distances))
        closest_obj = game_state['others'][index]
        obj_x = own_x-closest_obj[3][0]
        obj_y = own_y-closest_obj[3][1]
        if count_obj>1:
            num_obj = count_obj-1
            sum_x = 0.
            sum_y = 0.
            density = 0.
            for i, (a, b, c, (x, y)) in enumerate(game_state['others']):
                if i != index:
                    sum_x += x
                    sum_y += y
            center_of_gravity_x = sum_x / num_obj
            center_of_gravity_y = sum_y / num_obj
            for i, (a, b, c, (x, y)) in enumerate(game_state['others']):
                if i != index:
                    distance = np.sqrt((center_of_gravity_x - x) ** 2 + (center_of_gravity_y - y) ** 2)
                    density += distance
            density = density/num_obj
            relativ_center_of_gravity_x = own_x-center_of_gravity_x
            relativ_center_of_gravity_y = own_y-center_of_gravity_y
        else: 
            relativ_center_of_gravity_x= float('nan')
            relativ_center_of_gravity_y= float('nan')
            density= float('nan')
    else:
        obj_x = float('nan')
        obj_y = float('nan')
        relativ_center_of_gravity_x= float('nan')
        relativ_center_of_gravity_y= float('nan')
        density= float('nan')
    return torch.tensor([check_tile(own_x-1, own_y, game_state), check_tile(own_x+1, own_y, game_state), check_tile(own_x, own_y+1, game_state), check_tile(own_x, own_y-1, game_state), obj_x, obj_y, relativ_center_of_gravity_x, relativ_center_of_gravity_y, density, int(game_state['self'][2]), viability_check(own_x, own_y, game_state)], dtype=torch.float)

#Checks is moving is necessary or not
def viability_check(own_x: int, own_y: int, game_state: dict)-> float:
    for i, ((x, y), c) in enumerate(game_state['bombs']):
        if (own_y==y and -(3-c)<=own_x-x<0.0 and game_state['field'][x-1][y] != -1) or (own_y==y and (3-c)>=own_x-x>0.0 and game_state['field'][x+1][y] != -1) or (own_x==x and -(3-c)<=own_y-y<0.0 and game_state['field'][x][y-1] != -1) or (own_x==x and (3-c)>=own_y-y>0.0 and game_state['field'][x][y+1] != -1) or (own_y==y and own_x==x):
            return 1.0
    return 0.0

def calculate_gravD_bombs (own_x: int, own_y: int, game_state: dict) -> torch.tensor:
    count_obj = len(game_state['bombs'])
    viability = viability_check(own_x, own_y, game_state)
    if count_obj>0 :
        distances = [np.sqrt((x - own_x) ** 2 + (y - own_y) ** 2) for (x, y), c in game_state['bombs']]
        index=distances.index(min(distances))
        closest_obj = game_state['bombs'][index]
        obj_x = own_x-closest_obj[0][0]
        obj_y = own_y-closest_obj[0][1]
        timer = closest_obj[1]
        if count_obj>1:
            num_obj = 0.
            sum_x = 0.
            sum_y = 0.
            density = 0.
            for i, ((x, y), c) in enumerate(game_state['bombs']):
                if i != index:
                    sum_x += x
                    sum_y += y
                    num_obj += c
            center_of_gravity_x = sum_x / (count_obj-1)
            center_of_gravity_y = sum_y / (count_obj-1)
            mean_timer=num_obj/(count_obj-1)
            relativ_center_of_gravity_x = own_x-center_of_gravity_x
            relativ_center_of_gravity_y = own_y-center_of_gravity_y
        else: 
            relativ_center_of_gravity_x= float('nan')
            relativ_center_of_gravity_y= float('nan')
            mean_timer= float('nan')
    else:
        obj_x = float('nan')
        obj_y = float('nan')
        timer= float('nan')
        relativ_center_of_gravity_x= float('nan')
        relativ_center_of_gravity_y= float('nan')
        mean_timer= float('nan')
    return torch.tensor([check_tile(own_x-1, own_y, game_state), check_tile(own_x+1, own_y, game_state), check_tile(own_x, own_y+1, game_state), check_tile(own_x, own_y-1, game_state), obj_x, obj_y, timer, relativ_center_of_gravity_x, relativ_center_of_gravity_y, mean_timer, viability], dtype=torch.float)


