import os
import pickle
import random
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'BOMB', 'WAIT']
Running_model="Coin"
random_prob = 0.25
random_vec=[0.2, 0.2, 0.2, 0.2, 0.2, 0.0]

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
        #print(c_state)
        prediction, v = self.model(c_state) 
        #print("Prediction: ", prediction)    
        #print ("Item: ", torch.argmax(prediction).item(), "Action: ", ACTIONS[torch.argmax(prediction).item()])  
        action = ACTIONS[torch.argmax(prediction).item()]
        return action
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
        return torch.tensor([float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan')], dtype=torch.float)
    else:
        #Your own coordinates
        x_p=game_state['self'][3][0]
        y_p=game_state['self'][3][1]
        #print("Coordinaten", x_p, y_p)
        #print(game_state['field'])
        state = feature1(x_p, y_p, game_state)

        #print("State vektor: ", state) 
        return state


def check_tile(x: int, y: int, game_state: dict) -> float:
    if game_state['field'][x][y] != 0:
        #print("check_tile", x, y, "hindernis")
        return 0 #crates=1 and stones=1
    else:
        if game_state['explosion_map'][x][y] != 0:
            #print("check_tile", x, y, "explosion")
            return 0 #explosion=1
        if viability_check (x, y, game_state) != 1:
            #print("check_tile", x, y, "not viable")
            return 0 #death trap=1
        else:
            for item in game_state['others']:
                (x_, y_) = item[3]
                if (x_, y_) == (x, y):
                    #print("check_tile", x, y, "others")
                    return 0  #others=1
            for item in game_state['bombs']:
                (x_, y_) = item[0]
                if (x_, y_) == (x, y):
                    #print("check_tile", x, y, "bombs")
                    return 0  #bombs=1
            return 1  #free=0


# Helpers
#7 features
def calculate_gravD_coins (own_x: int, own_y: int, game_state: dict) -> torch.tensor:
    count_obj = len(game_state['coins'])
    if count_obj>0 :
        distances = [np.sqrt((x - own_x) ** 2 + (y - own_y) ** 2) for (x, y) in game_state['coins']]
        index=distances.index(min(distances))
        closest_obj = game_state['coins'][index]
        obj_x = own_x-closest_obj[0]
        obj_y = own_y-closest_obj[1]
    else:
        obj_x = float('nan')
        obj_y = float('nan')

    coord = np.column_stack(np.where(game_state['field'] == 1))
    list_crates = [(x, y) for x, y in coord]
    if len(list_crates)>0 :
        distances_c = [np.sqrt((x - own_x) ** 2 + (y - own_y) ** 2) for (x, y) in list_crates]
        index_c=distances_c.index(min(distances_c))
        closest_obj_c = list_crates[index_c]
        obj_x_c = own_x-closest_obj_c[0]
        obj_y_c = own_y-closest_obj_c[1]
    else:
        obj_x_c = float('nan')
        obj_y_c = float('nan')
    return torch.tensor([obj_x_c, obj_y_c, obj_x, obj_y], dtype=torch.float)

def check_coin(_x: int, _y: int, tile_x: int, tile_y: int, game_state: dict) -> float:
    if not math.isnan(calculate_gravD_coins (_x, _y, game_state)[2]):
        tile_x_coin= calculate_gravD_coins (tile_x, tile_y, game_state)[2]
        tile_y_coin= calculate_gravD_coins (tile_x, tile_y, game_state)[3]
        obj_x_coin= calculate_gravD_coins (_x, _y, game_state)[2]
        obj_y_coin= calculate_gravD_coins (_x, _y, game_state)[3]
        if np.sqrt((tile_x_coin) ** 2 + (tile_y_coin) ** 2)<np.sqrt((obj_x_coin) ** 2 + (obj_y_coin) ** 2):
            #print("Go to coin with ",tile_x, tile_y)
            return 1
        else:
            return 0
    else:
        if not math.isnan(calculate_gravD_coins (_x, _y, game_state)[0]):
            tile_x_crate= calculate_gravD_coins (tile_x, tile_y, game_state)[0]
            tile_y_crate= calculate_gravD_coins (tile_x, tile_y, game_state)[1]
            obj_x_crate= calculate_gravD_coins (_x, _y, game_state)[0]
            obj_y_crate= calculate_gravD_coins (_x, _y, game_state)[1]
            if np.sqrt((tile_x_crate) ** 2 + (tile_y_crate) ** 2)<np.sqrt((obj_x_crate) ** 2 + (obj_y_crate) ** 2):
                #print("Go to crate with ",tile_x, tile_y)
                return 1
            else:
                return 0
        else:
            return 0

def check_others(_x: int, _y: int, tile_x: int, tile_y: int, game_state: dict) -> float:
    if not math.isnan(calculate_gravD_others (_x, _y, game_state)[0]):
        tile_x_others= calculate_gravD_others (tile_x, tile_y, game_state)[0]
        tile_y_others= calculate_gravD_others (tile_x, tile_y, game_state)[1]
        obj_x_others= calculate_gravD_others (_x, _y, game_state)[0]
        obj_y_others= calculate_gravD_others (_x, _y, game_state)[1]
        if np.sqrt((tile_x_others) ** 2 + (tile_y_others) ** 2)<np.sqrt((obj_x_others) ** 2 + (obj_y_others) ** 2):
            #print("Go to others with ",tile_x, tile_y)
            return 1
        else:
            return 0
    else:
        return 0


def bomb_sense(own_x: int, own_y: int, game_state: dict) -> float:
    check=0
    for item in game_state['others']:
                (x_, y_) = item[3]
                if (x_, y_) == (own_x-1, own_y) or (x_, y_) == (own_x+1, own_y) or (x_, y_) == (own_x, own_y-1) or (x_, y_) == (own_x, own_y+1):
                    check=1
    counter=0
    if game_state['field'][own_x+1][own_y] != 0:
        counter+=1
    if game_state['field'][own_x][own_y-1] != 0:
        counter+=1
    if game_state['field'][own_x][own_y+1] != 0:
        counter+=1
    if game_state['field'][own_x-1][own_y] != 0:
        counter+=1
    #print("check", check, "counter", counter)
    if ((game_state['self'][2])==True and viability_check(own_x, own_y, game_state)==1 and (check_kill_likelyhood(own_x, own_y, game_state)==0 or counter>=3 or check==1)):
            bomb_sense=1
    else:
        bomb_sense=0
    return bomb_sense

def wait_sense(own_x: int, own_y: int, game_state: dict) -> float:
    if viability_check (own_x, own_y, game_state) != 1:
        wait_sense=0
    else:
        wait_sense=1
    return wait_sense


def check_kill_likelyhood(_x: int, _y: int, game_state: dict) -> float:
    if not math.isnan(calculate_gravD_others (_x, _y, game_state)[0]):
        obj_x= calculate_gravD_others (_x, _y, game_state)[0]
        obj_y= calculate_gravD_others (_x, _y, game_state)[1]
        if np.sqrt((obj_x) ** 2 + (obj_y) ** 2)<=4:
            if calculate_gravD_others (_x, _y, game_state)[2]*0.75>ways_to_die(_x, _y, game_state):
                return 0
            else:
                return 1
    else:
        return 1


#7 features
def calculate_gravD_others (own_x: int, own_y: int, game_state: dict) -> torch.tensor:
    count_obj = len(game_state['others'])
    ways_to_die_other=0
    if count_obj>0 :
        distances = [np.sqrt((x - own_x) ** 2 + (y - own_y) ** 2) for a, b, c, (x, y) in game_state['others']]
        index=distances.index(min(distances))
        closest_obj = game_state['others'][index]
        obj_x = own_x-closest_obj[3][0]
        obj_y = own_y-closest_obj[3][1]
        ways_to_die_other=ways_to_die(game_state['others'][index][3][0], game_state['others'][index][3][1], game_state)       
    else:
        obj_x = float('nan')
        obj_y = float('nan')
        ways_to_die_c=float('nan')
    return torch.tensor([obj_x, obj_y, ways_to_die_other], dtype=torch.float)

#Checks is moving is necessary or not
def viability_check(own_x: int, own_y: int, game_state: dict)-> float:
    for i, ((x, y), c) in enumerate(game_state['bombs']):
        #print("Bomb", x, y, c)
        if (own_y==y and -(3-c)<=own_x-x<0.0 and game_state['field'][x-1][y] != -1) or (own_y==y and (3-c)>=own_x-x>0.0 and game_state['field'][x+1][y] != -1) or (own_x==x and -(3-c)<=own_y-y<0.0 and game_state['field'][x][y-1] != -1) or (own_x==x and (3-c)>=own_y-y>0.0 and game_state['field'][x][y+1] != -1) or (own_y==y and own_x==x):
            return 0
    return 1


#7 features #left, #right, #up, #down, #wait, #bomb         'UP', 'RIGHT', 'DOWN', 'LEFT', 'BOMB', 'WAIT'
def feature1(own_x: int, own_y: int, game_state: dict) -> torch.tensor:
    c1= check_tile(own_x, own_y+1, game_state) 
    c2= check_tile(own_x+1, own_y, game_state)
    c3= check_tile(own_x, own_y-1, game_state)
    c4= check_tile(own_x-1, own_y, game_state)
    c5= bomb_sense(own_x, own_y, game_state)
    c6= wait_sense(own_x, own_y, game_state)
    last_check=[c3, c2, c1, c4, c5, c6]
    #print("last_check", last_check)
    a1= check_others(own_x, own_y, own_x, own_y+1, game_state) + check_coin(own_x, own_y, own_x, own_y+1, game_state)*2+1
    a2= check_others(own_x, own_y, own_x+1, own_y, game_state) + check_coin(own_x, own_y, own_x+1, own_y, game_state)*2+1
    a3= check_others(own_x, own_y, own_x, own_y-1, game_state) + check_coin(own_x, own_y, own_x, own_y-1, game_state)*2+1
    a4= check_others(own_x, own_y, own_x-1, own_y, game_state) + check_coin(own_x, own_y, own_x-1, own_y, game_state)*2+1
    a5= bomb_sense(own_x, own_y, game_state)*3+1
    a6= wait_sense(own_x, own_y, game_state)+1
    multiplier=[a3, a2, a1, a4, a5 ,a6]
    #print("multiplier", multiplier)
    for i in range(0, len(last_check)):
        last_check[i]=last_check[i]*multiplier[i]
    item= last_check.index(max(last_check))
    null_array=[0,0,0,0,0,0]
    null_array[item]=1
    #print("null_array", null_array)
    return torch.tensor([null_array[0], null_array[1], null_array[2], null_array[3], null_array[4], null_array[5]], dtype=torch.float)



def ways_to_die(own_x: int, own_y: int, game_state: dict) -> float:
    counter=0
    for i, ((x, y), c) in enumerate(game_state['bombs']):
        if game_state['field'][x+1][y] != -1 and game_state['field'][x-1][y] != -1:
            x_range=range(x-3, x+4)
        if game_state['field'][x+1][y] == -1 and game_state['field'][x-1][y] != -1:
            x_range=range(x-3, x+1)
        if game_state['field'][x+1][y] != -1 and game_state['field'][x-1][y] == -1:
            x_range=range(x, x+4)
        if game_state['field'][x+1][y] == -1 and game_state['field'][x-1][y] == -1:
            x_range=range(x, x+1)
        if game_state['field'][x][y+1] != -1 and game_state['field'][x][y-1] != -1:
            y_range=range(y-3, y+4)
        if game_state['field'][x][y+1] == -1 and game_state['field'][x][y-1] != -1:
            y_range=range(y-3, y+1)
        if game_state['field'][x][y+1] != -1 and game_state['field'][x][y-1] == -1:
            y_range=range(y, y+4)
        if game_state['field'][x][y+1] == -1 and game_state['field'][x][y-1] == -1:
            y_range=range(y, y+1)

        b_tuples = [(i, y) for i in x_range] + [(x, j) for j in y_range]
        for i in range(0,c+1+1,1):
            own_x_range= range(own_x-(c+1)+i, own_x+c+1-i+1) #c+1 wegen explosions
            own_tuples=[(a, own_y+i) for a in own_x_range] + [(a, own_y-i) for a in own_x_range]
        counter+=sum(1 for b_tuple in b_tuples if b_tuple in own_tuples)

    n, s, b, (x, y)= game_state['self']
    if b==True:
        c=5
        if game_state['field'][x+1][y] != -1 and game_state['field'][x-1][y] != -1:
            x_range=range(x-3, x+4)
        if game_state['field'][x+1][y] == -1 and game_state['field'][x-1][y] != -1:
            x_range=range(x-3, x+1)
        if game_state['field'][x+1][y] != -1 and game_state['field'][x-1][y] == -1:
            x_range=range(x, x+4)
        if game_state['field'][x+1][y] == -1 and game_state['field'][x-1][y] == -1:
            x_range=range(x, x+1)
        if game_state['field'][x][y+1] != -1 and game_state['field'][x][y-1] != -1:
            y_range=range(y-3, y+4)
        if game_state['field'][x][y+1] == -1 and game_state['field'][x][y-1] != -1:
            y_range=range(y-3, y+1)
        if game_state['field'][x][y+1] != -1 and game_state['field'][x][y-1] == -1:
            y_range=range(y, y+4)
        if game_state['field'][x][y+1] == -1 and game_state['field'][x][y-1] == -1:
            y_range=range(y, y+1)

        b_tuples = [(i, y) for i in x_range] + [(x, j) for j in y_range]
        for i in range(0,c+1+1,1):
            own_x_range= range(own_x-(c+1)+i, own_x+c+1-i+1) #c+1 wegen explosions
            own_tuples=[(a, own_y+i) for a in own_x_range] + [(a, own_y-i) for a in own_x_range]
        counter+=sum(1 for b_tuple in b_tuples if b_tuple in own_tuples)

    
    for i, (n, s, b, (x, y)) in enumerate(game_state['others']):
        if b==True:
            c=5
            if game_state['field'][x+1][y] != -1 and game_state['field'][x-1][y] != -1:
                x_range=range(x-3, x+4)
            if game_state['field'][x+1][y] == -1 and game_state['field'][x-1][y] != -1:
                x_range=range(x-3, x+1)
            if game_state['field'][x+1][y] != -1 and game_state['field'][x-1][y] == -1:
                x_range=range(x, x+4)
            if game_state['field'][x+1][y] == -1 and game_state['field'][x-1][y] == -1:
                x_range=range(x, x+1)
            if game_state['field'][x][y+1] != -1 and game_state['field'][x][y-1] != -1:
                y_range=range(y-3, y+4)
            if game_state['field'][x][y+1] == -1 and game_state['field'][x][y-1] != -1:
                y_range=range(y-3, y+1)
            if game_state['field'][x][y+1] != -1 and game_state['field'][x][y-1] == -1:
                y_range=range(y, y+4)
            if game_state['field'][x][y+1] == -1 and game_state['field'][x][y-1] == -1:
                y_range=range(y, y+1)

            b_tuples = [(i, y) for i in x_range] + [(x, j) for j in y_range]
            for i in range(0,c+1+1,1):
                own_x_range= range(own_x-(c+1)+i, own_x+c+1-i+1) #c+1 wegen explosions
                own_tuples=[(a, own_y+i) for a in own_x_range] + [(a, own_y-i) for a in own_x_range]
            counter+=sum(1 for b_tuple in b_tuples if b_tuple in own_tuples)

    
    return float(counter)
