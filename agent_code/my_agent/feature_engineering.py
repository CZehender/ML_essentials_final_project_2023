import numpy as np
import settings as settings


def get_explosion_area(x, y, explosion_radius):
    explosion_coordinates = []

    # Iterate over the up, down, left, and right directions
    for dx, dy in [(0, 1), (0, -1), (-1, 0), (1, 0)]:
        for distance in range(1, explosion_radius + 1):
            # Calculate the coordinates for this step in the current direction
            new_x = x + dx * distance
            new_y = y + dy * distance

            # Add the coordinates to the explosion area
            explosion_coordinates.append((new_x, new_y))

    return explosion_coordinates


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
    # This is the dict before the game begins and after it ends
    if game_state is None:
        return None

    field_shape = game_state["field"].shape

    # Create feature Matrix
    feature_matrix = np.zeros((4,) + field_shape, dtype=np.double)

    # Coins feature in channel 0
    for (x, y) in game_state["coins"]:
        feature_matrix[0, x, y] = 1

    # Walls feature in channel 2
    feature_matrix[2, :, :] = np.where(game_state["field"] == -1, 1, 0)

    # Position of user feature in channel 1
    _, _, _, (x, y) = game_state["self"]
    feature_matrix[1, x, y] = 1

    # Position of dangerous area in channel 3
    for (x, y), _ in game_state["bombs"]:
        dangerous_area = get_explosion_area(x, y, settings.BOMB_POWER)
        for (x_exp, y_exp) in dangerous_area:
            if 0 <= x_exp < 17 and 0 <= y_exp < 17:
                feature_matrix[3, x_exp, y_exp] = -1

    # return the feature_map(batch_size, channels, height, width) (-1, 4, 17, 17)
    return feature_matrix
