import pickle
from collections import namedtuple, deque
from typing import List

import events as e
from .feature_engineering import state_to_features

# This is only an example!
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 3  # keep only ... last transitions
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...


def setup_training(self):
    """
    Initialise self for training purpose.
    This is called after `setup` in callbacks.py.
    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # Example: Setup an array that will note transition tuples
    # (s, a, r, s')
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)


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
    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')

    self.model.train(old_game_state, self_action, new_game_state, reward_from_events(self, events))

    self.transitions.append(
        Transition(state_to_features(old_game_state), self_action, state_to_features(new_game_state),
                   reward_from_events(self, events)))


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.

    This is similar to reward_update. self.events will contain all events that
    occurred during your agent's final step.

    This is *one* of the places where you could update your agent.
    This is also a good place to store an agent that you updated.

    :param self: The same object that is passed to all of your callbacks.
    """
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')
    self.transitions.append(
        Transition(state_to_features(last_game_state), last_action, None, reward_from_events(self, events)))

    # Store the model
    with open("my-saved-model.pt", "wb") as file:
        pickle.dump(self.model, file)


def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    Self define reward for some special events.
    """
    game_rewards = {
            e.COIN_COLLECTED: 5.,
            e.KILLED_OPPONENT: 0.0,
            e.BOMB_DROPPED: -10.0,
            e.KILLED_SELF: -9.,
            e.MOVED_DOWN: .05,
            e.MOVED_UP: .05,
            e.MOVED_RIGHT: .05,
            e.MOVED_LEFT: .05,
            e.INVALID_ACTION: -.5,
            e.WAITED: -.3,
            e.CRATE_DESTROYED: -.2,
            e.SURVIVED_ROUND: 2,  # Small reward for surviving a step
    }

    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum
