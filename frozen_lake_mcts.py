"""
Play Farama Gym's FrozenLake environment (https://gymnasium.farama.org/environments/toy_text/frozen_lake/)
Using Monte-Carlo Tree Search plus machine learning, based on Alpha Zero (arXiv:1712.01815 [cs.AI])

Not generated using any AI
"""
import gymnasium as gym
import numpy as np
import keras
from copy import deepcopy
from typing import Tuple

NUM_GAMES = 3  # How many games to play?
NUM_SIMULATIONS = 1000  # How many MCTS simulations to run per game?
C = 0.2  # exploration parameter
# Dirichlet noise to add to the exploration prior (should be inversely proportional to the number of action choices)
# The noise is a crucial components: the model always seems to converge to poor choices without it
# It always prefers "left" actions - maybe this is because left is choice 0, and when we do an argmax 
DIRICHLET_ALPHA = 0.6
LEARNING_RATE = 1e-2
MOVES = ['left', 'down', 'right', 'up']  # What moves correspond to each action value?


class Game:
    """
    A class that handles some convenient functions around the FrozenLake environment
    Keeps track of state and whether the game has finished or been truncated
    """
    def __init__(self, map_name='4x4', is_slippery=False, **kwargs) -> None:
        """Initialize a new environment and game"""
        self.env = gym.make('FrozenLake-v1', map_name=map_name, is_slippery=False, **kwargs)
        self.state, _ = self.env.reset()
        self.done = False
        self.rewards = 0.0
    
    def step(self, action: int) -> Tuple[int, float, bool]:
        """Take one action and return the new state, reward, and whether the game is done"""
        self.state, reward, done, truncated, _ = self.env.step(action)
        self.rewards += float(reward)
        self.done = done or truncated
        return self.state, float(reward), self.done
    
    def rollout(self) -> float:
        """
        Perform a random rollout
        This is used when a new game node is discovered in MCTS
        Not used in this machine learning version, which uses the ML model's value function instead,
        but I'm keeping this around anyway
        """
        while self.done is False:
            action = self.env.action_space.sample()
            self.step(action)
        return self.rewards
    

def create_model(observation_space: int, action_space: int) -> keras.Model:
    """
    Create an ML model of the game
    It takes one integer input, the current state of the game
    It outputs two NxM matrices, where N is the observation space and M is the action space:
    the first matrix is the estimated value (from 0 to 1) in each state for each action taken;
    the second matrix is the policy (recommended action) in each state for each action taken
    """
    inp = keras.layers.Input(shape=(1,))
    x = keras.layers.Embedding(observation_space, 16)(inp)
    x = keras.layers.Dense(64, activation='gelu')(x)[:, 0, :]
    value_head = keras.layers.Dense(action_space, activation='sigmoid', name='value')(x)
    policy_head = keras.layers.Dense(action_space, name='policy')(x)
    optimizer = keras.optimizers.AdamW(LEARNING_RATE)
    losses = ['mse', keras.losses.SparseCategoricalCrossentropy(from_logits=True)]
    model = keras.Model(inputs=inp, outputs=[value_head, policy_head])
    model.compile(optimizer, losses)
    return model


game = Game()
# Initialize visit and reward matrices, which track how many times we've taken a
# (state, action) transition. This is a little different than other MCTS implementations
# which use a tree, but it's doable here because the state space is so small.
visits_by_action = np.zeros((game.env.observation_space.n, game.env.action_space.n), dtype=np.int32)
rewards_by_action = np.zeros((game.env.observation_space.n, game.env.action_space.n))
model = create_model(game.env.observation_space.n, game.env.action_space.n)
model.summary()

for game_idx in range(NUM_GAMES):
    for sim_idx in range(NUM_SIMULATIONS):
        # Run a bunch of simulations for each game
        # At the end of each sim, update the ML model
        print(f"\rGame {game_idx} Iteration {sim_idx}", end='')
        game_clone = deepcopy(game)
        state = game_clone.state
        game_history = []
        reward = 0.0
        # Estimate rewards using our model
        model_rewards, model_policy = model(np.arange(game.env.observation_space.n)[:, np. newaxis])
        model_policy = keras.layers.Softmax()(model_policy)
        while game_clone.done is False:
            # Pick an action using UCT
            v = rewards_by_action[state, :] / (visits_by_action[state, :] + 1e-9)
            u = np.sqrt(visits_by_action[state, :].sum()) / (visits_by_action[state, :] + 1e-9)
            prior = model_policy[state, :].numpy()
            # Add some noise to the prior
            prior += np.random.dirichlet(alpha=[DIRICHLET_ALPHA]*game.env.action_space.n)
            # Calculate the PUCT values (from the Alpha Zero paper)
            puct = v + C*prior*u + 1e-9
            # Choose an action proportional to the PUCT values
            action = np.random.choice(np.arange(game.env.action_space.n), p=puct / puct.sum())
            game_history.append((state, action))
            # If we haven't taken this action before, estimate the value using the ML model and stop this sim
            if visits_by_action[state, action] == 0:
                # Get estimated reward from the model
                reward = model_rewards[state, action]
                break
            # Otherwise, go ahead and take the action we selected and keep going with the sim
            state, reward, _ = game_clone.step(action)
        for state, action in game_history:
            # Back-propagate visits and rewards to all the state, action pairs we visited in this sim
            visits_by_action[state, action] += 1
            rewards_by_action[state, action] += float(reward)
        # At the end of each simulation, update the ML model
        x = np.arange(game.env.observation_space.n).reshape((-1, 1))
        y_rewards = rewards_by_action / (visits_by_action + 1e-9)
        y_policy = visits_by_action.argmax(axis=1).reshape((-1, 1))
        model.train_on_batch(x, [y_rewards, y_policy])
print('')

x = np.arange(game.env.observation_space.n).reshape((-1, 1))
with np.printoptions(precision=2):
    rewards, policy = model(x)
    policy = keras.layers.Softmax()(policy)
    print(f'{rewards=}\n{policy=}')
# present as the best move to take for each square
print('Best moves for each FrozenLake square:')
print(np.array([MOVES[x] for x in policy.numpy().argmax(axis=1).tolist()]).reshape((4,4)))