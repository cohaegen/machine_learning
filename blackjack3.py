import keras
from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf
import gym
import numpy as np


class DNNModel:
    def __init__(self):
        self._model = keras.models.Sequential()
        self._model.add(keras.layers.InputLayer((46,)))
        self._model.add(keras.layers.Dense(2000))
        self._model.add(keras.layers.Dense(200))
        self._model.add(keras.layers.Dense(2, activation='tanh'))
        self._model.compile(optimizer=keras.optimizers.Adam(lr=1e-5), loss='mse')
        
    def _convert_inputs(self, data):
        '''Convert raw data (e.g., integers) to inputs to the neural net'''
        data = np.array(data)
        p = keras.utils.to_categorical(data[:,0], 32)
        d = keras.utils.to_categorical(data[:,1], 12)
        a = keras.utils.to_categorical(data[:,2], 2)
        return np.concatenate([p,d,a], 1)
        
    def train(self, X, y):
        # convert training data to categorical one-hot arrays
        self._model.train_on_batch(self._convert_inputs(X), np.array(y))
        
    def fit(self, X, y, epochs=None):
        self._model.fit(self._convert_inputs(X), np.array(y), epochs=epochs)
        
    def predict(self, X):
        return self._model.predict(self._convert_inputs(X))
        
    def __getitem__(self, x):
        return self._model.predict(self._convert_inputs([x]))[0]
        
    def __setitem__(self, x, y):
        self.train([x], [y])
        
        
class TableModel():
    def __init__(self):
        self._table = np.zeros((32, 12, 2, 2))
        
    def train(self, X, y):
        for idx, x in enumerate(X):
            self[x] = y[idx]
        
    def fit(self, X, y, epochs=None):
        self.train(X, y)
        
    def predict(self, X):
        y = np.zeros((X.shape[0], 2))
        for idx, x in enumerate(X):
            y[idx] = self[x]
        return y
        
    def __getitem__(self, x):
        player, dealer, ace = x
        ace = int(ace)
        return self._table[(player, dealer, ace)]
        
    def __setitem__(self, x, y):
        player, dealer, ace = x
        ace = int(ace)
        self._table[(player, dealer, ace)] = np.array(y)
        
     
def play(env, Q):
    '''Play a game using policy Q and return the reward'''
    s = env.reset()
    s1 = s
    reward = 0
    total_reward = 0
    done = False
    while(done is False):
        action = np.argmax(Q[s])
        s1, reward, done, _ = env.step(action)
        total_reward += reward
        s = s1
    return total_reward
    

NUM_GAMES = 10000
GAME_HISTORY_SIZE = 100
MINIBATCH_SIZE = 30
Q = DNNModel()
Qt = TableModel()
game_history_inputs = np.zeros((GAME_HISTORY_SIZE, 3), dtype='int8')
game_history_rewards = np.zeros((GAME_HISTORY_SIZE, 2), dtype='float')
env = gym.make('Blackjack-v0')

# The following resulted in mean performance of -0.05266 over 100k games
# Qt = train_Q(Qt, alpha=0.0001, num_episodes=2000000)
# The following resulted in mean performance of -0.06795 over 100k games
# Q = train_Q(Q, alpha=0.0001, num_episodes=2000000)

def train_Q(Q, alpha=0.01, gamma=0.95, num_episodes=NUM_GAMES, initial_action_random_prob=1.0, final_action_random_prob=1e-3):
    '''Train a Q function estimator'''
    # Calculate the exponential factor k that we need in order to match the 
    # initial and final probabilities given
    k = -np.log(final_action_random_prob/initial_action_random_prob)
    
    for e_i in range(num_episodes):
        if((e_i % 1000) == 0 or e_i == (num_episodes-1)):
            perf = np.mean([play(env,Q) for i in range(10000)])
            rand_prob = initial_action_random_prob*np.exp(-k*e_i/num_episodes)
            print('%d games played; random probability: %f; performance: %f' % (e_i, rand_prob, perf))
            
        player, dealer, ace = env.reset()
        done = False
        while(False == done):
            Q_predict = Q[(player, dealer, ace)]
            if(np.random.random() < initial_action_random_prob*np.exp(-k*e_i/num_episodes)):
                a = env.action_space.sample()
            else:
                a = np.argmax(Q_predict)
            s1, reward, done, _ = env.step(a)
            if(done == True):
                Q_predict[a] = Q_predict[a] + alpha*(reward - Q_predict[a])
            else:
                Q_predict[a] = Q_predict[a] + alpha*(reward + gamma*np.max(Q[s1]) - Q_predict[a])
            update_random_history = np.random.choice(len(game_history_inputs), 1)
            game_history_inputs[update_random_history] = np.array([player, dealer, ace])
            game_history_rewards[update_random_history] = Q_predict
            # train on a mini-batch
            minibatch_indexes = np.random.choice(len(game_history_inputs), MINIBATCH_SIZE)
            Q.train(game_history_inputs[minibatch_indexes], game_history_rewards[minibatch_indexes])
            player, dealer, ace = s1
    return Q
    
    
# Q = DNNModel()
# train_Q(Q, gamma=0.99, num_episodes=200000)