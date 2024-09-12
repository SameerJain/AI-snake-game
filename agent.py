'''
Implements a Q-learning agent to play the classic Snake game. Uses a neural network to approximate the Q-value function, which helps the agent make decisions about which actions to take in different game states.

Components:
1. Agent: Manages the Q-learning process, including state representation, action selection, and learning.
2. Neural Network: Approximates the Q-value function.
3. Game Environment: A custom implementation of the Snake game for AI training.

The training process involves:
- Updating the Q-network based on experiences
- Balancing exploration and exploitation
- Visualizing the learning progress over time

Dependencies:
- torch: For neural network implementation
- numpy: For numerical operations
- collections: For efficient data structures
- matplotlib (implied through helper.plot): For visualizing training progress
'''
import os
import torch
import random
import numpy as np
from collections import deque
from game import SnakeGameAI, Direction, Point
from model import Linear_QNet, QTrainer
from helper import plot
''' 
may need to set this value to false depending on your machine. Test if it works or with True or False. This is a threading issue so don't be overly concerned.
'''
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

MAX_MEMORY = 100_000  # Max size of replay memory
BATCH_SIZE = 1000     # Number of experiences to sample for batch learning
'''
Below is the Learning Rate! Feel fre to play around with this and watch the AI speed up or slow down
'''
LR = 0.01            

class Agent:

    def __init__(self):
        
        self.n_games = 0
        self.epsilon = 0 # randomness
        self.gamma = 0.9 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        self.model = Linear_QNet(11, 256, 3) # init Q network 
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma) # init Q trainer, gamma)


    """
        Extract relevant game information and return a feature vector representing the game state.
        
        This method creates a binary feature vector that encodes:
        1. Danger in different directions
        2. Current moving direction
        3. Food location relative to the snake's head
        """  
    def get_state(self, game):
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)
        
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # Danger straight
            (dir_r and game.is_collision(point_r)) or 
            (dir_l and game.is_collision(point_l)) or 
            (dir_u and game.is_collision(point_u)) or 
            (dir_d and game.is_collision(point_d)),

            # Danger right
            (dir_u and game.is_collision(point_r)) or 
            (dir_d and game.is_collision(point_l)) or 
            (dir_l and game.is_collision(point_u)) or 
            (dir_r and game.is_collision(point_d)),

            # Danger left
            (dir_d and game.is_collision(point_r)) or 
            (dir_u and game.is_collision(point_l)) or 
            (dir_r and game.is_collision(point_u)) or 
            (dir_l and game.is_collision(point_d)),
            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            # Food location 
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y  # food down
            ]

        return np.array(state, dtype=int)
#store experience in replay memory 
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # popleft if MAX_MEMORY is reached
#train model using experiences from the replay memory 
    def train_long_memory(self):
        """
        Train model using experiences from the replay memory.
        This method implements experience replay, which helps to stabilize learning
        by breaking the correlation between consecutive samples.
        """
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    """Train from a single experience"""
    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)
        
        """
        Choose action based on current state.
        Implements epsilon-greedy strategy for balancing exploration and exploitation.
        As more games are played, epsilon decreases, favoring exploitation over exploration.
        """
        
    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        self.epsilon = 80 - self.n_games
        final_move = [0,0,0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move

''' 
flowchart:
    Get the old state of the game 
    choose an action based on the current state 
    perform the action based on the current state 
    perfrom chosen action and get new state, reward, done status
    train agent's Q network with current experience
    remember experience in the replay memory 

    if done:
        train agent with experiences from memory 
        update the record if a new high score is chieved
        print result 
'''

def train():
    #hyperparameters
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI()
    while True:
        # get old state
        state_old = agent.get_state(game)

        # get move
        final_move = agent.get_action(state_old)

        # perform move and get new state
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # remember
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # train long memory, plot result
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()

            print('Game', agent.n_games, 'Score', score, 'Record:', record)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)

#entry point for the script 
if __name__ == '__main__':
    train()
