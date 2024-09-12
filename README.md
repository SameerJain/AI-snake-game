# Snake Game Q-Learning AI
This project implements an AI agent that learns to play the classic Snake game using Q-learning, a type of reinforcement learning algorithm. The AI improves its performance over time by learning from its experiences in the game. NOTE: This uses Anaconda

## Requirements

Python 3.7+

Anaconda

PyTorch

Pygame

Matplotlib

NumPy


## How to get started
1) Download the zip and extract the file

2) Run Anaconda:
`conda create -n pygame_env`

3) Run the AI Agent:
`python agent.py`

4) Watch the AI learn!

## Project Structure

**snake_game.py:** Implements the Snake game using Pygame.

**model.py:** Defines the neural network model for Q-learning.

**agent.py:** Implements the Q-learning agent that interacts with the game.

**plot.py:** Provides functionality to visualize the training progress.


### Contains:

Game initialization and reset

Food placement

Game state updates

Collision detection

UI rendering


## How it Works

The Q-learning process works as follows:

**State Representation:**
The game state is represented as a vector of 11 binary values in get_state() method.
This includes information about danger directions, current direction, and food location.


**Q-Network:**
The Q-function is approximated using a neural network (Linear_QNet class).
It takes the state as input and outputs Q-values for each possible action.


**Action Selection:**
The agent uses an epsilon-greedy strategy in get_action() method.
With probability epsilon, it chooses a random action (exploration).
Otherwise, it chooses the action with the highest Q-value (exploitation).
Epsilon decreases over time to favor exploitation as the agent learns.


**Learning Process:**
After each action, the agent receives a reward and observes the new state.
The experience (state, action, reward, next_state) is stored in memory.
The Q-network is updated using experience replay in train_long_memory():

A batch of experiences is randomly sampled from memory.
The Q-values are updated using the Bellman equation:
Q(s,a) = r + γ * max(Q(s',a'))
The network is trained to minimize the difference between predicted and target Q-values.


**Game Integration:**
The SnakeGameAI class provides the environment for the agent.
In each step, the game updates based on the agent's action and returns the new state and reward.


**Visualization:**
The plot() function provides real-time visualization of the agent's performance.
It shows how the score improves over time as the agent learns.



