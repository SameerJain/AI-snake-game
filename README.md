Snake Game Q-Learning AI
This project implements an AI agent that learns to play the classic Snake game using Q-learning, a type of reinforcement learning algorithm. The AI improves its performance over time by learning from its experiences in the game. NOTE: This uses Anaconda

Requirements
Python 3.7+
Anaconda
PyTorch
Pygame
Matplotlib
NumPy

How to get started
Download the zip file
Extract the file
Run on a conda enviroment: conda create -n pygame_env
Run python agent.py
Watch the AI learn!

Project Structure
The project consists of four main Python files:

snake_game.py: Implements the Snake game using Pygame.
model.py: Defines the neural network model for Q-learning.
agent.py: Implements the Q-learning agent that interacts with the game.
plot.py: Provides functionality to visualize the training progress.

Components
Snake Game (snake_game.py)
This file contains the SnakeGameAI class, which implements the Snake game logic. Key features include:

Game initialization and reset
Food placement
Game state updates
Collision detection
UI rendering

The game is designed to be easily interfaced with the Q-learning agent.
Q-Learning Model (model.py)
This file defines two main classes:

Linear_QNet: A neural network model that approximates the Q-function.
QTrainer: Handles the training process for the Q-network.

The model uses PyTorch for implementation and includes functionality to save the trained model.
Q-Learning Agent (agent.py)
The Agent class in this file manages the Q-learning process. It includes methods for:

State management
Action selection (with epsilon-greedy strategy)
Experience storage and replay
Short-term and long-term memory training

Plotting Utility (plot.py)
This file contains a function to create a real-time plot of the agent's performance during training. It visualizes:

Individual game scores
Moving average of scores

How It Works

The agent interacts with the Snake game environment.
It learns to make decisions (move the snake) based on the current game state.
The Q-learning algorithm helps the agent improve its strategy over time.
Training progress is visualized in real-time using matplotlib.

Requirements

Usage
To train the AI:
