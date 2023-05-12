import networkx as nx
import matplotlib.pyplot as plt
from networkx.drawing.nx_pydot import graphviz_layout
import pprint
import random
from collections import defaultdict
import matplotlib.animation as anim
import numpy as np
import math
import pprint
import sys
import imageio
import os
import shutil
import random
from faker import Faker

fake = Faker()

plt.style.use('ggplot')

from lib import Player
from lib import create_dag
from lib import update_lambda
from lib import gradient_descent

G = create_dag()
players = dict()

for _ in range(20):
    name = fake.first_name()  # Generate a random first name
    number = random.uniform(10, 11)  # Generate a random number between 4 and 50
    players[name] = Player(name, G, number)

t_iterates = 50 # Runs from 1 to iterates
l_iterates = 5    # Runs from 1 to iterates
l_stepsize = 0.1
x_stepsize = 0.2

# Create a directory to store the frames
if not os.path.exists('frames'):
    os.makedirs('frames')

for t in range(t_iterates):

    print(f"Starting Iteration {t+1}/{t_iterates}")

    for lambda_iter in range(l_iterates):

        # Run one gradient ascent step on lambdas
        update_lambda(players, l_stepsize)

    # Run a gradient descent step on Psi
    gradient_descent(G, players, x_stepsize)

    # After each gradient descent step, plot the strategy of each player
    fig, axs = plt.subplots(len(players), 1, figsize=(8, 15))
    for idx, (player_name, player) in enumerate(players.items()):
        axs[idx].bar(range(len(player.strategy)), player.strategy)
        axs[idx].set_title(f'{player_name} Strategy at Iteration {t+1}')
        axs[idx].set_xlabel('Strategy')
        axs[idx].set_ylabel('Probability')
        axs[idx].set_ylim([0, 1])

    plt.tight_layout()
    plt.savefig(f'frames/strategy_{t+1}.png')
    plt.clf()

# After all iterations, compile the frames into a GIF
with imageio.get_writer('strategy_evolution.gif', mode='I') as writer:
    for t in range(t_iterates):
        image = imageio.imread(f'frames/strategy_{t+1}.png')
        writer.append_data(image)

# Optionally, remove the frames directory after compiling the GIF
shutil.rmtree('frames')

# Print Nash Gap
# Print Constraint Violation
# DAG density edges