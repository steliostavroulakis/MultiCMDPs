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
import os
import shutil
import random
from faker import Faker
import imageio.v2 as imageio
import copy

fake = Faker()

plt.style.use('ggplot')

from lib import Player
from lib import create_dag
from lib import update_lambda
from lib import gradient_descent
from lib import calculate_nash_gap

G = create_dag()
players = dict()

for _ in range(2):
    name = fake.first_name()  # Generate a random first name
    number = random.uniform(3.001,3.001)  # Generate a random number between 4 and 50
    players[name] = Player(name, G, number)

t_iterates = 200 # Runs from 1 to iterates
l_iterates = 250    # Runs from 1 to iterates

l_stepsize = 0.01
x_stepsize = 0.01

# Plotting variables
constr_sums = []
l_sums = []
gaps = []

# Create a directory to store the frames
if not os.path.exists('frames'):
    os.makedirs('frames')

for t in range(t_iterates):

    print(calculate_nash_gap(G, players))
    gaps.append(calculate_nash_gap(G, players))

    print(f"Starting Iteration {t+1}/{t_iterates}")

    for lambda_iter in range(l_iterates):

        # Run one gradient ascent step on lambdas
        update_lambda(players, l_stepsize)

    # Run a gradient descent step on Psi
    gradient_descent(G, players, x_stepsize)

    # Include this part after policy update:
    # After each gradient descent step, plot the strategy of each player
    # fig, axs = plt.subplots(len(players), 1, figsize=(8, 15))
    # for idx, (player_name, player) in enumerate(players.items()):
    #     axs[idx].bar(range(len(player.strategy)), player.strategy)
    #     axs[idx].set_title(f'{player_name} Strategy at Iteration {t+1}')
    #     axs[idx].set_xlabel('Strategy')
    #     axs[idx].set_ylabel('Probability')
    #     axs[idx].set_ylim([0, 1])
    # plt.tight_layout()
    # plt.savefig(f'frames/strategy_{t+1}.png')
    # plt.close()

    constr_sum = []
    l_sum = 0

    for player in players.values():
        # Calculate expected gas consumption
        expected_consumption = sum(x * y for x, y in zip(player.strategy, player.path_lengths))
        #print(expected_consumption)
        #sys.exit(0)
        # Calculate expected constraint violation
        constraint_function = expected_consumption - player.gas
        if constraint_function < 0:
            constraint_function = 0
        else:
            pass
        constr_sum.append(constraint_function)
        l_sum += player.l

    constr_sums.append(max(constr_sum))
    l_sums.append(l_sum)

# Plot for constr_violation
plt.figure(figsize=(8, 6))
plt.plot(range(1, t_iterates + 1), gaps, color='tab:red')
plt.xlabel('Iteration')
plt.ylabel('Sum of nash_gap')
plt.title('Sum of nash_gap over time')
plt.savefig('images/nash_gap.png')
plt.close()

# Plot for constr_violation
plt.figure(figsize=(8, 6))
plt.plot(range(1, t_iterates + 1), constr_sums, color='tab:red')
plt.xlabel('Iteration')
plt.ylabel('Sum of constr_violation')
plt.title('Sum of constr_violation over time')
plt.savefig('images/constr_violation.png')
plt.close()

# Plot for player.l
plt.figure(figsize=(8, 6))
plt.plot(range(1, t_iterates + 1), l_sums, color='tab:blue')
plt.xlabel('Iteration')
plt.ylabel('Sum of lambdas')
plt.title('Sum of lambdas over time')
plt.savefig('images/lambdas.png')
plt.close()

# # After all iterations, compile the frames into a GIF
# with imageio.get_writer('strategy_evolution.gif', mode='I') as writer:
#     for t in range(t_iterates):
#         image = imageio.imread(f'frames/strategy_{t+1}.png')
#         writer.append_data(image)

# Optionally, remove the frames directory after compiling the GIF
shutil.rmtree('frames')

# Print Nash Gap
# Print Constraint Violation
# DAG density edges
#shutil.rmtree('frames')

# # Include this part after policy update:
# # After each gradient descent step, plot the strategy of each player
fig, axs = plt.subplots(len(players), 1, figsize=(8, 15))
for idx, (player_name, player) in enumerate(players.items()):
    axs[idx].bar(range(len(player.strategy)), player.strategy)
    axs[idx].set_title(f'{player_name}')
    axs[idx].set_xlabel('Strategy')
    axs[idx].set_ylabel('Probability')
    axs[idx].set_ylim([0, 1])
plt.tight_layout()
plt.savefig('images/strategy_final.png')
plt.clf()

# Print Nash Gap - OK
# Print Constraint Violation - OK
# DAG density edges
