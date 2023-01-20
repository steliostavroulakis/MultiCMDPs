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

from lib import create_dag
from lib import print_dag
from lib import Player
from lib import play_game
from lib import save_animation

G = create_dag()
#print_dag(G)

highway = set()
highway.add(('s','l1'))
highway.add(('l1','l2'))
highway.add(('l2','l3'))
highway.add(('l3','l4'))
highway.add(('l4','l5'))
highway.add(('l5','l6'))
highway.add(('l6','t'))

congestion_func = lambda x: x
congestion_func_derivative = lambda x: 1

highway_congestion_func = lambda x: x / 2
highway_congestion_func_derivative = lambda x: 0.5

congestion_funcs = {(a, b) : highway_congestion_func if (a, b) in highway else congestion_func 
                    for a in G.nodes for b in G.nodes}
congestion_func_derivatives = {(a, b) : highway_congestion_func_derivative if (a, b) in highway else congestion_func_derivative
                               for a in G.nodes for b in G.nodes}

gas_func = lambda x: 1
gas_func_derivative = lambda x: 0

highway_gas_func = lambda x: 2
highway_gas_func_derivative = lambda x: 0

gas_funcs = {(a, b) : highway_gas_func if (a, b) in highway else gas_func
                    for a in G.nodes for b in G.nodes}
gas_func_derivatives = {(a, b) : highway_gas_func_derivative if (a, b) in highway else gas_func_derivative
                    for a in G.nodes for b in G.nodes}

rewards = [congestion_funcs, gas_funcs]
reward_derivatives = [congestion_func_derivatives, gas_func_derivatives]

"""
For this example, we say everybody responds the same to congestion and gas. This doesn't have to be the case!
The model is perfectly capable of letting someone be on a motorcycle and be able to weave through congested traffic
or be in a hybrid and have better fuel economy off the highway compared to others
"""

players = dict()
players['Alice']   = Player('Alice'  , G, rewards, [16])
players['Bob']     = Player('Bob'    , G, rewards, [5] )
players['Charlie'] = Player('Charlie', G, rewards, [8] )

"""
Alice has a higher gas restriction: this cannot model how much gas she has in her tank because it is a constraint on expectation.
Perhaps this is a commute that she makes daily, and she is willing to spend a certain amount of money on gas on average.

In this case then, we could say that Bob and Charlie would not be able to afford to take the highway all of the time, but Alice could.
"""

plt_kl_Alice = []
plt_kl_Bob = []
plt_kl_Charlie = []

plt_wass_Alice = []
plt_wass_Bob = []
plt_wass_Charlie = []

alice_str_over_time = []
bob_str_over_time = []
charlie_str_over_time = []


for i in range(100):
    print(f'{i}/50')
    
    # choose actions
    players['Alice'].choose_action()
    players['Bob'].choose_action()
    players['Charlie'].choose_action()

    # calculate congestion of each edge
    congestion_dict = {edge : 0 for edge in G.edges}
    for player in players.values():
        for edge in player.edges:
            congestion_dict[edge] += 1

    # update strategies
    players['Alice'].update_primal(congestion_dict ,step_size=0.01)
    # players['Alice'].update_dual(step_size=0.01)

    
    """
    # play_game(G, players)
    alice_str_over_time.append(players['Alice'].strategy)
    bob_str_over_time.append(players['Bob'].strategy)
    charlie_str_over_time.append(players['Charlie'].strategy)
    """

    # Tracking differences between distributions
    plt_kl_Alice.append(players['Alice'].kldiv)
    plt_kl_Bob.append(players['Bob'].kldiv)
    plt_kl_Charlie.append(players['Charlie'].kldiv)

    plt_wass_Alice.append(players['Alice'].wasser)
    plt_wass_Bob.append(players['Bob'].wasser)
    plt_wass_Charlie.append(players['Charlie'].wasser)

print(alice_str_over_time[-1])
print(bob_str_over_time[-1])
print(charlie_str_over_time[-1])

save_animation(alice_str_over_time,'Alice')
save_animation(bob_str_over_time,'Bob')
save_animation(charlie_str_over_time,'Charlie')

plt.plot(plt_kl_Alice)
plt.plot(plt_kl_Bob)
plt.plot(plt_kl_Charlie)
plt.savefig('probs.jpg')
plt.close()

plt.plot(plt_wass_Alice)
plt.plot(plt_wass_Bob)
plt.plot(plt_wass_Charlie)
plt.savefig('probs_wasser.jpg')
plt.close()

for player in players.values():
    player.render_path()

#for name, player in players.items():
#  print("Player: ,",name,": Congestion: ",player.congestion)
#  print(player.path)
#  print(player.strategy)
