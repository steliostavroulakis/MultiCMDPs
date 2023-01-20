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
from lib import primal_update
#from lib import dual_update
from lib import save_animation

G = create_dag()
#print_dag(G)

players = dict()
players['Alice'] = Player('Alice',G)
players['Bob'] = Player('Bob',G)
players['Charlie'] = Player('Charlie',G)

#print(players['Charlie'].strategy)

#sys.exit(0)

plt_kl_Alice = []
plt_kl_Bob = []
plt_kl_Charlie = []

plt_wass_Alice = []
plt_wass_Bob = []
plt_wass_Charlie = []

alice_str_over_time = []
bob_str_over_time = []
charlie_str_over_time = []

iterates = 1000
total_gas_constr = 20
lamda = 0

for i in range(iterates):
    print(f"Starting Iteration {i}/{iterates}")
    primal_update(G,players)
    #dual_update(G,players,total_gas_constr)

    alice_str_over_time.append(players['Alice'].strategy)
    bob_str_over_time.append(players['Bob'].strategy)
    charlie_str_over_time.append(players['Charlie'].strategy)

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

#save_animation(alice_str_over_time,'Alice')
#save_animation(bob_str_over_time,'Bob')
#save_animation(charlie_str_over_time,'Charlie')

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