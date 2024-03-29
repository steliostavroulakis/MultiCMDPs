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

from lib import create_dag, gradient_descent
from lib import update_lambda
from lib import print_dag
from lib import Player
#from lib import dual_update
from lib import save_animation
plt.style.use('ggplot')

G = create_dag()
#print_dag(G)

log_space = np.logspace(np.log10(8), np.log10(100), 25)
knobs = np.round(log_space).astype(int)
#knobs = [8]
#print(lst)
#sys.exit(0)

for knob in knobs:

    players = dict()
    players['Alice'] = Player('Alice',G)
    players['Bob'] = Player('Bob',G)
    players['Charlie'] = Player('Charlie',G)

    #print(players['Charlie'].strategy)

    #sys.exit(0)

    # plt_kl_Alice = []
    # plt_kl_Bob = []
    # plt_kl_Charlie = []

    # plt_wass_Alice = []
    # plt_wass_Bob = []
    # plt_wass_Charlie = []

    alice_str_over_time = []
    bob_str_over_time = []
    charlie_str_over_time = []

    iterates = 150
    lamda = [0]
    total_gas_bound = knob

    for i in range(iterates):
        print(f"Starting Iteration {i}/{iterates}")
        
        update_lambda(players)
        gradient_descent(players)
        gradient_descent_ascent(G,players, lamda, total_gas_bound)

        alice_str_over_time.append(players['Alice'].strategy)
        bob_str_over_time.append(players['Bob'].strategy)
        charlie_str_over_time.append(players['Charlie'].strategy)

        # Tracking differences between distributions
        # plt_kl_Alice.append(players['Alice'].kldiv)
        # plt_kl_Bob.append(players['Bob'].kldiv)
        # plt_kl_Charlie.append(players['Charlie'].kldiv)

        # plt_wass_Alice.append(players['Alice'].wasser)
        # plt_wass_Bob.append(players['Bob'].wasser)
        # plt_wass_Charlie.append(players['Charlie'].wasser)

    fig, axs = plt.subplots(1, 3, figsize=(12,5))

    bars = ['P1', 'P2', 'P3', 'P4', 'HW']
    y_pos = np.arange(len(bars))

    axs[0].bar(range(len(alice_str_over_time[-1])), alice_str_over_time[-1], color=['#000000', '#000000', '#000000', '#000000', '#b2050f'])
    axs[0].set_title('Alice')
    axs[0].set_xlabel('Action')
    axs[0].set_ylabel('Probability')
    axs[0].set_ylim([0, 1])


    axs[1].bar(range(len(bob_str_over_time[-1])), bob_str_over_time[-1], color=['#000000', '#000000', '#000000', '#000000', '#b2050f'])
    axs[1].set_title('Bob')
    axs[1].set_xlabel('Action')
    axs[1].set_ylim([0, 1])


    axs[2].bar(range(len(charlie_str_over_time[-1])), charlie_str_over_time[-1], color=['#000000', '#000000', '#000000', '#000000', '#b2050f'])
    axs[2].set_title('Charlie')
    axs[2].set_xlabel('Action')
    axs[2].set_ylim([0, 1])

    for ax in axs:
        ax.axhline(y=0.25, color='gray', linestyle='--')

    #plt.ylim([0, 1])

    plt.setp(axs, xticks=y_pos, xticklabels=bars)
    plt.subplots_adjust(left=0.1, right=0.9, bottom = 0.2)
    fig.text(0.5, 0.05, f'Total gas constraint: {knob}', ha='center', fontsize='14')
    plt.savefig('experiment_result_{}.png'.format(total_gas_bound))
    plt.close()

    #save_animation(alice_str_over_time,'Alice')
    #save_animation(bob_str_over_time,'Bob')
    #save_animation(charlie_str_over_time,'Charlie')

    # plt.plot(plt_kl_Alice)
    # plt.plot(plt_kl_Bob)
    # plt.plot(plt_kl_Charlie)
    # plt.savefig('probs.jpg')
    # plt.close()

    # plt.plot(plt_wass_Alice)
    # plt.plot(plt_wass_Bob)
    # plt.plot(plt_wass_Charlie)
    # plt.savefig('probs_wasser.jpg')
    # plt.close()

    # for player in players.values():
    #     player.render_path()

    #for name, player in players.items():
    #  print("Player: ,",name,": Congestion: ",player.congestion)
    #  print(player.path)
    #  print(player.strategy)