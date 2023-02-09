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

from lib import create_dag
from lib import gradient_descent_ascent
from lib import print_dag
from lib import Player
#from lib import dual_update
from lib import save_animation
import argparse
plt.style.use('ggplot')

# To modify arguments easier
parser = argparse.ArgumentParser()
parser.add_argument('-p', '--primal_step_size', type=float, default=0.00005, help='Primal step size')
parser.add_argument('-d', '--dual_step_size', type=float, default=0.01, help='Dual update size')
parser.add_argument('-i', '--iterates', type=int, default=15000, help='Iteration')
parser.add_argument('-f', '--pic_folder', type=str, default='.', help='Folder for generated pictures')
args = parser.parse_args()
step_size = args.primal_step_size
dual_step_size = args.dual_step_size
iterates = args.iterates
pic_folder = args.pic_folder
if pic_folder[-1] != '/':
    pic_folder += '/'
if not os.path.exists(pic_folder):
    os.mkdir(pic_folder)

G = create_dag()
#print_dag(G)

log_space = np.logspace(np.log10(8), np.log10(100), 25)
knobs = np.round(log_space).astype(int)
player_names = ['Alice', 'Bob', 'Charlie']
#print(lst)
#sys.exit(0)

for knob in knobs:
    print('knob', knob)

    players = dict()
    plt_kl = dict()
    plt_wass = dict()
    str_over_time = dict()
    for name in player_names:
        players[name] = Player(name, G)
        plt_kl[name] = []
        plt_wass[name] = []
        str_over_time[name] = np.empty((0, len(players[name].strategy))) # shape: (iterate number, number of strategy)

    #print(players['Charlie'].strategy)

    #sys.exit(0)

    # plt_kl_Alice = []
    # plt_kl_Bob = []
    # plt_kl_Charlie = []

    # plt_wass_Alice = []
    # plt_wass_Bob = []
    # plt_wass_Charlie = []

    #iterates = 15000
    lamda = [0]
    total_gas_bound = knob

    for i in range(iterates):
        if not i % 1000:
            print(f"Starting Iteration {i}/{iterates}")
        
        gradient_descent_ascent(G,players, lamda, total_gas_bound, step_size, dual_step_size)

        for name in player_names:
            str_over_time[name] = np.append(str_over_time[name], np.array([players[name].strategy]), axis=0)
            # Tracking differences between distributions
            plt_kl[name].append(players[name].kldiv)
            plt_wass[name].append(players[name].wasser)

    #fig, axs = plt.subplots(1, 3, figsize=(12,5))
    fig, axs = plt.subplots(1, len(player_names), figsize=(3*len(player_names) + 3, 5))

    bars = ['P1', 'P2', 'P3', 'P4', 'HW']
    y_pos = np.arange(len(bars))

    for i, name in enumerate(player_names):
        axs[i].bar(range(len(players[name].strategy)), str_over_time[name][-1], color=['navy', 'navy', 'navy', 'navy', 'purple'])
        axs[i].set_title(name)
        axs[i].set_xlabel('Action')
        axs[i].set_ylim([0, 1])
    axs[0].set_ylabel('Probability') # set y label only for the left most axis

    for ax in axs:
        ax.axhline(y=0.25, color='gray', linestyle='--')

    #plt.ylim([0, 1])

    plt.setp(axs, xticks=y_pos, xticklabels=bars)
    plt.subplots_adjust(left=0.1, right=0.9, bottom = 0.2)
    fig.text(0.5, 0.05, f'Total gas constraint: {knob}', ha='center', fontsize='14')
    plt.savefig(pic_folder + 'experiment_result_{}.png'.format(total_gas_bound))
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
