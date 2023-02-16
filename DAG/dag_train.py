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
from lib import save_bar_chart
from lib import save_line_chart
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

    # print strategy chart
    bars = ['P1', 'P2', 'P3', 'P4', 'HW']
    color = ['navy', 'navy', 'navy', 'navy', 'purple']
    save_bar_chart(str_over_time, player_names, bars, color, pic_folder
                        + 'experiment_result_{}.png'.format(total_gas_bound),
                   f'Total gas constraint: {knob}')
    save_line_chart(str_over_time, player_names, bars, True,
                    pic_folder + 'experiment_trend_{}_step_{}_dual_{}.png'.format(total_gas_bound, step_size, dual_step_size), f'Total gas constraint: {knob}')

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
