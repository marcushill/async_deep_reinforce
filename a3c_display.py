# -*- coding: utf-8 -*-
import cv2
import tensorflow as tf
import threading
import numpy as np

import signal
import random
import math
import os
import time

from vizdoom import GameVariable
from doom_game_state import DoomGameState
from game_ac_network import GameACFFNetwork, GameACLSTMNetwork
from a3c_training_thread import A3CTrainingThread
# from game_state import GameState
from rmsprop_applier import RMSPropApplier

from constants import ACTION_SIZE
from constants import PARALLEL_SIZE
from constants import INITIAL_ALPHA_LOW
from constants import INITIAL_ALPHA_HIGH
from constants import INITIAL_ALPHA_LOG_RATE
from constants import MAX_TIME_STEP
from constants import CHECKPOINT_DIR
from constants import LOG_FILE
from constants import RMSP_EPSILON
from constants import RMSP_ALPHA
from constants import GRAD_NORM_CLIP
from constants import USE_GPU
from constants import USE_LSTM


def log_uniform(lo, hi, rate):
    log_lo = math.log(lo)
    log_hi = math.log(hi)
    v = log_lo * (1 - rate) + log_hi * rate
    return math.exp(v)


device = "/cpu:0"
if USE_GPU:
    device = "/gpu:0"

initial_learning_rate = log_uniform(INITIAL_ALPHA_LOW,
                                    INITIAL_ALPHA_HIGH,
                                    INITIAL_ALPHA_LOG_RATE)

global_t = 0

stop_requested = False

global_game = DoomGameState(scenario_path="scenarios/cig.cfg", window_visible=True)
if USE_LSTM:
    global_network = GameACLSTMNetwork(global_game.get_action_size(), -1, device)
else:
    global_network = GameACFFNetwork(global_game.get_action_size(), -1, device)

training_threads = []

learning_rate_input = tf.placeholder("float")

grad_applier = RMSPropApplier(learning_rate=learning_rate_input,
                              decay=RMSP_ALPHA,
                              momentum=0.0,
                              epsilon=RMSP_EPSILON,
                              clip_norm=GRAD_NORM_CLIP,
                              device=device)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

saver = tf.train.Saver()
checkpoint = tf.train.get_checkpoint_state(CHECKPOINT_DIR)
if checkpoint and checkpoint.model_checkpoint_path:
    saver.restore(sess, checkpoint.model_checkpoint_path)
    print("checkpoint loaded:", checkpoint.model_checkpoint_path)
else:
    print("Could not find old checkpoint")


def choose_action(pi_values):
    return np.random.choice(range(len(pi_values)), p=pi_values)


global_game.start()
print(global_game.game.get_available_game_variables())
while not global_game.terminal:
    # labels = global_game.labels_buffer
    # if labels is not None:
    #     cv2.imshow('ViZDoom Labels Buffer', labels)
    #
    cv2.waitKey(300)

    # for l in global_game.labels:
    #      print("Object id:", l.object_id, "object name:", l.object_name, "label:", l.value)
    # #     print("Object position X:", l.object_position_x, "Y:", l.object_position_y, "Z:", l.object_position_z)
    # print("Sees Enemy: ", any(x.object_name == "DoomPlayer" and x.value != 255 for x in global_game.labels))
    #
    print("Pre-reward: ", global_game.game.get_last_reward())
    print("Kill Count: ", global_game.kill_count)
    print("Death Count: ", global_game.death_count)
    print("Suicide Count: ", global_game.suicide_count)
    print("Frag Count: ", global_game.game.get_game_variable(GameVariable.FRAGCOUNT))
    print("Reward: ", global_game.reward)
    print("Total Reward: ", global_game.game.get_total_reward())
    print("=====================")


    pi_values = global_network.run_policy(sess, global_game.s_t)

    action = choose_action(pi_values)
    # print("Policy: ", pi_values)
    # print("Action: ", action)
    global_game.process(action)

    if not global_game.terminal:
        global_game.update()
