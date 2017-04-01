# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import random
import time
import sys

# from game_state import GameState
# from game_state import ACTION_SIZE
from game_ac_network import GameACFFNetwork, GameACLSTMNetwork

from constants import GAMMA
from constants import LOCAL_T_MAX
from constants import ENTROPY_BETA
# from constants import USE_LSTM

LOG_INTERVAL = 100
PERFORMANCE_LOG_INTERVAL = 1000


class A3CTrainingThread(object):
    def __init__(self,
                 thread_index,
                 local_network,
                 global_network,
                 initial_learning_rate,
                 learning_rate_input,
                 max_global_time_step,
                 grad_applier,
                 game,
                 device):

        self.thread_index = thread_index
        self.learning_rate_input = learning_rate_input
        self.max_global_time_step = max_global_time_step

        # if USE_LSTM:
        #   self.local_network = GameACLSTMNetwork(ACTION_SIZE, thread_index, device)
        # else:
        #   self.local_network = GameACFFNetwork(ACTION_SIZE, thread_index, device)

        self.local_network = local_network
        self.local_network.prepare_loss(ENTROPY_BETA)

        with tf.device(device):
            var_refs = [v._ref() for v in self.local_network.get_vars()]
            self.gradients = tf.gradients(
                self.local_network.total_loss, var_refs,
                gate_gradients=False,
                aggregation_method=None,
                colocate_gradients_with_ops=False)

        self.apply_gradients = grad_applier.apply_gradients(
            global_network.get_vars(),
            self.gradients)

        self.sync = self.local_network.sync_from(global_network)

        self.game = game

        self.local_t = 0

        self.initial_learning_rate = initial_learning_rate

        self.episode_reward = 0

        # variable controlling log output
        self.prev_local_t = 0

    def _anneal_learning_rate(self, global_time_step):
        learning_rate = self.initial_learning_rate * (
        self.max_global_time_step - global_time_step) / self.max_global_time_step
        if learning_rate < 0.0:
            learning_rate = 0.0
        return learning_rate

    def choose_action(self, pi_values):
        return np.random.choice(range(len(pi_values)), p=pi_values)

    def _record_score(self, sess, summary_writer, summary_op, score_input, score, global_t):
        summary_str = sess.run(summary_op, feed_dict={
            score_input: score
        })
        summary_writer.add_summary(summary_str, global_t)
        summary_writer.flush()

    def set_start_time(self, start_time):
        self.start_time = start_time

    def process(self, sess, global_t, summary_writer, summary_op, score_input):
        states = []
        actions = []
        rewards = []
        values = []

        terminal_end = False

        # copy weights from shared to local
        sess.run(self.sync)

        start_local_t = self.local_t

        self.local_network.start_train()

        try:
            self.game.start()
        except AttributeError:
            pass

        # t_max times loop
        # for i in range(LOCAL_T_MAX):
        episode_step_count = 1
        skip_counter = 4
        while not self.game.terminal:
            if skip_counter != 4:
                skip_counter += 1
                self.game.process(action)
            else:
                skip_counter = 0

            pi_, value_ = self.local_network.run_policy_and_value(sess, self.game.s_t)
            action = self.choose_action(pi_)

            states.append(self.game.s_t)
            actions.append(action)
            values.append(value_)

            if (self.thread_index == 0) and (self.local_t % LOG_INTERVAL == 0):
                print("    pi= {}".format(pi_))
                print("     V= {}".format(value_))
                print(" Score=", self.game.score)
                print("Reward=", self.game.reward)

            # process game
            self.game.process(action)

            # receive game result
            reward = self.game.reward
            terminal = self.game.terminal

            # self.episode_reward += reward

            # clip reward
            rewards.append(np.clip(reward, -1, 1))

            self.local_t += 1

            if len(actions) == 30 and not self.game.terminal and episode_step_count < self.game.max_episode_length -1:
                value = self.local_network.run_value(sess, self.game.s_t)
                self.train(sess, global_t, actions, states, rewards, values, value)
                states = []
                actions = []
                rewards = []
                values = []

            # s_t1 -> s_t
            self.game.update()
            episode_step_count += 1

            if terminal:
                terminal_end = True
                print("score={}".format(self.game.score))

                self._record_score(sess, summary_writer, summary_op, score_input,
                                   self.game.score, global_t)

                self.game.reset()
                try:
                    self.local_network.reset_state()
                except AttributeError:
                    pass
                break

        R = 0.0
        if not terminal_end:
            R = self.local_network.run_value(sess, self.game.s_t)

        self.train(sess, global_t, actions, states, rewards, values, R)

        if (self.thread_index == 0) and (self.local_t - self.prev_local_t >= PERFORMANCE_LOG_INTERVAL):
            self.prev_local_t += PERFORMANCE_LOG_INTERVAL
            elapsed_time = time.time() - self.start_time
            steps_per_sec = global_t / elapsed_time
            print("### Performance : {} STEPS in {:.0f} sec. {:.0f} STEPS/sec. {:.2f}M STEPS/hour".format(
                global_t, elapsed_time, steps_per_sec, steps_per_sec * 3600 / 1000000.))

        # return advanced local step size
        diff_local_t = self.local_t - start_local_t
        return diff_local_t

    def train(self, sess, global_t, actions, states, rewards, values, R=0.0):
        # The other code does the below every 30 frames
        actions.reverse()
        states.reverse()
        rewards.reverse()
        values.reverse()

        batch_si = []
        batch_a = []
        batch_td = []
        batch_R = []

        # compute and accmulate gradients
        for (ai, ri, si, Vi) in zip(actions, rewards, states, values):
            R = ri + GAMMA * R
            td = R - Vi
            a = np.zeros([self.game.get_action_size()])
            a[ai] = 1

            batch_si.append(si)
            batch_a.append(a)
            batch_td.append(td)
            batch_R.append(R)

        cur_learning_rate = self._anneal_learning_rate(global_t)

        self.local_network.apply_gradients(sess, self.apply_gradients,
                                           batch_si,
                                           batch_a,
                                           batch_td,
                                           batch_R,
                                           cur_learning_rate)

    def run_test_game(self, sess):
        self.game.reset()
        self.game.start("saves/doom_thread{}_{}.lmp".format(str(self.thread_index), str(time.time())))
        while not self.game.terminal:
            pi_values = self.local_network.run_policy(sess, self.game.s_t)

            action = np.random.choice(range(len(pi_values)), p=pi_values)
            self.game.process(action)
        self.game.game.close()
        self.game.game.init()
        self.game.reset()

