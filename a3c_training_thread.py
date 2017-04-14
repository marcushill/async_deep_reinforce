# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import random
import time
import sys

# from game_state import GameState
# from game_state import ACTION_SIZE
from game_ac_network import GameACLSTMNetwork, GameNavigationNetwork

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
                 device,
                 navigation_network,
                 global_navigation_network):

        self.thread_index = thread_index
        self.learning_rate_input = learning_rate_input
        self.max_global_time_step = max_global_time_step

        self.local_network = local_network
        self.local_network.prepare_loss(ENTROPY_BETA)
        self.navigation_network = navigation_network
        self.navigation_network.prepare_loss(ENTROPY_BETA)

        with tf.device(device):
            var_refs = [v._ref() for v in self.local_network.get_vars()]
            feature_refs = [v._ref() for v in self.local_network.get_feature_vars()]
            self.gradients = tf.gradients(
                self.local_network.total_loss, var_refs,
                gate_gradients=False,
                aggregation_method=None,
                colocate_gradients_with_ops=False)

            self.feature_gradients = tf.gradients(
                self.local_network.feature_loss, feature_refs,
                gate_gradients=False,
                aggregation_method=None,
                colocate_gradients_with_ops=False)

            if navigation_network is not None and global_navigation_network is not None:
                nav_var_refs = [v._ref() for v in self.navigation_network.get_vars()]
                self.nav_gradients = tf.gradients(
                    self.navigation_network.total_loss, nav_var_refs,
                    gate_gradients=False,
                    aggregation_method=None,
                    colocate_gradients_with_ops=False)

        self.apply_gradients = grad_applier.apply_gradients(
            global_network.get_vars(),
            self.gradients)

        self.apply_feature_gradients = grad_applier.apply_gradients(
            global_network.get_feature_vars(),
            self.feature_gradients
        )

        if self.nav_gradients is not None:
            self.apply_nav_gradients = grad_applier.apply_gradients(
                global_navigation_network.get_vars(),
                self.nav_gradients)
            self.sync_nav = self.navigation_network.sync_from(global_navigation_network)

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

    def _record_score(self, sess, summary_writer, feature_accuracy, step_count):
        summary = tf.Summary()
        k_d = self.game.kill_count / (float(self.game.death_count) if self.game.death_count > 0 else 1.0)
        summary.value.add(tag="Score", simple_value=float(self.game.score))
        summary.value.add(tag="Deaths", simple_value=float(self.game.death_count))
        summary.value.add(tag="Suicides", simple_value=float(self.game.suicide_count))
        summary.value.add(tag="Kills", simple_value=float(self.game.kill_count))
        summary.value.add(tag="K/D", simple_value=float(k_d))
        summary.value.add(tag="Enemy Detection Rate", simple_value=float(feature_accuracy))
        summary_writer.add_summary(summary, step_count)
        summary_writer.flush()

    def set_start_time(self, start_time):
        self.start_time = start_time

    def process(self, sess, global_t, episode_count, summary_writer):
        states = []
        actions = []
        rewards = []
        values = []
        predicted_features = []
        actual_features = []

        nav_states = []
        nav_actions = []
        nav_rewards = []
        nav_values = []

        terminal_end = False

        # copy weights from shared to local
        sess.run(self.sync)

        start_local_t = self.local_t

        self.local_network.start_train()
        if self.sync_nav is not None:
            sess.run(self.sync_nav)
            self.navigation_network.start_train()

        try:
            self.game.start()
        except AttributeError:
            pass

        episode_step_count = 1
        skip_counter = 4
        while not self.game.terminal:
            if skip_counter != 4:
                skip_counter += 1
                self.game.process(action, is_skip=True)
            else:
                skip_counter = 0

            pi_, value_ = self.local_network.run_policy_and_value(sess, self.game.s_t)
            feat_pred = self.local_network.run_feature_detection(sess, self.game.s_t)
            actual_feature = 0
            if self.game.sees_enemy:
                actual_feature = 1

            sees_enemy = np.argmax(feat_pred) == 1
            used_nav = False

            if self.navigation_network is None or (sees_enemy and self.game.avaliable_ammo > 0):
                action = self.choose_action(pi_)
            else:
                n_pi_, n_value_ = self.navigation_network.run_policy_and_value(sess, self.game.s_t)
                action = self.choose_action(n_pi_) + 4  # MOVE_Forward is at index 4, turn right and left immediate follow
                nav_states.append(self.game.s_t)
                nav_actions.append(action)
                nav_values.append(n_value_)
                used_nav = True

            states.append(self.game.s_t)
            actions.append(action)
            values.append(value_)
            predicted_features.append(feat_pred)
            actual_features.append(actual_feature)

            if (self.thread_index == 0) and (self.local_t % LOG_INTERVAL == 0):
                print("    pi= {}".format(pi_))
                print("Action=", action)
                print("     V= {}".format(value_))
                print(" Score=", self.game.score)
                print("Reward=", self.game.reward)
                print("Sees enemy=", sees_enemy)

            # process game
            self.game.process(action)

            # receive game result
            reward = self.game.reward
            terminal = self.game.terminal

            # self.episode_reward += reward

            # clip reward
            # rewards.append(np.clip(reward, -1, 1))
            rewards.append(reward)
            if used_nav:
                nav_rewards.append(reward)

            self.local_t += 1

            if len(actions) == LOCAL_T_MAX and not self.game.terminal and episode_step_count < self.game.max_episode_length - 1:
                value = self.local_network.run_value(sess, self.game.s_t)
                self.train(sess, global_t, actions, states, rewards, values, actual_features, predicted_features, value)
                states = []
                actions = []
                rewards = []
                values = []
                predicted_features = []
                actual_features = []

                if self.navigation_network is not None and len(nav_states) > 0:
                    nav_val = self.navigation_network.run_value(sess, self.game.s_t)
                    self.train_nav(sess, global_t, nav_actions, nav_states, nav_rewards, nav_values, nav_val)
                    nav_states = []
                    nav_actions = []
                    nav_rewards = []
                    nav_values = []

            # s_t1 -> s_t
            self.game.update()
            episode_step_count += 1

            if terminal:
                terminal_end = True
                print("score={}".format(self.game.score))

                feature_accuracy = sum(
                    1 if np.argmax(v) == actual_features[i] else 0 for i, v in enumerate(predicted_features)) / len(
                    predicted_features)

                print("feature_accuracy={}".format(feature_accuracy))

                self._record_score(sess, summary_writer, feature_accuracy, episode_count)

                self.game.reset()
                try:
                    self.local_network.reset_state()
                except AttributeError:
                    pass
                break

        R = 0.0
        if not terminal_end:
            R = self.local_network.run_value(sess, self.game.s_t)

        self.train(sess, global_t, actions, states, rewards, values, actual_features, predicted_features, R)
        if len(nav_states) > 0:
            self.train_nav(sess, global_t, nav_actions, nav_states, nav_rewards, nav_values, R)

        if (self.thread_index == 0) and (self.local_t - self.prev_local_t >= PERFORMANCE_LOG_INTERVAL):
            self.prev_local_t += PERFORMANCE_LOG_INTERVAL
            elapsed_time = time.time() - self.start_time
            steps_per_sec = global_t / elapsed_time
            print("### Performance : {} STEPS in {:.0f} sec. {:.0f} STEPS/sec. {:.2f}M STEPS/hour".format(
                global_t, elapsed_time, steps_per_sec, steps_per_sec * 3600 / 1000000.))

        # return advanced local step size
        diff_local_t = self.local_t - start_local_t
        return diff_local_t

    def train(self, sess, global_t, actions, states, rewards, values, actual_features, predicted_features, R=0.0):
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

        self.local_network.apply_feature_gradient(sess, self.apply_feature_gradients,
                                                  batch_si,
                                                  predicted_features,
                                                  actual_features,
                                                  cur_learning_rate)

        self.local_network.apply_gradients(sess, self.apply_gradients,
                                           batch_si,
                                           batch_a,
                                           batch_td,
                                           batch_R,
                                           cur_learning_rate)

    def train_nav(self, sess, global_t, actions, states, rewards, values, R=0.0):
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
            a = np.zeros([3])
            a[ai - 4] = 1

            batch_si.append(si)
            batch_a.append(a)
            batch_td.append(td)
            batch_R.append(R)

        cur_learning_rate = self._anneal_learning_rate(global_t)

        self.navigation_network.apply_gradients(sess, self.apply_nav_gradients,
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
            feat_pred = self.local_network.run_feature_detection(sess, self.game.s_t)
            sees_enemy = np.argmax(feat_pred) == 1

            if sees_enemy:
                action = np.random.choice(range(len(pi_values)), p=pi_values)
            else:
                pi_values = self.navigation_network.run_policy(sess, self.game.s_t)
                action = np.random.choice(range(len(pi_values)), p=pi_values) + 4
            self.game.process(action)

        self.game.game.close()
        self.game.game.init()
        self.game.reset()
