from itertools import combinations_with_replacement, product

from vizdoom.vizdoom import DoomGame, ScreenResolution, ScreenFormat, Button, GameVariable, Mode
import scipy
import scipy.misc
from scipy.spatial import distance
import numpy as np
from constants import FRAME_SKIP


class DoomGameState:
    def __init__(self, scenario_path="scenarios/defend_center.cfg", window_visible=False):
        self.reward = 0
        game = DoomGame()
        game.load_config(scenario_path)  # This corresponds to the simple task we will pose our agent\
        # game.load_config("../../scenarios/cig.cfg")

        # game.set_doom_map("map01")  # Limited deathmatch.
        # game.set_doom_map("map02")  # Full deathmatch.
        game.set_window_visible(window_visible)

        # # Start multiplayer game only with your AI (with options that will be used in the competition, details in cig_host example).
        game.add_game_args("-host 1 -deathmatch +timelimit 1.0 "
                           "+sv_forcerespawn 1 +sv_noautoaim 1 +sv_respawnprotect 1 +sv_spawnfarthest 1")

        # Name your agent and select color
        # colors: 0 - green, 1 - gray, 2 - brown, 3 - red, 4 - light gray, 5 - light brown, 6 - light red, 7 - light blue
        game.add_game_args("+name AI +colorset 0")
        game.add_available_game_variable(GameVariable.POSITION_X)
        game.add_available_game_variable(GameVariable.POSITION_Y)
        game.set_labels_buffer_enabled(True)

        self.game = game

        # generates the the actual action arrays [True, False, False], etc for each action...
        # generates all possible button combinations which I learned from the original vizdoom paper
        self.real_actions = [[i == j for i in range(game.get_available_buttons_size())]
                             for j in range(game.get_available_buttons_size())]
        # self.real_actions = [list(x) for x in
        #                      product([True, False], repeat=game.get_available_buttons_size())]

        self.last_variables = None
        self.position_buffer = None
        self.skip_next_round = False

        self.reset()
        game.init()

    def _process_frame(self, action, reshape):
        if action is not None:
            reward = self.game.make_action(action, FRAME_SKIP)
        else:
            reward = None

        frame = self.game.get_state().screen_buffer
        if frame is None:
            return reward, None
        # print(frame.shape)
        # frame = frame[10:-10, 30:-30]
        frame = scipy.misc.imresize(frame, [108, 60])
        # print("After Resize: ", frame.shape)
        if reshape:
            frame = np.reshape(frame, (1, 108, 60, 3))

        frame = frame.astype(np.float32)
        frame *= (1.0 / 255.0)
        # print(frame.shape)

        return reward, frame

    @property
    def terminal(self):
        return self.game.is_episode_finished()

    def reset(self):
        self.reward = 0
        self.skip_next_round = False
        self.kill_count = 0  # INCREMENT EACH TIME FRAGS INCREASES
        self.death_count = 0  # INCREMENT EACH TIME WE DIE INCREASES
        self.suicide_count = 0  # INCREMENT EACH TIME FRAG COUNT DECREASES
        self.in_dying_cycle = False

    @property
    def score(self):
        return self.game.get_game_variable(GameVariable.FRAGCOUNT)

    def start(self, name=""):
        self.last_variables = None
        self.position_buffer = None

        self.game.new_episode(name)

        # Add specific number of bots
        # (file examples/bots.cfg must be placed in the same directory as the Doom executable file,
        # edit this file to adjust bots).
        self.game.send_game_command("removebots")
        for i in range(7):
            self.game.send_game_command("addbot")

        _, screen = self._process_frame(None, False)
        self.s_t = np.stack((screen, screen, screen, screen), axis=0)

    def process(self, action, is_skip=False):
        real_action = self.real_actions[action]
        reward, frame = self._process_frame(real_action, True)
        # print("Before reward:", reward)
        self.reward = self.__calculate_reward(reward)
        # print("After reward: ", self.reward)

        # self.s_t is the state over time an 84x84x4 3-dimensional matrix
        # self.s_t1 is taking the original s_t array which you can think of as a
        # 1-dimensional array of 4 frames, it's slicing the array so that it is now of length 3
        # and then it appends a new frame to the end of this 1-dimensional array
        if frame is not None:
            #        print('s_t shape', self.s_t.shape)
            #        print('frame shape', frame.shape)
            self.s_t1 = np.append(self.s_t[1:, :, :, :], frame, axis=0)

    def __calculate_reward(self, r):
        was_dead = self.in_dying_cycle
        current_position = (self.game.get_game_variable(GameVariable.POSITION_X),
                            self.game.get_game_variable(GameVariable.POSITION_Y))

        if self.last_variables is None:
            # We want to reset everything if the player died
            self.last_variables = {}
            self.initial_position = current_position
            self.position_buffer = []

        if was_dead:
            self.last_variables[GameVariable.HEALTH] = self.game.get_game_variable(GameVariable.HEALTH)
            self.last_variables[GameVariable.SELECTED_WEAPON_AMMO] = self.game.get_game_variable(
                GameVariable.SELECTED_WEAPON_AMMO)
            self.position_buffer = [current_position]

        if self.game.is_player_dead() and not was_dead:
            self.death_count += 1

        self.in_dying_cycle = self.game.is_player_dead()

        old_variables = self.last_variables.copy()
        self.position_buffer.append(current_position)

        self.last_variables[GameVariable.HEALTH] = self.game.get_game_variable(GameVariable.HEALTH)
        self.last_variables[GameVariable.KILLCOUNT] = self.game.get_game_variable(GameVariable.KILLCOUNT)
        self.last_variables[GameVariable.FRAGCOUNT] = self.game.get_game_variable(GameVariable.FRAGCOUNT)
        self.last_variables[GameVariable.SELECTED_WEAPON_AMMO] = self.game.get_game_variable(
            GameVariable.SELECTED_WEAPON_AMMO)

        if old_variables == {}:
            return r

        diff_dict = {k: old_variables[k] - self.last_variables[k] for k in old_variables.keys()}

        # Health
        if diff_dict[GameVariable.HEALTH] < 0:  # Old Health less than new health
            r += (diff_dict[GameVariable.HEALTH] * -0.04)
        else:  # old health > new health
            r -= (diff_dict[GameVariable.HEALTH] * 0.03)

        # Frag count
        if diff_dict[GameVariable.FRAGCOUNT] < 0:  # Frags when up
            r += diff_dict[GameVariable.FRAGCOUNT] * -2.5
        else:  # frags went down
            r += diff_dict[GameVariable.FRAGCOUNT] * -0.5  # mixes with death penalty

        if diff_dict[GameVariable.FRAGCOUNT] > 0:  # Old frags > New frags
            self.suicide_count += 1  # INCREMENT EACH TIME FRAG COUNT DECREASES
        elif diff_dict[GameVariable.FRAGCOUNT] < 0:
            self.kill_count += 1  # INCREMENT EACH TIME FRAGS INCREASES

        # Ammo
        if diff_dict[GameVariable.SELECTED_WEAPON_AMMO] < 0:  # Old Ammo < New Ammo
            r += (diff_dict[GameVariable.SELECTED_WEAPON_AMMO] * -0.15)
        # else:
        #     r -= (diff_dict[GameVariable.SELECTED_WEAPON_AMMO] * 0.04)

        # Displacement -- just encouraging movement
        last_place = self.position_buffer[0]
        distance_moved = distance.euclidean(last_place, current_position) * 4e-5
        r += distance_moved
        if len(self.position_buffer) > 1:
            self.position_buffer = self.position_buffer[1:]

        return r

    def update(self):
        self.s_t = self.s_t1

    def get_action_size(self):
        return len(self.real_actions)

    @property
    def sees_enemy(self):
        return any(x.object_name == "DoomPlayer" and x.object_id != 0 for x in self.labels)

    @property
    def labels_buffer(self):
        return self.game.get_state().labels_buffer

    @property
    def labels(self):
        return self.game.get_state().labels

    @property
    def max_episode_length(self):
        return self.game.get_episode_timeout()

    @property
    def avaliable_ammo(self):
        return self.game.get_game_variable(
            GameVariable.SELECTED_WEAPON_AMMO)
