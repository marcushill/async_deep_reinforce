import scipy
import scipy.misc
from vizdoom.vizdoom import DoomGame, ScreenResolution, ScreenFormat, Button, GameVariable, Mode
import numpy as np


class DoomGameState:
    def __init__(self, scenario_path="scenarios/defend_center.cfg"):
        self.reward = 0
        game = DoomGame()
        game.load_config(scenario_path)  # This corresponds to the simple task we will pose our agent\
        # game.load_config("../../scenarios/cig.cfg")

        game.set_doom_map("map01")  # Limited deathmatch.
        # game.set_doom_map("map02")  # Full deathmatch.
        # game.set_window_visible(True)

        # Start multiplayer game only with your AI (with options that will be used in the competition, details in cig_host example).
        game.add_game_args("-host 1 -deathmatch +timelimit 1.0 "
                           "+sv_forcerespawn 1 +sv_noautoaim 1 +sv_respawnprotect 1 +sv_spawnfarthest 1")

        # Name your agent and select color
        # colors: 0 - green, 1 - gray, 2 - brown, 3 - red, 4 - light gray, 5 - light brown, 6 - light red, 7 - light blue
        game.add_game_args("+name AI +colorset 0")



        self.game = game

        # generates the the actual action arrays [True, False, False], etc for each action...
        # In preparation for actually using config files....
        self.real_actions = [[i == j for i in range(game.get_available_buttons_size())]
                             for j in range(game.get_available_buttons_size())]

        self.reset()
        game.init()

    def _process_frame(self, action, reshape):
        if action is not None:
            reward = self.game.make_action(action) / 100.0
        else:
            reward = None

        frame = self.game.get_state().screen_buffer
        if frame is None:
            return reward, None
        # print(frame.shape)
        frame = frame[:, 10:-10, 30:-30]
        frame = scipy.misc.imresize(frame, [84, 84])
        # print("After Resize: ", frame.shape)
        if reshape:
            frame = np.reshape(frame, (1, 84, 84, 3))

        frame = frame.astype(np.float32)
        frame *= (1.0 / 255.0)
        # print(frame.shape)

        return reward, frame

    @property
    def terminal(self):
        return self.game.is_episode_finished()

    def reset(self):
        self.reward = 0

    def start(self):
        self.game.new_episode()

        # Add specific number of bots
        # (file examples/bots.cfg must be placed in the same directory as the Doom executable file,
        # edit this file to adjust bots).
        self.game.send_game_command("removebots")
        for i in range(7):
            self.game.send_game_command("addbot")

        _, screen = self._process_frame(None, False)
        self.s_t = np.stack((screen, screen, screen, screen), axis=0)

    def process(self, action):
        real_action = self.real_actions[action]

        reward, frame = self._process_frame(real_action, True)
        self.reward = reward

        # self.s_t is the state over time an 84x84x4 3-dimensional matrix
        # self.s_t1 is taking the original s_t array which you can think of as a
        # 1-dimensional array of 4 frames, it's slicing the array so that it is now of length 3
        # and then it appends a new frame to the end of this 1-dimensional array
        if frame is not None:
            # print('s_t shape', self.s_t.shape)
            # print('frame shape', frame.shape)
            self.s_t1 = np.append(self.s_t[1:, :, :, :], frame, axis=0)

    def update(self):
        self.s_t = self.s_t1

    def get_action_size(self):
        return len(self.real_actions)

    @property
    def max_episode_length(self):
        return self.game.get_episode_timeout()
