import scipy
import scipy.misc
from vizdoom.vizdoom import DoomGame, ScreenResolution, ScreenFormat, Button, GameVariable, Mode
import numpy as np


class DoomGameState:
    def __init__(self, scenario_path="scenarios/defend_center.cfg"):
        self.reward = 0
        game = DoomGame()
        game.load_config(scenario_path)  # This corresponds to the simple task we will pose our agent\
        game.set_window_visible(True)
        game.init()
        self.game = game

        # generates the the actual action arrays [True, False, False], etc for each action...
        # In preparation for actually using config files....
        self.real_actions = [[i == j for i in range(game.get_available_buttons_size())]
                             for j in range(game.get_available_buttons_size())]

        self.reset()

    def _process_frame(self, action, reshape):
        if action is not None:
            reward = self.game.make_action(action) / 100.0
        else:
            reward = None

        frame = self.game.get_state().screen_buffer
        if frame is None:
            return reward, None
        frame = frame[10:-10, 30:-30]
        frame = scipy.misc.imresize(frame, [84, 84])

        if reshape:
            frame = np.reshape(frame, (84, 84, 1))

        frame = frame.astype(np.float32)
        frame *= (1.0 / 255.0)
        return reward, frame

    @property
    def terminal(self):
        return self.game.is_episode_finished()

    def reset(self):
        self.reward = 0

    def start(self, name=""):
        self.game.new_episode(name)
        _, screen = self._process_frame(None, False)
        self.s_t = np.stack((screen, screen, screen, screen), axis=2)

    def process(self, action):
        real_action = self.real_actions[action]

        reward, frame = self._process_frame(real_action, True)
        self.reward = reward

        # self.s_t is the state over time an 84x84x4 3-dimensional matrix
        # self.s_t1 is taking the original s_t array which you can think of as a
        # 1-dimensional array of 4 frames, it's slicing the array so that it is now of length 3
        # and then it appends a new frame to the end of this 1-dimensional array
        if frame is not None:
            self.s_t1 = np.append(self.s_t[:, :, 1:], frame, axis=2)

    def update(self):
        self.s_t = self.s_t1

    def get_action_size(self):
        return len(self.real_actions)

    @property
    def max_episode_length(self):
        return self.game.get_episode_timeout()
