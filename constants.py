# -*- coding: utf-8 -*-

LOCAL_T_MAX = 40  # repeat step size
RMSP_ALPHA = 0.99  # decay parameter for RMSProp
RMSP_EPSILON = 0.1  # epsilon parameter for RMSProp
CHECKPOINT_DIR = 'checkpoints_deathmatch_fixed19'
LOG_FILE = 'tmp/a3c_log_doom_deathmatch_fixed19/log_'
INITIAL_ALPHA_LOW = 1e-6  # log_uniform low limit for learning rate
INITIAL_ALPHA_HIGH = 1e-4  # log_uniform high limit for learning rate
FRAME_SKIP = 4

PARALLEL_SIZE = 16  # parallel thread size
ROM = "pong.bin"  # action size = 3
ACTION_SIZE = 36  # action size

INITIAL_ALPHA_LOG_RATE = 0.4226  # log_uniform interpolate rate for learning rate (around 7 * 10^-4)
GAMMA = 0.99  # discount factor for rewards
ENTROPY_BETA = 0.01  # entropy regularization constant
MAX_TIME_STEP = 10 * 10 ** 7
GRAD_NORM_CLIP = 40.0  # gradient norm clipping
USE_GPU = True  # To use GPU, set True
USE_LSTM = True  # True for A3C LSTM, False for A3C FF
