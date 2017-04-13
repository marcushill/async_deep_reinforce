from time import sleep

import sys
from vizdoom.vizdoom import DoomGame, ScreenResolution, Mode


def replay_game(save_file):
    game = DoomGame()
    game.load_config("scenarios/cig.cfg")
    game.set_screen_resolution(ScreenResolution.RES_800X600)
    game.set_render_hud(True)
    game.set_mode(Mode.SPECTATOR)
    game.add_game_args("-host 1 -deathmatch +timelimit 1.0 "
                       "+sv_forcerespawn 1 +sv_noautoaim 1 +sv_respawnprotect 1 +sv_spawnfarthest 1")
    game.init()
    game.replay_episode(save_file)
    while not game.is_episode_finished():
        game.advance_action()
        sleep(100)
    game.close()


if __name__ == '__main__':
    replay_game(sys.argv[0])
