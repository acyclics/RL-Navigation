import os
import sys
import time
import pickle
import tensorflow as tf
import numpy as np
from datetime import datetime
from copy import deepcopy

from environment.envs.icra import make_env
from environment.envhandler import EnvHandler
from environment.viewer.monitor import Monitor
from architecture.navigation import Navigation


def load_main_player(mp_file):
    if os.path.isfile(mp_file):
        with open(mp_file, 'rb') as f:
            return pickle.load(f)
    else:
        raise Exception('Main player file does not exist')


def manual_evaluate():
    # Build network architecture
    n_agents = 4
    nav = Navigation(1, False)
    nav.call_build()
    model_save_path = os.path.join(".", "data", "nav_model.npy")
    weights = np.load(model_save_path, allow_pickle=True)
    nav.set_weights(weights)
    # Setup environment and vid monitor
    #video_dir = os.path.join(os.getcwd(), "data", "vids", datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    #eval_env = Monitor(make_env(env_no=100), video_dir, video_callable=lambda episode_id:True, force=True)
    eval_env = make_env(env_no=100, add_bullets_visual=True)
    eval_env = EnvHandler(eval_env)
    obs = eval_env.reset()
    # Visualize either the agent's POV or real-time POV
    nsteps = 1000000
    RT_POV = True
    # Load main agent
    for _ in range(nsteps):
        actions, neglogp, entropy, value, mean, logstd = nav(np.expand_dims(obs, axis=0))
        agent_actions = {'action_movement': np.array([ actions[0, 0:3],
                                                           actions[0, 3:6],
                                                           [0.0, 0.0, 0.0],
                                                           [0.0, 0.0, 0.0] ])}
        obs, rewards, dones, infos = eval_env.step(agent_actions)
        #if not RT_POV:
        eval_env.render(mode='human')


manual_evaluate()
