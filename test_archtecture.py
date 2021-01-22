import numpy as np
import tensorflow as tf
import time

from architecture.navigation import Navigation


def test_navigation():
    nav = Navigation()
    #nav(np.zeros([1, 15]))
    nav._set_inputs(np.zeros([1, 15]))


test_navigation()
