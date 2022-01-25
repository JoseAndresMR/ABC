import os
from distutils import util
from pynput.keyboard import Listener
import threading
from torch.utils.tensorboard import SummaryWriter


class Environment(object):
    """ Parent class for every type of environment. Manages their common information and functions. """

    def __init__(self, id, log_path: str, use_kb_render = False):
        self.render_flag = False
        if use_kb_render:
            keyboard_input_thread = threading.Thread(target=self.keyboard_input_thread_fn)
            keyboard_input_thread.start()
        self.tensorboard_writer = SummaryWriter(
            os.path.join(log_path, "envs", "{}".format(id)))

    def get_environment_info(self):
        return self.env_info

    def set_action(self, action):
        self.actions = action

    def keyboard_input_thread_fn(self):
        def keyboard_pressed(key):
            if key.char == "r":
                self.render_flag = not self.render_flag

        keyboard_listener = Listener(on_press=keyboard_pressed)
        keyboard_listener.start()
        keyboard_listener.join()
