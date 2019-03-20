import random
from collections import deque
from functools import partial
import torch.multiprocessing as mp
import atari_py
import torch
import cv2

COMMAND_RESET = 0
COMMAND_STEP = 1
COMMAND_TERMINATE = 2


class Env:
    def __init__(self, game_name, seed, max_episode_length=1e10, history_length=4, reward_clip=1, device='cpu'):
        self.device = device
        self.ale = atari_py.ALEInterface()
        self.ale.setInt('random_seed', seed)
        self.ale.setInt('max_num_frames_per_episode', max_episode_length)
        self.ale.setFloat('repeat_action_probability', 0)  # Disable sticky actions
        self.ale.setInt('frame_skip', 0)
        self.ale.setBool('color_averaging', False)
        self.ale.loadROM(atari_py.get_game_path(game_name))  # ROM loading must be done after setting options
        actions = self.ale.getMinimalActionSet()
        self.actions = dict(zip(range(len(actions)), actions))
        self.reward_clip = reward_clip
        self.lives = 0  # Life counter (used in DeepMind training)
        self.life_termination = False  # Used to check if resetting only from loss of life
        self.window = history_length  # Number of frames to concatenate
        self.state_buffer = deque([], maxlen=history_length)
        self.training = True  # Consistent with model training mode
        self.viewer = None

    def _get_state(self):
        state = cv2.resize(self.ale.getScreenGrayscale(), (84, 84), interpolation=cv2.INTER_LINEAR)
        return torch.tensor(state, dtype=torch.float32, device=self.device).div_(255)

    def _reset_buffer(self):
        for _ in range(self.window):
            self.state_buffer.append(torch.zeros(84, 84, device=self.device))

    def reset(self):
        if self.life_termination:
            self.life_termination = False  # Reset flag
            self.ale.act(0)  # Use a no-op after loss of life
        else:
            # Reset internals
            self._reset_buffer()
            self.ale.reset_game()
            # Perform up to 30 random no-ops before starting
            for _ in range(random.randrange(30)):
                self.ale.act(0)  # Assumes raw action 0 is always no-op
                if self.ale.game_over():
                    self.ale.reset_game()
        # Process and return "initial" state
        observation = self._get_state()
        self.state_buffer.append(observation)
        self.lives = self.ale.lives()
        return torch.stack(list(self.state_buffer), 0)

    def step(self, action):
        # Repeat action 4 times, max pool over last 2 frames
        frame_buffer = torch.zeros(2, 84, 84, device=self.device)
        reward, done = 0, False
        for t in range(4):
            reward += self.ale.act(self.actions.get(action))
            if t == 2:
                frame_buffer[0] = self._get_state()
            elif t == 3:
                frame_buffer[1] = self._get_state()
            done = self.ale.game_over()
            if done:
                break
        observation = frame_buffer.max(0)[0]
        self.state_buffer.append(observation)
        # Detect loss of life as terminal in training mode
        if self.training:
            lives = self.ale.lives()
            if self.lives > lives > 0:  # Lives > 0 for Q*bert
                self.life_termination = not done  # Only set flag when not truly done
                done = True
            self.lives = lives
        # Return state, reward, done
        reward = max(min(reward, self.reward_clip), -self.reward_clip)
        return torch.stack(list(self.state_buffer), 0), reward, done

    # Uses loss of life as terminal signal
    def train(self):
        self.training = True

    # Uses standard terminal signal
    def eval(self):
        self.training = False

    def action_space(self):
        return len(self.actions)

    def render(self):
        from gym.envs.classic_control import rendering
        if self.viewer is None:
            self.viewer = rendering.SimpleImageViewer()
        self.viewer.imshow(self.ale.getScreenRGB2())
        return self.viewer.isopen

        # cv2.imshow('screen', self.ale.getScreenRGB()[:, :, ::-1])
        # cv2.waitKey(1)

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None


def worker(conn, env):
    env = env.x()
    while True:
        command, arg = conn.recv()
        if command == COMMAND_RESET:
            obs = env.reset()
            conn.send(obs)
        elif command == COMMAND_STEP:
            obs, reward, terminal = env.step(arg)
            conn.send([obs, reward, terminal])
        elif command == COMMAND_TERMINATE:
            break
        else:
            print("bad command: {}".format(command))
    env.close()
    conn.close()


class CloudpickleWrapper(object):
    def __init__(self, x):
        self.x = x

    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.x)

    def __setstate__(self, ob):
        import pickle
        self.x = pickle.loads(ob)


class Environment(object):
    def __init__(self, env_args):
        super(Environment, self).__init__()
        env = partial(Env, **env_args)
        self.env_args = env_args
        self.conn, child_conn = mp.Pipe()
        self.proc = mp.Process(target=worker, args=(child_conn, CloudpickleWrapper(env)))
        self.proc.start()

    @staticmethod
    def get_action_size(env_args):
        env = Env(**env_args)
        action_size = env.action_space()
        env.close()
        del env
        return action_size

    def reset(self):
        self.conn.send([COMMAND_RESET, 0])
        return self.conn.recv()

    def close(self):
        self.conn.send([COMMAND_TERMINATE, None])
        self.conn.close()
        self.proc.join()
        print("Environment closed")

    def step(self, actions):
        self.conn.send([COMMAND_STEP, actions])
        state, reward, terminal = self.conn.recv()
        return state, reward, terminal
