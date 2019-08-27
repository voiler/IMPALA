import argparse

import torch

from model import Network
import torch.multiprocessing as mp
from learner import learner
from actor import actor
from utils import env_process, get_action_size, ParameterServer

if __name__ == '__main__':
    mp.set_start_method('spawn')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actors", type=int, default=4,
                        help="the number of actors to start, default is 8")
    parser.add_argument("--seed", type=int, default=123,
                        help="the seed of random, default is 123")
    parser.add_argument("--game_name", type=str, default='breakout',
                        help="the name of atari game, default is breakout")
    parser.add_argument('--length', type=int, default=20,
                        help='Number of steps to run the agent')
    parser.add_argument('--batch_size', type=int, default=2,
                        help='Number of steps to run the agent')
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="the discount factor, default is 0.99")
    parser.add_argument("--entropy_cost", type=float, default=0.00025,
                        help="Entropy cost/multiplier, default is 0.00025")
    parser.add_argument("--baseline_cost", type=float, default=.5,
                        help="Baseline cost/multiplier, default is 0.5")
    parser.add_argument("--lr", type=float, default=0.00048,
                        help="Learning rate, default is 0.00048")
    parser.add_argument("--decay", type=float, default=.99,
                        help="RMSProp optimizer decay, default is .99")
    parser.add_argument("--momentum", type=float, default=0,
                        help="RMSProp momentum, default is 0")
    parser.add_argument("--epsilon", type=float, default=.1,
                        help="RMSProp epsilon, default is 0.1")
    parser.add_argument('--save_path', type=str, default="./checkpoint.pt",
                        help='Set the path to save trained model parameters')
    parser.add_argument('--load_path', type=str, default="./checkpoint.pt",
                        help='Set the path to load trained model parameters')

    args = parser.parse_args()
    data = mp.Queue(maxsize=1)
    env_args = {'game_name': args.game_name, 'seed': args.seed}
    action_space = get_action_size(env_args)
    args.action_space = action_space
    ps = ParameterServer()
    model = Network(action_space=args.action_space)
    ps.push(model.state_dict())
    if torch.cuda.is_available():
        model.cuda()
    learner = mp.Process(target=learner, args=(model, data, ps, args))
    envs, conns = list(zip(*[env_process(env_args) for _ in range(args.actors)]))
    actors = [mp.Process(target=actor, args=(idx, ps, data, conns[idx], args))
              for idx in range(args.actors)]
    learner.start()
    [env.start() for env in envs]
    [actor.start() for actor in actors]
    [actor.join() for actor in actors]
    [env.join() for env in envs]
    learner.join()
