import torch
import argparse
import torch.multiprocessing as mp

from model import Network
from learner import learner
from actor import actor
from environment import Atari, EnvironmentProxy, get_action_size

if __name__ == '__main__':
    mp.set_start_method('spawn')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actors", type=int, default=8,
                        help="the number of actors to start, default is 8")
    parser.add_argument("--seed", type=int, default=123,
                        help="the seed of random, default is 123")
    parser.add_argument("--game_name", type=str, default='breakout',
                        help="the name of atari game, default is breakout")
    parser.add_argument('--length', type=int, default=20,
                        help='Number of steps to run the agent')
    parser.add_argument('--total_steps', type=int, default=80000000,
                        help='Number of steps to run the agent')
    parser.add_argument('--batch_size', type=int, default=32,
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
    action_size = get_action_size(Atari, env_args)
    args.action_size = action_size
    model = Network(action_size=args.action_size)
    model.share_memory()

    if torch.cuda.is_available():
        model.cuda()
    envs = [EnvironmentProxy(Atari, env_args)
            for idx in range(args.actors)]

    learner = mp.Process(target=learner, args=(model, data, args))
    actors = [mp.Process(target=actor, args=(idx, model, data, envs[idx], args))
              for idx in range(args.actors)]
    learner.start()
    [actor.start() for actor in actors]
    [actor.join() for actor in actors]
    learner.join()
