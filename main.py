import argparse
import torch.multiprocessing as _mp

from env import Environment
from learner import Learner
from model import Network
from actor import Actor

if __name__ == '__main__':
    mp = _mp.get_context('spawn')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actors", type=int, default=2,
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
    action_space = Environment.get_action_size(env_args)
    g_net = Network(action_space=action_space).share_memory()
    learner = Learner(g_net, data, args.gamma, args.batch_size, args.baseline_cost,
                      args.entropy_cost, args.lr, args.decay, args.momentum, args.epsilon)

    actors = [Actor(idx, g_net, data, env_args, action_space, args.length,
                    args.save_path, args.load_path)
              for idx in range(args.actors)]
    learner.start()
    [actor.start() for actor in actors]
    [actor.join() for actor in actors]
    learner.join()
