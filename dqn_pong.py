import os

from lib import wrappers
from lib import dqn_model
from lib import dqn_loss, double_dqn_loss

import matplotlib.pyplot as plt

import argparse
import time
import numpy as np
import collections

import torch
import torch.optim as optim

# from torch.utils.tensorboard import SummaryWriter

from utils import alpha_sync, load_hyperparams

Experience = collections.namedtuple('Experience', field_names=['state', 'action', 'reward', 'done', 'new_state'])


class ExperienceBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=int(capacity))

    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, dones, next_states = zip(*[self.buffer[idx] for idx in indices])
        return np.array(states), np.array(actions), np.array(rewards, dtype=np.float32), \
               np.array(dones, dtype=np.uint8), np.array(next_states)


class Agent:
    def __init__(self, env, exp_buffer):
        self.env = env
        self.exp_buffer = exp_buffer
        self._reset()

    def _reset(self):
        self.state = env.reset()
        self.total_reward = 0.0

    def play_step(self, net, epsilon=0.0, device="cpu"):
        done_reward = None

        if np.random.random() < epsilon:
            action = env.action_space.sample()
        else:
            state_a = np.array([self.state], copy=False)
            state_v = torch.tensor(state_a).to(device)
            q_vals_v = net(state_v)
            _, act_v = torch.max(q_vals_v, dim=1)
            action = int(act_v.item())

        # do step in the environment
        new_state, reward, is_done, _ = self.env.step(action)
        self.total_reward += reward

        exp = Experience(self.state, action, reward, is_done, new_state)
        self.exp_buffer.append(exp)
        self.state = new_state
        if is_done:
            done_reward = self.total_reward
            self._reset()
        return done_reward


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="PongNoFrameskip-v4",
                        help="Name of the environment")
    parser.add_argument("-n", "--network", default='duelling-dqn', help="DQN type - one of dqn, double-dqn and duelling-dqn")
    parser.add_argument("-c", "--config_file", default="config/pong.yaml", help="Config file with hyper-parameters")
    args = parser.parse_args()

    double_r100 = list()
    double_epsilon = list()
    double_reward = list()
    double_speed = list()

    single_r100 = list()
    single_epsilon = list()
    single_reward = list()
    single_speed = list()

    duelling_r100 = list()
    duelling_epsilon = list()
    duelling_reward = list()
    duelling_speed = list()

    networks = ['single-dqn', 'double-dqn', 'duelling-dqn']

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Training on GPU")
    else:
        device = torch.device("cpu")
        print("Training on CPU")

    params = load_hyperparams(args)

    for network in networks:
        env = wrappers.make_env(args.env)
        if args.network == 'duelling-dqn':
            net = dqn_model.DuellingDQN(env.observation_space.shape, params["hidden_size"], env.action_space.n).to(device)
            tgt_net = dqn_model.DuellingDQN(env.observation_space.shape, params["hidden_size"], env.action_space.n).to(device)
        else:
            net = dqn_model.DQN(env.observation_space.shape, params["hidden_size"], env.action_space.n).to(device)
            tgt_net = dqn_model.DQN(env.observation_space.shape, params["hidden_size"], env.action_space.n).to(device)

        if args.network == 'double-dqn':
            loss_calc = double_dqn_loss
        else:
            loss_calc = dqn_loss

        if not os.path.exists(params['save_path']):
            os.makedirs(params['save_path'])

        # writer = SummaryWriter(comment="-" + args.env)
        print(net)

        buffer = ExperienceBuffer(params["replay_size"])
        agent = Agent(env, buffer)
        epsilon = params["epsilon_start"]

        optimizer = optim.Adam(net.parameters(), lr=params["learning_rate"])
        total_rewards = []
        frame_idx = 0
        ts_frame = 0
        ts = time.time()
        best_mean_reward = 19

        while True:
            frame_idx += 1
            epsilon = max(params["epsilon_final"], params["epsilon_start"] - frame_idx / params["epsilon_decay_last_frame"])

            reward = agent.play_step(net, epsilon, device=device)
            if reward is not None:
                total_rewards.append(reward)
                speed = (frame_idx - ts_frame) / (time.time() - ts)
                ts_frame = frame_idx
                ts = time.time()
                mean_reward = np.mean(total_rewards[-100:])
                print("%d: done %d games, R100 %.3f, R: %.3f, eps %.2f, speed %.2f f/s" % (
                    frame_idx, len(total_rewards), mean_reward, reward, epsilon,
                    speed
                ))
                if network == 'single-dqn':
                    single_r100.append(mean_reward)
                    single_epsilon.append(epsilon)
                    single_reward.append(reward)
                    single_speed.append(speed)
                elif network == 'double-dqn':
                    double_r100.append(mean_reward)
                    double_epsilon.append(epsilon)
                    double_reward.append(reward)
                    double_speed.append(speed)
                else:
                    duelling_r100.append(mean_reward)
                    duelling_epsilon.append(epsilon)
                    duelling_reward.append(reward)
                    duelling_speed.append(speed)
                # writer.add_scalar("epsilon", epsilon, frame_idx)
                # writer.add_scalar("speed", speed, frame_idx)
                # writer.add_scalar("reward_100", mean_reward, frame_idx)
                # writer.add_scalar("reward", reward, frame_idx)
                if best_mean_reward is None or best_mean_reward < mean_reward:
                    torch.save(net.state_dict(), f"{params['save_path']}/{args.env}-best.dat")
                    if best_mean_reward is not None:
                        print("Best mean reward updated %.3f -> %.3f, model saved" % (best_mean_reward, mean_reward))
                    best_mean_reward = mean_reward
                if mean_reward > params["stopping_reward"]:
                    print("Solved in %d frames!" % frame_idx)
                    break

            if len(buffer) < params["replay_start_size"]:
                continue

            if params['alpha_sync']:
                alpha_sync(net, tgt_net, alpha=1 - params['tau'])
            elif frame_idx % params['target_net_sync'] == 0:
                tgt_net.load_state_dict(net.state_dict())

            optimizer.zero_grad()
            batch = buffer.sample(params["batch_size"])
            loss_t = loss_calc.calc_loss(batch, net, tgt_net, params["gamma"], device=device)
            loss_t.backward()
            optimizer.step()
        # writer.close()
    
    plt.clf()
    plt.figure(figsize=(10, 5))
    plt.plot(single_r100, label='Single DQN')
    plt.plot(double_r100, label='Double DQN')
    plt.plot(duelling_r100, label='Duelling DQN')
    plt.title('Pong R100 against Frame for Varying Networks')
    plt.xlabel('Frame')
    plt.ylabel('R100')
    plt.legend()
    plt.grid(True)
    plt.savefig('results/pong-r100.png')

    plt.clf()
    plt.figure(figsize=(10, 5))
    plt.plot(single_epsilon, label='Single DQN')
    plt.plot(double_epsilon, label='Double DQN')
    plt.plot(duelling_epsilon, label='Duelling DQN')
    plt.title('Pong Epsilon against Frame for Varying Networks')
    plt.xlabel('Frame')
    plt.ylabel('Epsilon')
    plt.legend()
    plt.grid(True)
    plt.savefig('results/pong-epsilon.png')

    plt.clf()
    plt.figure(figsize=(10, 5))
    plt.plot(single_speed, label='Single DQN')
    plt.plot(double_speed, label='Double DQN')
    plt.plot(duelling_speed, label='Duelling DQN')
    plt.title('Pong Speed against Frame for Varying Networks')
    plt.xlabel('Frame')
    plt.ylabel('Speed')
    plt.legend()
    plt.grid(True)
    plt.savefig('results/pong-speed.png')

    plt.clf()
    plt.figure(figsize=(10, 5))
    plt.plot(single_reward, label='Single DQN')
    plt.plot(double_reward, label='Double DQN')
    plt.plot(duelling_reward, label='Duelling DQN')
    plt.title('Pong Reward against Frame for Varying Networks')
    plt.xlabel('Frame')
    plt.ylabel('Episode Reward')
    plt.legend()
    plt.grid(True)
    plt.savefig('results/pong-reward.png')
