import numpy as np
import gym
import random

import torch

from collections import namedtuple
from collections import deque

from agents import Simple_DQNAgent

Observations = namedtuple("Observations", ["s", "a", "r", "n_s", "d"])

NUM_EPISODES = 10000
MAX_STEPS = 200

class Environment:

    def __init__(self, ENV, batch_size, gamma):
        self.env = gym.make(ENV)
        self.num_states = self.env.observation_space.shape[0]
        self.num_actions = self.env.action_space.n
        self.batch_size = batch_size
        self.gamma = gamma
        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'

        self.agent = Simple_DQNAgent(self.num_states, self.num_actions, self.batch_size, self.gamma, self.device)
        self.total_step = np.zeros(10)

    def run(self):
        complete_episodes = 0
        episode_final = False
        frames = []
        for episode in range(NUM_EPISODES):
                observation = self.env.reset()
                state = observation
                state = torch.from_numpy(state).type(torch.FloatTensor)

                state = torch.unsqueeze(state, 0)

                for step in range(MAX_STEPS):
                    if episode_final is True:
                        """framesに各時刻の画像を追加していく"""
                        frames.append(self.env.render(mode='rgb_array'))

                    action = self.agent.policy(state, episode)

                    observation_next, _, done, _ = self.env.step(action[0, 0])

                    if done:
                        state_next = None  # 次の状態はないので、Noneを格納
                        self.total_step = np.hstack(
                            (self.total_step[1:], step + 1))  # step数を保存
                        if step < 195:
                            reward = torch.FloatTensor(
                                [-1.0])  # 途中でこけたら罰則として報酬-1を与える
                            self.complete_episodes = 0  # 連続成功記録をリセット
                        else:
                            reward = torch.FloatTensor([1.0])  # 立ったまま終了時は報酬1を与える
                            self.complete_episodes = self.complete_episodes + 1  # 連続記録を更新
                    else:
                        reward = torch.FloatTensor([0.0])  # 普段は報酬0
                        state_next = observation_next  # 観測をそのまま状態とする
                        state_next = torch.from_numpy(state_next).type(
                            torch.FloatTensor)  # numpyとPyTorchのテンソルに

                        # テンソルがsize 4になっているので、size 1x4に変換
                        state_next = torch.unsqueeze(state_next, 0)

                    # メモリに経験を追加
                    self.agent.memorize(state, action, state_next, reward)

                    # Experience ReplayでQ関数を更新する
                    self.agent.train_loop()

                    # 観測の更新
                    state = state_next

                    # 終了時の処理
                    if done:
                        print('%d Episode: Finished after %d steps：10Average = %.1lf' % (
                            episode, step + 1, self.total_step.mean()))
                        break

                if episode_final is True:
                    # 動画を保存と描画
                    #display_frames_as_gif(frames)
                    break

                # 10連続で200step立ち続けたら成功
                if self.complete_episodes >= 10:
                    print('10回連続成功')
                    episode_final = True  # 次の試行を描画を行う最終試行とする

def main():
    cartpole_env = Environment('CartPole-v0', 32, 0.99)
    cartpole_env.run()

main()
