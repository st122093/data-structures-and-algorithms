import time
import pandas as pd
import os
from shutil import copyfile
import copy
import random
from binance.client import Client
import numpy as np
from collections import deque
import pickle

import torch
from torch import nn
from torch.nn import functional as F

LOAD_LAST_DQN = False

LR = 0.0001
GAMMA = 0.9

PLAY_INTERVAL = 900

EPSILON = 1
EPSILON_DECAY = 0.99999
MIN_EPSILON = 0.01

MAX_REPLAY_MEMORY = 100000
REPLAY_SIZE = 32

class BitcoinEnv():
    def __init__(self):
        API_KEY = 'brxMsvSc33RpzqsHEFD6bFpRcEjIN2pKMdlwYIhOPcsNzw6BqxPML45fDtMCIY1K'
        SECRET_KEY = '02Zruc8AMSQqcl3xT3XLduLB4ym9wtpF8RRWsZKXkx2xhTxocI6YoUqHlnyXBHnk'

        client = Client(API_KEY, SECRET_KEY)

        his = client.get_historical_klines('BTCUSDT', Client.KLINE_INTERVAL_30MINUTE, '1 Oct, 2020', '1 Nov, 2021')
        self.open_price = (np.array(his)[0:-1, 1]).astype(np.float32)
        del his

        price_mean = 39065.56
        price_std = 15103.736

        self.open_price = (self.open_price-price_mean)/price_std

    def reset(self, sp=None):
        if sp is None:
            sp = random.randint(335, 17381)

        self.sp = sp

    def step_buy(self, action):
        future_price = self.open_price[self.sp:self.sp+96]
        future_min = np.min(future_price)
        future_max = np.max(future_price)
        current_price = self.open_price[self.sp]
        
        if action == 0:
            reward = (2/(future_min-future_max)*(current_price-future_max) - 1) * -1
        elif action == 1:
            reward = 2/(future_min-future_max)*(current_price-future_max) - 1

        self.sp += 1

        if self.sp == 17382 or self.sp == 18870:
            end = True
        else:
            end = False

        return reward, end

class NNModel_Buy(nn.Module):
    def __init__(self):
        super().__init__()
        self.neural1 = nn.Linear(336, 512)
        self.neural2 = nn.Linear(512, 512)
        self.neural3 = nn.Linear(512, 512)
        self.neural4 = nn.Linear(512, 2)

    def forward(self, x):
        h = F.relu(self.neural1(x))
        h = F.relu(self.neural2(h))
        h = F.relu(self.neural3(h))
        h = self.neural4(h)
        return h
    
class DQN():
    def __init__(self):
        self.criterion_buy = nn.SmoothL1Loss()
        self.model_buy = NNModel_Buy().cuda()

        if LOAD_LAST_DQN:
            self.model_buy.load_state_dict(torch.load('last_dqn_buy.pth'))

        self.optimizer_buy = torch.optim.Adam(self.model_buy.parameters(), LR)

    def predict_buy(self, s):
        s = torch.FloatTensor(s).cuda()
        self.model_buy.eval()
        with torch.no_grad():
            return self.model_buy(s)

    def replay_buy(self):
        if len(memory.memory_buy) >= REPLAY_SIZE:
            replay_data = memory.sample_buy()
            states = []
            td_targets = []

            for data in replay_data:
                states.append(copy.deepcopy(data['state']))
                q_values = self.predict_buy(data['state'])

                if data['end']:
                    q_values[data['action']] = data['reward']
                else:
                    q_values_next = self.predict_buy(data['next_state'])
                    q_values[data['action']] = data['reward'] + GAMMA * torch.max(q_values_next).item()

                td_targets.append(q_values)

            loss = self.update_buy(states, td_targets)
            return loss

    def update_buy(self, s_in, y_in):
        s = np.zeros([REPLAY_SIZE, 336], dtype=np.float32)
        y = torch.zeros([REPLAY_SIZE, 2], dtype=torch.float32).cuda()
        
        for n in range(REPLAY_SIZE):
            s[n] = s_in[n]
            y[n] = y_in[n]

        s = torch.FloatTensor(s).cuda()
        
        y_pred = self.model_buy(s)
        self.model_buy.train()
        loss = self.criterion_buy(y_pred, y)
        self.optimizer_buy.zero_grad()
        loss.backward()
        self.optimizer_buy.step()

        return loss.item()

class Memory():
    def __init__(self):
        if LOAD_LAST_DQN:
            with open('memory_buy.dq', 'rb') as handle:
                self.memory_buy = pickle.load(handle)
        else:
            self.memory_buy = deque(maxlen=MAX_REPLAY_MEMORY)

    def append_buy(self, state, action, next_state, reward, end):
        if len(self.memory_buy) >= MAX_REPLAY_MEMORY:
            try:
                os.remove('memory/buy/'+self.memory_buy[0])
            except:
                pass
        timestamp = str(time.time())
        self.memory_buy.append(timestamp)
        data = {'state': state, 'action': int(action), 'next_state': next_state, 'reward': float(reward), 'end': bool(end)}
        with open('memory/buy/'+timestamp, 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open('memory_buy.dq', 'wb') as handle:
            pickle.dump(self.memory_buy, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def sample_buy(self):
        replay_data_timestamp = random.sample(self.memory_buy, REPLAY_SIZE)
        replay_data = []
        for n in range(REPLAY_SIZE):
            with open('memory/buy/'+replay_data_timestamp[n], 'rb') as handle:
                data = pickle.load(handle)
            replay_data.append(data)
        return replay_data

def gen_epsilon_greedy_policy_buy():
    def policy_function(state):
        if random.random() < EPSILON:
            return random.randint(0, 1)
        else:
            q_values = estimator.predict_buy(copy.deepcopy(state))
            return torch.argmax(q_values).item()
    return policy_function

def q_learning():
    global df
    
    step = 1
    reward_step_buy = []
    epsilon = EPSILON
    best_sum_play_score_buy = -99999
    train_loss_buy = []
    
    policy_buy = gen_epsilon_greedy_policy_buy()
    
    while True:
        env.reset()
        state = env.open_price[env.sp-335:env.sp+1]

        while True:
            action = policy_buy(state)

            reward, end = env.step_buy(action)

            reward_step_buy.append(reward)
            next_state = env.open_price[env.sp-335:env.sp+1]

            memory.append_buy(state, action, next_state, reward, end)

            loss = estimator.replay_buy()
            if loss is not None:
                train_loss_buy.append(loss)

            if step % PLAY_INTERVAL == 0:
                torch.save(estimator.model_buy.state_dict(), 'last_dqn_buy.pth')

                play_reward_buy = []

                estimator.model_buy.eval()
                with torch.no_grad():
                    env.reset(17813)
                    state = env.open_price[env.sp-335:env.sp+1]

                    while True:
                        q_values = estimator.predict_buy(state)
                        action = torch.argmax(q_values).item()

                        reward, end = env.step_buy(action)

                        play_reward_buy.append(reward)

                        state = env.open_price[env.sp-335:env.sp+1]

                        if env.sp == 18870:
                            break

                curr_timestamp = time.time()
                curr_time = time.ctime(curr_timestamp)
                print(f'[{curr_time}] STEP: {step}, STEP REWARD BUY: {np.sum(reward_step_buy):.6f}, ', end='')
                print(f'LOSS BUY: {np.mean(train_loss_buy):.6f}, EPSILON: {epsilon:.4f}, ', end='')
                print(f'SUM PLAY SCORE BUY: {np.sum(play_reward_buy):.6f}', end='')

                if np.sum(play_reward_buy) > best_sum_play_score_buy:
                    best_sum_buy = True
                    best_sum_play_score_buy = np.sum(play_reward_buy)
                else:
                    best_sum_buy = False

                if best_sum_buy:
                    print(f' [BEST BUY]')

                    df = df.append({'DATE TIME': curr_time, 'STEP': step, 'STEP REWARD BUY': np.sum(reward_step_buy),
                                    'LOSS BUY': np.mean(train_loss_buy), 'EPSILON': epsilon,
                                    'SUM PLAY SCORE BUY': np.sum(play_reward_buy),
                                    'BEST SUM PLAY SCORE BUY': True,
                                    'TIMESTAMP': curr_timestamp}, ignore_index=True)
                    df.to_csv(f'{path_save_name}/history.csv', index=False)

                    in_save_path = os.listdir(f'{path_save_name}/best_buy')
                    save_best_path = f'{path_save_name}/best_buy/{len(in_save_path)}_{np.sum(play_reward_buy):.6f}'
                    os.mkdir(save_best_path)

                    torch.save(estimator.model_buy.state_dict(), f'{save_best_path}/model_buy.pth')

                else:
                    print('')

                    df = df.append({'DATE TIME': curr_time, 'STEP': step, 'STEP REWARD BUY': np.sum(reward_step_buy),
                                    'LOSS BUY': np.mean(train_loss_buy), 'EPSILON': epsilon,
                                    'SUM PLAY SCORE BUY': np.sum(play_reward_buy),
                                    'TIMESTAMP': curr_timestamp}, ignore_index=True)
                    df.to_csv(f'{path_save_name}/history.csv', index=False)

                reward_step_buy = []
                train_loss_buy = []
                step += 1
                epsilon = max(epsilon*EPSILON_DECAY, MIN_EPSILON)
                break
            
            step += 1
            epsilon = max(epsilon*EPSILON_DECAY, MIN_EPSILON)

            if end:
                break

            state = next_state
    
env = BitcoinEnv()
estimator = DQN()
memory = Memory()

df = pd.DataFrame(columns = ['DATE TIME', 'STEP', 'STEP REWARD BUY', 'LOSS BUY', 'EPSILON',
                             'SUM PLAY SCORE BUY',
                             'BEST SUM PLAY SCORE BUY', 'TIMESTAMP'])

curr_timestamp = time.time()
curr_time = time.ctime(curr_timestamp)
curr_time = curr_time.replace(':', '-')
curr_time = curr_time.replace('  ', '_')
curr_time = curr_time.replace(' ', '_')

if not os.path.isdir('save'):
    os.mkdir('save')

if not os.path.isdir('memory'):
    os.mkdir('memory')

if not os.path.isdir('memory/buy'):
    os.mkdir('memory/buy')

path_save_name = f'save/{len(os.listdir("save"))+1}_[{curr_time}][{os.path.basename(__file__)[:-3]}]'
os.mkdir(path_save_name)
os.mkdir(f'{path_save_name}/best_buy')

copyfile(os.path.basename(__file__), f'{path_save_name}/{os.path.basename(__file__)}')

q_learning()
