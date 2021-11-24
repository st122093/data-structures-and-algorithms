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

    def step_sell(self, action):
        future_price = self.open_price[self.sp:self.sp+96]
        future_min = np.min(future_price)
        future_max = np.max(future_price)
        current_price = self.open_price[self.sp]
        
        if action == 0:
            reward = 2/(future_min-future_max)*(current_price-future_max) - 1
        elif action == 1:
            reward = (2/(future_min-future_max)*(current_price-future_max) - 1) * -1

        self.sp += 1

        if self.sp == 17382 or self.sp == 18870:
            end = True
        else:
            end = False

        return reward, end

class NNModel_Sell(nn.Module):
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
        self.criterion_sell = nn.SmoothL1Loss()
        self.model_sell = NNModel_Sell().cuda()

        if LOAD_LAST_DQN:
            self.model_sell.load_state_dict(torch.load('last_dqn_sell.pth'))

        self.optimizer_sell = torch.optim.Adam(self.model_sell.parameters(), LR)

    def predict_sell(self, s):
        s = torch.FloatTensor(s).cuda()
        self.model_sell.eval()
        with torch.no_grad():
            return self.model_sell(s)

    def replay_sell(self):
        if len(memory.memory_sell) >= REPLAY_SIZE:
            replay_data = memory.sample_sell()
            states = []
            td_targets = []

            for data in replay_data:
                states.append(copy.deepcopy(data['state']))
                q_values = self.predict_sell(data['state'])

                if data['end']:
                    q_values[data['action']] = data['reward']
                else:
                    q_values_next = self.predict_sell(data['next_state'])
                    q_values[data['action']] = data['reward'] + GAMMA * torch.max(q_values_next).item()

                td_targets.append(q_values)

            loss = self.update_sell(states, td_targets)
            return loss

    def update_sell(self, s_in, y_in):
        s = np.zeros([REPLAY_SIZE, 336], dtype=np.float32)
        y = torch.zeros([REPLAY_SIZE, 2], dtype=torch.float32).cuda()
        
        for n in range(REPLAY_SIZE):
            s[n] = s_in[n]
            y[n] = y_in[n]

        s = torch.FloatTensor(s).cuda()
        
        y_pred = self.model_sell(s)
        self.model_sell.train()
        loss = self.criterion_sell(y_pred, y)
        self.optimizer_sell.zero_grad()
        loss.backward()
        self.optimizer_sell.step()

        return loss.item()

class Memory():
    def __init__(self):
        if LOAD_LAST_DQN:
            with open('memory_sell.dq', 'rb') as handle:
                self.memory_sell = pickle.load(handle)
        else:
            self.memory_sell = deque(maxlen=MAX_REPLAY_MEMORY)

    def append_sell(self, state, action, next_state, reward, end):
        if len(self.memory_sell) >= MAX_REPLAY_MEMORY:
            try:
                os.remove('memory/sell/'+self.memory_sell[0])
            except:
                pass
        timestamp = str(time.time())
        self.memory_sell.append(timestamp)
        data = {'state': state, 'action': int(action), 'next_state': next_state, 'reward': float(reward), 'end': bool(end)}
        with open('memory/sell/'+timestamp, 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open('memory_sell.dq', 'wb') as handle:
            pickle.dump(self.memory_sell, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def sample_sell(self):
        replay_data_timestamp = random.sample(self.memory_sell, REPLAY_SIZE)
        replay_data = []
        for n in range(REPLAY_SIZE):
            with open('memory/sell/'+replay_data_timestamp[n], 'rb') as handle:
                data = pickle.load(handle)
            replay_data.append(data)
        return replay_data

def gen_epsilon_greedy_policy_sell():
    def policy_function(state):
        if random.random() < EPSILON:
            return random.randint(0, 1)
        else:
            q_values = estimator.predict_sell(copy.deepcopy(state))
            return torch.argmax(q_values).item()
    return policy_function

def q_learning():
    global df
    
    step = 1
    reward_step_sell = []
    epsilon = EPSILON
    best_sum_play_score_sell = -99999
    train_loss_sell = []
    
    policy_sell = gen_epsilon_greedy_policy_sell()
    
    while True:
        env.reset()
        state = env.open_price[env.sp-335:env.sp+1]

        while True:
            action = policy_sell(state)

            reward, end = env.step_sell(action)

            reward_step_sell.append(reward)
            next_state = env.open_price[env.sp-335:env.sp+1]

            memory.append_sell(state, action, next_state, reward, end)

            loss = estimator.replay_sell()
            if loss is not None:
                train_loss_sell.append(loss)

            if step % PLAY_INTERVAL == 0:
                torch.save(estimator.model_sell.state_dict(), 'last_dqn_sell.pth')

                play_reward_sell = []

                estimator.model_sell.eval()
                with torch.no_grad():
                    env.reset(17813)
                    state = env.open_price[env.sp-335:env.sp+1]

                    while True:
                        q_values = estimator.predict_sell(state)
                        action = torch.argmax(q_values).item()

                        reward, end = env.step_sell(action)

                        play_reward_sell.append(reward)

                        state = env.open_price[env.sp-335:env.sp+1]

                        if env.sp == 18870:
                            break

                curr_timestamp = time.time()
                curr_time = time.ctime(curr_timestamp)
                print(f'[{curr_time}] STEP: {step}, STEP REWARD SELL: {np.sum(reward_step_sell):.6f}, ', end='')
                print(f'LOSS SELL: {np.mean(train_loss_sell):.6f}, EPSILON: {epsilon:.4f}, ', end='')
                print(f'SUM PLAY SCORE SELL: {np.sum(play_reward_sell):.6f}', end='')

                if np.sum(play_reward_sell) > best_sum_play_score_sell:
                    best_sum_sell = True
                    best_sum_play_score_sell = np.sum(play_reward_sell)
                else:
                    best_sum_sell = False

                if best_sum_sell:
                    print(f' [BEST SELL]')

                    df = df.append({'DATE TIME': curr_time, 'STEP': step, 'STEP REWARD SELL': np.sum(reward_step_sell),
                                    'LOSS SELL': np.mean(train_loss_sell), 'EPSILON': epsilon,
                                    'SUM PLAY SCORE SELL': np.sum(play_reward_sell),
                                    'BEST SUM PLAY SCORE SELL': True,
                                    'TIMESTAMP': curr_timestamp}, ignore_index=True)
                    df.to_csv(f'{path_save_name}/history.csv', index=False)

                    in_save_path = os.listdir(f'{path_save_name}/best_sell')
                    save_best_path = f'{path_save_name}/best_sell/{len(in_save_path)}_{np.sum(play_reward_sell):.6f}'
                    os.mkdir(save_best_path)

                    torch.save(estimator.model_sell.state_dict(), f'{save_best_path}/model_sell.pth')

                else:
                    print('')

                    df = df.append({'DATE TIME': curr_time, 'STEP': step, 'STEP REWARD SELL': np.sum(reward_step_sell),
                                    'LOSS SELL': np.mean(train_loss_sell), 'EPSILON': epsilon,
                                    'SUM PLAY SCORE SELL': np.sum(play_reward_sell),
                                    'TIMESTAMP': curr_timestamp}, ignore_index=True)
                    df.to_csv(f'{path_save_name}/history.csv', index=False)

                reward_step_sell = []
                train_loss_sell = []
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

df = pd.DataFrame(columns = ['DATE TIME', 'STEP', 'STEP REWARD SELL', 'LOSS SELL', 'EPSILON',
                             'SUM PLAY SCORE SELL',
                             'BEST SUM PLAY SCORE SELL', 'TIMESTAMP'])

curr_timestamp = time.time()
curr_time = time.ctime(curr_timestamp)
curr_time = curr_time.replace(':', '-')
curr_time = curr_time.replace('  ', '_')
curr_time = curr_time.replace(' ', '_')

if not os.path.isdir('save'):
    os.mkdir('save')

if not os.path.isdir('memory'):
    os.mkdir('memory')

if not os.path.isdir('memory/sell'):
    os.mkdir('memory/sell')

path_save_name = f'save/{len(os.listdir("save"))+1}_[{curr_time}][{os.path.basename(__file__)[:-3]}]'
os.mkdir(path_save_name)
os.mkdir(f'{path_save_name}/best_sell')

copyfile(os.path.basename(__file__), f'{path_save_name}/{os.path.basename(__file__)}')

q_learning()
