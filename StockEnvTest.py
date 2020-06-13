import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

data_1 = pd.read_csv('./Data/dow_jones_30_daily_price.csv')

equal_4711_list = list(data_1.tic.value_counts() == 4711)
names = data_1.tic.value_counts().index

# select_stocks_list = ['NKE','KO']
select_stocks_list = list(names[equal_4711_list])+['NKE','KO']

data_2 = data_1[data_1.tic.isin(select_stocks_list)][~data_1.datadate.isin(['20010912','20010913'])]

data_3 = data_2[['iid','datadate','tic','prccd','ajexdi']]

data_3['adjcp'] = data_3['prccd'] / data_3['ajexdi']

test_data = data_3[data_3.datadate > 20140000]

test_daily_data = []

for date in np.unique(test_data.datadate):
    test_daily_data.append(test_data[test_data.datadate == date])

iteration = 0


class StockEnvTest:
    # a simple multi-agent stock market environment
    def __init__(self, day = 0, money = 10, agent_count = 2, scope = 1):
        self.day = day
        self.agent_count = agent_count
        # buy or sell maximum 5 shares for n number of agents
        self.action_space = []
        for i in range(agent_count):
            # each entry ranges from -5 to +5 
            self.action_space.append([0 for j in range(28)])
        self.action_space = np.array(self.action_space)

        # [money]+[prices 1-28]+[owned shares 1-28] for n number of agents
        self.observation_space = []
        for i in range(agent_count):
            self.observation_space.append([0 for j in range(57)])
        self.observation_space = np.array(self.observation_space)

        # required monetary information of all stocks for that given day
        self.data = test_daily_data[self.day]
        
        # initial state
        agent_list = [10000] + self.data.adjcp.values.tolist() + [0 for j in range(28)]
        self.state = np.array([agent_list for i in range(agent_count)]) 

        # all the above defined variables are 2D lists. Row i = state of agent i

        # 1D list. Each column corresponds to an agent's termination condition
        self.terminal = np.array([False for i in range(agent_count)])
        
        # 1D list. Each column corresponds to an agent's reward
        self.reward = np.array([0 for i in range(agent_count)])

        # 1D list. Each column corresponds to an agent's asset memory
        self.asset_memory = [[10000] for i in range(self.agent_count)]
        
        self.reset()


    def _sell_stock(self, index, action, agent):    
        if self.state[agent][index+29] > 0:
            self.state[agent][0] += self.state[agent][index+1]*min(abs(action), self.state[agent][index+29])
            self.state[agent][index+29] -= min(abs(action), self.state[agent][index+29])
        else:
            pass
        
    def _buy_stock(self, index, action, agent):
        available_amount = self.state[agent][0] // self.state[agent][index+1]
        # print('available_amount:{}'.format(available_amount))
        self.state[agent][0] -= self.state[agent][index+1]*min(available_amount, action)
        # print(min(available_amount, action))
        self.state[agent][index+29] += min(available_amount, action)
    
    def step(self, actions):

        """
        # print(self.day)
        
        # print(actions)

        if self.terminal:
            for i in range(self.agent_count):
                plt.plot(self.asset_memory[i],'r')
                plt.savefig('/home/prahlad/asset_memory_training_{}.png'.format(i))
                plt.close()
            #print("total_reward:{}".format(self.state[0]+ sum(np.array(self.state[1:29])*np.array(self.state[29:]))- 10000 ))
            
            # print('total asset: {}'.format(self.state[0]+ sum(np.array(self.state[1:29])*np.array(self.state[29:]))))
            return self.state, self.reward, self.terminal,{}

        else:
            """
            # print(np.array(self.state[1:29]))

        # 1189 days to run this!
        self.day += 1
        for agent in range(self.agent_count):
            begin_total_asset = self.state[agent][0]+ sum(self.state[agent][1:29]*self.state[agent][29:])
            # print("begin_total_asset:{}".format(begin_total_asset))
            argsort_actions = np.argsort(actions[agent])
            sell_index = argsort_actions[:np.where(actions[agent] < 0)[0].shape[0]]
            buy_index = argsort_actions[::-1][:np.where(actions[agent] > 0)[0].shape[0]]

            for index in sell_index:
                # print('take sell action'.format(actions[index]))
                self._sell_stock(index, actions[agent][index], agent)

            for index in buy_index:
                # print('take buy action: {}'.format(actions[index]))
                self._buy_stock(index, actions[agent][index], agent)

            self.data = test_daily_data[self.day] 
            if self.day >=1189:        
                self.terminal = np.array([True for i in range(self.agent_count)])

            # print("stock_shares:{}".format(self.state[29:]))
            self.state[agent] =  np.array([self.state[agent][0]] + self.data.adjcp.values.tolist() + list(self.state[agent][29:]))
            end_total_asset = self.state[agent][0]+ sum(self.state[agent][1:29]*self.state[agent][29:])
            # print("end_total_asset:{}".format(end_total_asset))
            
            self.reward[agent] = end_total_asset - begin_total_asset            
            # print("step_reward:{}".format(self.reward))

            self.asset_memory[agent].append(end_total_asset)

        return self.state, self.reward, self.terminal, {}

    def reset(self):
        self.asset_memory = [[10000] for i in range(self.agent_count)]
        self.day = 0
        self.action_space = []
        for i in range(self.agent_count):
            # each entry ranges from -5 to +5 
            self.action_space.append([0 for j in range(28)])
        self.action_space = np.array(self.action_space)
        
        self.data = test_daily_data[self.day]
        agent_list = [10000] + self.data.adjcp.values.tolist() + [0 for j in range(28)]
        self.state = np.array([agent_list for i in range(self.agent_count)])   
        self.terminal = np.array([False for i in range(self.agent_count)])
        # iteration += 1 
        return self.state