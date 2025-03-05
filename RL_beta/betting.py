import gym
from gym import spaces
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

INITIAL_BALANCE = 100

class BettingEnv(gym.Env):
    # metadata = {'render.modes': ['human']}

    def __init__(self, df, results, INITIAL_BALANCE=100):
        self.df = df
        self.results = results
        self.initial_balance = INITIAL_BALANCE
        self.balance = INITIAL_BALANCE
        self.profit = 0
        
        self.starting_point = np.random.randint(len(self.df) - len(self.df) * 0.1) # Start anywhere but in the end 10%
        self.timestep = 0
        self.games_won = 0
        self.game_bets = []
        self.game_number = self.starting_point + self.timestep

        self.action_space = spaces.MultiDiscrete([3,10])
        
        self.observation_space = spaces.Box(
            low = self.df.min().min(), # Lowest value found in df
            high = self.df.max().max(), # Search the df for the max value (this may change with different data)
            shape = (df.shape[1],), # shape of one row of the df
            # dtype = np.float16
        )
        print('First ob: ',self.df.loc[self.game_number])
        
    def _next_obs(self):
        
        print('Get next obs')
        # Get next game row
        obs = self.df.loc[self.timestep]
        print('next obs success')
        return obs

    def _print_bet_csv(self):
        
        # Create bet_info_df
        bet_info_df = pd.DataFrame(self.game_bets)
        results_df = self.results.reset_index()

        
        # #Merge dfs
        self.merged_df = pd.merge(bet_info_df, results_df, on=['index', 'Home Odds', 'Vis Odds', 'Home Win'])
        self.merged_df.set_index('index', inplace=True)
        
        # #Print df
        self.merged_df.to_csv('./temp/MLB Bot Betting DF.csv', index=True)


    def _print_bet_chart(self):
        
        x_axis = list(range(self.timestep))
        plt.plot(x_axis, self.merged_df['Bankroll'])
        plt.title('Bankroll')
        plt.ylabel('Dollars')
        plt.xlabel('Games')
        plt.savefig('./temp/NBA_Bot_Betting.png')


    def _take_action(self, action):
        
        print('Start action')
        # Init
        action_type = action[0]
        amount = action[1] + 1

        self.game_number = self.starting_point + self.timestep

        game_result = self.results['Home Win'][self.game_number]
        odds = 0
        bet_on = 'NA'

        # VISITOR BET
        if action_type == 0:
            bet_on = 'False'

            # Find vis odds
            odds = self.results['Vis Odds'][self.game_number]
            if odds == 0:
                amount = 0

            # Place bet
            self.balance -= amount

            # Check if win
            if game_result == False:
                self.balance += round(amount * odds, 2)
                self.games_won += 1

        elif action_type == 1:
            bet_on = 'No bet'


        elif action_type == 2:
            bet_on = 'True'

            # Find home odds
            odds = self.results['Home Odds'][self.game_number]
            if odds == 0:
                amount = 0

            # Place bet
            self.balance -= amount

            # Check win
            if game_result == True:
                self.balance += round(amount * odds, 2)
                self.games_won += 1

        self.balance = round(self.balance, 2)

        bet_info = {
            'index': self.game_number,
            'Home Odds': self.results['Home Odds'][self.game_number],
            'Vis Odds': self.results['Vis Odds'][self.game_number],
            'Bet on': bet_on,
            'Home Win': game_result,
            'Amount': amount,
            'Odds': odds,
            'Bankroll': self.balance
        }

        self.game_bets.append(bet_info)
        print('Finish action')
        return bet_info

    def step(self, action):
        
        print('Start step')
        info = self._take_action(action)
        self.timestep += 1
        
        # Reward
        gamma = (self.timestep / len(self.df)) # time discount
        self.profit = self.balance - self.initial_balance
        reward = self.profit * gamma
        
        # Done
        done = self.balance <= 0
        
        # Obs
        obs = self._next_obs()
        
        # If last game, print results and start from beginning
        #test the printing of csv
        if self.timestep == 2500:
            self._print_bet_csv()
            self._print_bet_chart()
            self.game_bets = []
            print('Starting point: ',self.starting_point)
            print('Chart printed')
            
        print('Finished Step')
        return obs, reward, done, info

    def reset(self):
        
        self.initial_balance = INITIAL_BALANCE
        self.balance = INITIAL_BALANCE
        self.profit = 0
        
        self.starting_point = np.random.randint(len(self.df) - len(self.df) * 0.1) # Start anywhere but in the end 10%
        self.timestep = 0
        self.games_won = 0
        self.game_bets = []
        

    def render(self, mode='human', close=False):
        
        print('Timestep: ', self.timestep)
        print('Profit: ', self.profit)
        print('Games Won: ', self.games_won)
        print('Balance: ', self.balance)