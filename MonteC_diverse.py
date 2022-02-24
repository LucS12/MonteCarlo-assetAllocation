'''
Monte Carlo Simulations: Diversification Analysis
Importance of Diversification for Asset Allocation
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas_datareader as web

#Read in SPX Members:
path = 'spx_names.csv'
spx_names = pd.read_csv(path)

#Adjust Ticker names to a list:
stocks = [stock[0] for stock in spx_names.Symbols.str.split()]

#Gather price data for names 
start = '2019-01-01'
end = '2022-02-01'
source = 'stooq'
data = web.DataReader(stocks, source, start, end).Close

#Setting Up Monte Carlo:
num_stocks = list(range(2,21))  #Number of stocks each iteration
sims = 10000                    #Simulations

#Lists to store Average returns/volatilities/Sharpe for each:
rets = []
vols = []
sr = []

#Loop for simulations 2 to 20 number of stocks:
for num in num_stocks:
    
    #To keep each simulations Returns/Volatilities:
    sim_rets = []
    sim_vols = []
    sim_sr = []
    
    #Singular simulation of 10000 portfolios for num of stocks:
    for _ in range(sims):
        
        #Get random num amount of stock locations:
        rand_stocks = np.random.permutation(len(stocks[:283]))[:num]
        temp_df = data.iloc[:, rand_stocks]  #select above stocks only
        
        #Calculate weights:
        ws = np.random.rand(num)   #Random num amount of weights
        ws = ws / ws.sum()         #Normalize to add to 1
        
        #Portfolio returns:
        returns = temp_df.resample('Y').last().pct_change().mean()  #Annual return
        return_df = np.dot(returns, ws)  #portfolio return
        sim_rets.append(return_df)       #add to list to later calculate average
        
        #Portfolio Volatilities:
        cov = temp_df.pct_change().cov()        #Covariance matrix
        var = np.dot(ws.T, np.dot(cov, ws))     #variance = weights.T*Cov*weights
        yearly_vol = np.sqrt(var)*np.sqrt(252)  #Std = sqrt(var), times sqrt(252) gives annual vol
        sim_vols.append(yearly_vol)             #Add it to list for later average calc.
        
        #Portfolio Sharpe:
        sharpe = (return_df - 0.01) / yearly_vol  #(Returns - risk-free rate) / volatility
        sim_sr.append(sharpe)                     #Add to list for later average calc.

    #Find average portfolio return/volatility/sharpe from 10,000 runs:
    #Returns:
    avg_ret = np.mean(sim_rets)  #Average
    rets.append(avg_ret)         #Add to original empty list
    
    #Volatilities:
    avg_vol = np.mean(sim_vols)  #Average
    vols.append(avg_vol)         #Add to original empty list
    
    #Sharpe Ratios:
    avg_sr = np.mean(sim_sr)   #Average
    sr.append(avg_sr)          #Add to original empty list  

#Graph Line Charts Showing Returns/Vols/Sharpes changing with number of stocks:
import seaborn as sea
sea.set_theme()

#Returns vs. Number of stocks:
plt.figure(figsize=(8,5))  
#Lineplot with Seaborn:
sea.lineplot(x=num_stocks, y=rets)
#Adding proper title and labels:
plt.title('Returns vs. Num. of Stocks', fontsize=18)
plt.xlabel('Number of Stocks', fontsize=15)
plt.ylabel('Return of Portfolio', fontsize=15)

#Volatilities vs. Number of stocks:
plt.figure(figsize=(8,5))  
#Lineplot with Seaborn:
sea.lineplot(x=num_stocks, y=vols)
#Adding proper title and labels:
plt.title('Risk vs. Num. of Stocks', fontsize=18)
plt.xlabel('Number of Stocks', fontsize=15)
plt.ylabel('Volatility of Portfolio', fontsize=15)

#Sharpe Ratio vs. Number of stocks:
plt.figure(figsize=(8,5))
#Lineplit with Seaborn:
sea.lineplot(x=num_stocks, y=sr)
#Adding proper title and labels:
plt.title('Sharpe Ratio vs. Num. of Stocks', fontsize=18)
plt.xlabel('Number of stocks', fontsize=15)
plt.ylabel('Sharpe Ratio of Portfolio', fontsize=15)