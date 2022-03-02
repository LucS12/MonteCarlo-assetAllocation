'''
Monte Carlo Simulation: Portfolio Optimization
    1. Minimum volatility
    2. Maximum Return
    3. Max Sharpe Ratio
'''

#Necessary Packages:
import pandas as pd
import pandas_datareader as web
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sea

#Stocks to be used:
stocks = ['AMZN','IBM','TSLA','AAL','UAL','DAL',
          'MRK','MRNA','PFE','BAC','ECL','WFC','CVX',
          'BK', 'CDW', 'CTSH','MCO','NOC','AAPL','ABT','AEE',
          'FANG','ALK','ADI','AIZ','AKAM','AMD','AOS',
          'BA','AVY','CBRE','DHI','DVA']

#Gather closing prices from Yahoo for more than 3 years:
start = '2019-01-01'
end = '2022-03-01'
source = 'yahoo'
data = web.DataReader(stocks, source, start, end)['Close']

#Annual returns and covariance matrix:
yearly_ret = data.resample('Y').last().pct_change().mean()
cov = data.pct_change().cov()

#Empty DataFrame to store generated portfolios of max return, min vol, max Sharpe:
sim_port = pd.DataFrame()

#Number of simulations/portfolios and number of weights:
num_weights = len(stocks)
ports = 5000
sims = 2500

num_weights   #Show number of weights that need to be made

#Loop to make 2500 simulations:
for sim in range(sims):
    #Empty lists to hold portfolio info:
    rets, volas, wgts, sharpe_ratio = [], [], [], []
    
    #Loop to make random 5000 portfolios:
    for port in range(ports):
        
        #Weights generation:
        ws = np.random.random(num_weights)   #Random positive floats for each stock
        ws = ws/ws.sum()                     #Standardize for weights to add to 1
        wgts.append(ws)                      #Add them to list
        
        #Returns Calculations:
        ret = np.dot(ws, yearly_ret)    #Weighted sum with dot product
        rets.append(ret)                #Add it to list
        
        #Volatility Calculation:
        var = np.dot(ws.T, np.dot(cov, ws))     #variance = weights.T*Cov*weights
        yearly_vol = np.sqrt(var)*np.sqrt(252)  #Std = sqrt(var), times sqrt(252) gives annual vol
        volas.append(yearly_vol)                #Add it to list
        
        #Sharpe Ratio Calculation: 
        sharpe = (ret - 0.01) / yearly_vol    #(Returns - risk-free) / volatility
        sharpe_ratio.append(sharpe)           #Add to list
    
    #Add list of portfolios made to a dictionary:
    ports_dict = {'Returns':rets, 'Risk':volas,'Sharpe_R':sharpe_ratio}
    
    #Placing weights with stock symbol into the dict above:
    for counter, sym in enumerate(data.columns.tolist()):
        ports_dict[sym+'_Weight'] = [weight[counter] for weight in wgts]
    
    #Make DataFrame out of dictionary:
    ports_df = pd.DataFrame(ports_dict)
    
    #Place Max return, min vol, and max Sharpe portfolios in new DataFrame:
        #Sharpe Ratio = (return - risk-free return) / volatility
    max_ret = ports_df.iloc[ports_df.Returns.idxmax()]     #Max return portfolio made
    min_vol = ports_df.iloc[ports_df.Risk.idxmin()]        #Min volatility portfolio made
    max_sharpe = ports_df.iloc[ports_df.Sharpe_R.idxmax()] #Max Sharpe Ratio portfolio made
    
    #Store simulation results in empty DataFrame previously made:
    sim_port = sim_port.append([max_ret,min_vol,max_sharpe], ignore_index=True)

#Plot histograms for Return, Volatility, and Sharpe Distributions:
sea.set_theme()

for col in sim_port.columns[:3]:
    plt.figure()
    sea.histplot(sim_port[col], bins=50)
    plt.title(col + ' Distribution', fontsize=16)
    plt.show()

#Plotting All Portfolios, extremes (ex: max return), and average portfolio:
sim_max_return = sim_port.iloc[sim_port.Returns.idxmax()]   #Max return found in simulations
sim_min_vol = sim_port.iloc[sim_port.Risk.idxmin()]         #Min risk found
sim_max_sharpe = sim_port.iloc[sim_port.Sharpe_R.idxmax()]  #Max Sharpe Ratio found
sim_mean = sim_port.mean()                                  #Average portfolio found

#Making figure and plotting portfolios on a scatter plot:
plt.figure(figsize=(10,7)) 
plt.scatter(sim_port.Risk, sim_port.Returns, alpha=0.3, s=3)       
plt.scatter(sim_max_return[1], sim_max_return[0], s=300, c='g', marker='*',label='Max Return')
plt.scatter(sim_min_vol[1], sim_min_vol[0], s=300, c='r', marker='*', label='Min Vol')       
plt.scatter(sim_max_sharpe[1], sim_max_sharpe[0], s=300, marker='*', c='y', label='Max Sharpe') 
plt.scatter(sim_mean[1], sim_mean[0], s=300, c='purple', marker='*', label='Simulation Avgs') 
             
#Setting appropriate labels:
plt.xlabel('Volatility', fontsize=15)     
plt.ylabel('Return', fontsize=15)
plt.title('Monte Carlo Simulation Portfolios', fontsize=18)  
plt.legend(loc='lower right', fontsize=15)

plt.show()
