# dataset downloaded from http://www.kibot.com/Support.aspx#aggregate_bidask_data_format
# http://api.kibot.com/?action=history&symbol=IVE&interval=tickbidask&bp=1&user=guest
# ETF: IVE: S&P 500 Value ETF

import pandas as pd
import numpy as np

# read in the data
df = pd.read_csv('IVE_tickbidask.csv', sep=',', header=None)
df.columns = ['Date', 'Time', 'Price', 'Bid', 'Ask', 'Volume']

# CLEAN DATA ############################################################
# convert date and time columns to datetime format
df['Time'] = pd.to_datetime(df['Time'], format='%H:%M:%S')
df['Time_pretty'] = df['Time'].dt.time
df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')

print(df.head())

# group the date into years and months bins
df['Year'] = pd.DatetimeIndex(df['Date']).year
df['Month'] = pd.DatetimeIndex(df['Date']).month

# remove rows with missing data and 0's for the prices
starting_rows = df.shape[0]
print('before cleaning junk rows: {} rows'.format(starting_rows))
df = df.dropna()
df = df[df['Bid'] > 0]

# remove rows with time before 9:30am and after 4:00pm
df = df[df['Time'] >= pd.to_datetime('09:30:00', format='%H:%M:%S')]
df = df[df['Time'] < pd.to_datetime('16:00:00', format='%H:%M:%S')] # < not <= because the interval bins would only have 1 type of entry for the bin starting at 4:00
ending_rows = df.shape[0]
print('after cleaning rows before 9:30am, after 4:00pm: {} rows'.format(ending_rows))
print('removed {} rows'.format(starting_rows - ending_rows))

# CALCULATED FIELDS #####################################################
# insert column for average prices
df['Avg_price'] = (df['Bid'] + df['Ask']) / 2

# insert columns for spread
df['Spread'] = df['Ask'] - df['Bid']
df['Spread_as_Pct'] = df['Spread'] / df['Avg_price'] * 100 # in percent

# SMALLER DATAFRAME #####################################################
# create smaller dataframe with only the columns we need
df_small = df.drop(['Date', 'Price'], inplace=False, axis=1)

# reorder columns
cols_order = ['Year', 'Month', 'Time', 'Time_pretty', 'Bid', 'Ask', 'Avg_price', 'Spread', 'Spread_as_Pct', 'Volume']
df_small = df_small[cols_order]

print(df_small.head())

# ANALYSIS ##############################################################
intervals = [1, 5, 10, 15, 30, 60]
for interval in intervals:
    print('!!!!!interval: {} minutes'.format(interval))

    # group the time into intervals bins (floor)
    label = f'Time_{interval}'
    df_small[label] = df_small['Time'].dt.floor(f'{interval}min')
    
    # corresponding total volumes over that time period
    df_vol = df_small.groupby([label])['Volume'].sum().reset_index()
    # normalize the volume to a % of total daily volume
    df_vol['Vol_as_pct_of_daily_vol'] = df_vol['Volume'] / df_vol['Volume'].sum() * 100 # in percent
    df_vol.drop(['Volume'], inplace=True, axis=1)
    print(df_vol.head())

    # examine absolute and relative spread
    # create a new dataframe with the average spread for each half hour
    df_spread = df_small.groupby([label])['Spread', 'Spread_as_Pct'].mean().reset_index()
    print(df_spread.head())

    # merge the two dataframes
    df_merged = pd.merge(df_spread, df_vol, on=label)
    print(df_merged.head())




# # CHECK IF ANY YOY CHANGE ###############################################
# df_vol = df_small.groupby(['Time_30', 'Year'])['Volume'].sum().reset_index()
# # normalize the volume to a % of total daily volume
# df_vol['Vol_as_pct_of_daily_vol'] = df_vol['Volume'] / df_vol['Volume'].sum() * 100 # in percent
# print(df_vol.head(25))