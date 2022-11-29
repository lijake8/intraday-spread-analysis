# dataset downloaded from http://www.kibot.com/Support.aspx#aggregate_bidask_data_format
# ETF: IVE: S&P 500 Value ETF

import pandas as pd
import numpy as np

# # read in the data
# df = pd.read_csv('IVE_bidask1min.txt', sep=',', header=None)
# df.columns = ['Date', 'Time', 'Bid_open', 'Bid_high', 'Bid_low', 'Bid_close', 'Ask_open', 'Ask_high', 'Ask_low', 'Ask_close']

# # CLEAN DATA ############################################################
# # convert date and time columns to datetime format
# df['Time'] = pd.to_datetime(df['Time'], format='%H:%M')
# df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')

# # group the time into intervals bins (floor)
# df['Time_30'] = df['Time'].dt.floor('30min')
# df['Time_hr'] = df['Time'].dt.floor('60min')

# # group the date into years and months bins
# df['Year'] = pd.DatetimeIndex(df['Date']).year
# df['Month'] = pd.DatetimeIndex(df['Date']).month

# # remove rows with missing data and 0's for the prices
# print('before cleaning junk rows: {} rows'.format(df.shape[0]))
# df = df.dropna()
# df = df[df['Bid_high'] > 0]

# # remove rows with time before 9:30am and after 4:00pm
# df = df[df['Time'] >= pd.to_datetime('09:30', format='%H:%M')]
# df = df[df['Time'] < pd.to_datetime('16:00', format='%H:%M')] # < not <= because the interval bins would only have 1 type of entry for the bin starting at 4:00
# print('after cleaning rows before 9:30am, after 4:00pm: {} rows'.format(df.shape[0]))

# # CALCULATED FIELDS #####################################################
# # insert column for average bid and ask prices
# df.insert(6, 'Bid_avg', (df['Bid_high'] + df['Bid_low'] + df['Bid_open'] + df['Bid_close']) / 4)
# df.insert(10, 'Ask_avg', (df['Ask_high'] + df['Ask_low'] + df['Ask_open'] + df['Ask_close']) / 4)

# # insert column for average prices
# df['Avg_price'] = (df['Bid_avg'] + df['Ask_avg']) / 2

# # insert columns for spread
# df['Spread'] = df['Ask_avg'] - df['Bid_avg']
# df['Spread_as_Pct'] = df['Spread'] / df['Avg_price'] * 100 # in percent

# # create smaller dataframe with only the columns we need
# df_small = df.drop(['Date', 'Bid_open', 'Bid_high', 'Bid_low', 'Bid_close', 'Ask_open', 'Ask_high', 'Ask_low', 'Ask_close'], inplace=False, axis=1)

# # reorder columns
# cols_order = ['Year', 'Month', 'Time', 'Time_30', 'Time_hr', 'Bid_avg', 'Ask_avg', 'Avg_price', 'Spread', 'Spread_as_Pct']
# df_small = df_small[cols_order]

# # ANALYSIS ##############################################################
# # create a new dataframe with the average spread for each half hour
# df_spread = df_small.groupby(['Time_30'])['Spread_as_Pct'].mean().reset_index()
# print(df_spread.head(25))

# print('-----------------')

# # create a new dataframe with the average spread for each month
# df_spread2 = df_small.groupby(['Month'])['Spread_as_Pct'].mean().reset_index()
# print(df_spread2.head(25))












# read in the data
df = pd.read_csv('IVE_tickbidask.csv', sep=',', header=None)
df.columns = ['Date', 'Time', 'Price', 'Bid', 'Ask', 'Volume']

# CLEAN DATA ############################################################
# convert date and time columns to datetime format
df['Time'] = pd.to_datetime(df['Time'], format='%H:%M:%S')
df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')



# group the date into years and months bins
df['Year'] = pd.DatetimeIndex(df['Date']).year
df['Month'] = pd.DatetimeIndex(df['Date']).month

# remove rows with missing data and 0's for the prices
print('before cleaning junk rows: {} rows'.format(df.shape[0]))
df = df.dropna()
df = df[df['Bid'] > 0]

# remove rows with time before 9:30am and after 4:00pm
df = df[df['Time'] >= pd.to_datetime('09:30:00', format='%H:%M:%S')]
df = df[df['Time'] < pd.to_datetime('16:00:00', format='%H:%M:%S')] # < not <= because the interval bins would only have 1 type of entry for the bin starting at 4:00
print('after cleaning rows before 9:30am, after 4:00pm: {} rows'.format(df.shape[0]))

# CALCULATED FIELDS #####################################################
# insert column for average prices
df['Avg_price'] = (df['Bid'] + df['Ask']) / 2

# insert columns for spread
df['Spread'] = df['Ask'] - df['Bid']
df['Spread_as_Pct'] = df['Spread'] / df['Avg_price'] * 100 # in percent

# SMALLER DATAFRAME #####################################################
# create smaller dataframe with only the columns we need
df_small = df.drop(['Date', 'Price'], inplace=False, axis=1)

# group the time into intervals bins (floor)
df_small['Time_30'] = df_small['Time'].dt.floor('30min')
df_small['Time_hr'] = df_small['Time'].dt.floor('60min')


# reorder columns
cols_order = ['Year', 'Month', 'Time', 'Time_30', 'Time_hr', 'Bid', 'Ask', 'Avg_price', 'Spread', 'Spread_as_Pct', 'Volume']
df_small = df_small[cols_order]

print(df_small.head(20))
print(df_small.tail(20))

# ANALYSIS ##############################################################
# 30 MINUTE INTERVALS ##################################################
# corresponding total volumes over that time period
df_vol = df_small.groupby(['Time_30'])['Volume'].sum().reset_index()
# normalize the volume to a % of total daily volume
df_vol['Vol_as_pct_of_daily_vol'] = df_vol['Volume'] / df_vol['Volume'].sum() * 100 # in percent
print(df_vol.head(25))

# examine absolute spread
# create a new dataframe with the average spread for each half hour
df_spread = df_small.groupby(['Time_30'])['Spread'].mean().reset_index()
print(df_spread.head(25))

# examine relative spread
# create a new dataframe with the average spread for each half hour
df_spread = df_small.groupby(['Time_30'])['Spread_as_Pct'].mean().reset_index()
print(df_spread.head(25))




# CHECK IF ANY YOY CHANGE ###############################################
df_vol = df_small.groupby(['Time_30', 'Year'])['Volume'].sum().reset_index()
# normalize the volume to a % of total daily volume
df_vol['Vol_as_pct_of_daily_vol'] = df_vol['Volume'] / df_vol['Volume'].sum() * 100 # in percent
print(df_vol.head(25))