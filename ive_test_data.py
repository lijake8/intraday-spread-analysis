# dataset downloaded from http://www.kibot.com/Support.aspx#aggregate_bidask_data_format
# http://api.kibot.com/?action=history&symbol=IVE&interval=tickbidask&bp=1&user=guest
# ETF: IVE: S&P 500 Value ETF

import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# read in the data
df = pd.read_csv('IVE_tickbidask.csv', sep=',', header=None)
df.columns = ['Date', 'Time', 'Price', 'Bid', 'Ask', 'Volume']

# CLEAN DATA ############################################################
# convert date and time columns to datetime format
df['Time'] = pd.to_datetime(df['Time'], format='%H:%M:%S')
df['Time_pretty'] = df['Time'].dt.time
df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')

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



# CALCULATED FIELDS #####################################################
# insert column for average prices
df['Avg_price'] = (df['Bid'] + df['Ask']) / 2

# insert columns for spread
df['Spread'] = df['Ask'] - df['Bid']
df['Spread_as_Pct'] = df['Spread'] / df['Avg_price'] * 100 # in percent

# find the max spread in the dataset
max_spread = df['Spread_as_Pct'].max()
print(f'The max spread as % of price is {max_spread}, which means there is bad data')

# remove rows with bad data, e.g. spread % is more than 2% of the price
df = df[df['Spread_as_Pct'] < 2]
ending_rows = df.shape[0]
print('after cleaning rows based on market hours and spread outliers: {} rows'.format(ending_rows))
print('removed {} rows'.format(starting_rows - ending_rows))

# SMALLER DATAFRAME #####################################################
# create smaller dataframe with only the columns we need
df_small = df.drop(['Date', 'Price'], inplace=False, axis=1)

# reorder columns
cols_order = ['Year', 'Month', 'Time', 'Time_pretty', 'Bid', 'Ask', 'Avg_price', 'Spread', 'Spread_as_Pct', 'Volume']
df_small = df_small[cols_order]

print('df_small')
print(df_small.head())

# ANALYSIS ##############################################################
intervals = [1, 5, 20, 30, 60]
for interval in intervals:
    print('!!!!!interval: {} minutes'.format(interval))

    # group the time into intervals bins (floor: 9:31am -> 9:30am, for example)
    label = f'Time_{interval}'
    df_small[label] = df_small['Time'].dt.floor(f'{interval}min')
    
    # get corresponding total volumes for this interval schema
    df_vol = df_small.groupby([label])['Volume'].sum().reset_index()

    # normalize the volume to a % of total daily volume
    df_vol['Vol_as_pct_of_daily_vol'] = df_vol['Volume'] / df_vol['Volume'].sum() * 100 # in percent
    df_vol['Time_pretty'] = df_vol[label].dt.time

    # examine absolute and relative spread
    # create a new dataframe with the average spread for each interval bin
    df_spread = df_small.groupby([label])['Spread', 'Spread_as_Pct'].mean().reset_index()

    # merge the two dataframes to bring spread and volume together
    df_merged = pd.merge(df_spread, df_vol, on=label)
    df_merged.drop(['Volume', label], inplace=True, axis=1)
    
    # convert the time column to a string for graphing
    df_merged['Time_pretty'] = df_merged['Time_pretty'].astype(str)
    df_merged['Time_pretty'] = df_merged['Time_pretty'].str.slice(0, 5)

    print('df_merged')
    print(df_merged.head(20))

    # PLOT ################################################################
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=df_merged['Time_pretty'], y=df_merged['Spread_as_Pct'], name='Spread as % of Avg Price'), secondary_y=False)
    fig.add_trace(go.Scatter(x=df_merged['Time_pretty'], y=df_merged['Vol_as_pct_of_daily_vol'], name='Volume as % of Daily Volume'), secondary_y=True)
    fig.update_layout(title=f'Spread and Volume in {interval} Minute Intervals Throughout Trading Hours', xaxis_title='Time', yaxis_title='Spread (as % of Price)', yaxis2_title='Volume (as % of Daily Volume)')
    fig.write_image(f'spread_plot_{interval}.png')

    # make a plotly scatter plot with the spread and volume on the same graph
    fig2 = px.scatter(df_merged, x='Vol_as_pct_of_daily_vol', y='Spread_as_Pct', title=f'Spread and Volume in {interval} Minute Intervals Throughout Trading Hours', trendline='ols')
    fig2.update_layout(title=f'Spread and Volume Correspondence, {interval} Minute Intervals', xaxis_title='Volume (as % of Daily Volume)', yaxis_title='Spread (as % of Price)')
    fig2.write_image(f'scatter_plot_{interval}.png')

    # generates a grouped bar chart based on a macro factor (e.g. year, month, etc.)
    def macro_change(factor):
        # get corresponding total volumes for this interval schema
        df_vol_grouped_by_factor = df_small.groupby([factor, label])['Volume'].sum().reset_index()

        # normalize the volume to a % of total daily volume
        df_vol_grouped_by_factor['Vol_as_pct_of_daily_vol'] = df_vol_grouped_by_factor['Volume'] / df_vol_grouped_by_factor['Volume'].sum() * 100 # in percent
        df_vol_grouped_by_factor['Time_pretty'] = df_vol_grouped_by_factor[label].dt.time

        # examine absolute and relative spread
        # create a new dataframe with the average spread for each interval bin
        df_spread_grouped_by_factor = df_small.groupby([factor, label])['Spread', 'Spread_as_Pct'].mean().reset_index()

        # merge the two dataframes to bring spread and volume together based on interval and month
        df_merged_by_factor = pd.merge(df_spread_grouped_by_factor, df_vol_grouped_by_factor, on=[factor, label])
        df_merged_by_factor.drop(['Volume', label], inplace=True, axis=1)
        
        # convert the time column to a string for graphing
        df_merged_by_factor['Time_pretty'] = df_merged_by_factor['Time_pretty'].astype(str)
        df_merged_by_factor['Time_pretty'] = df_merged_by_factor['Time_pretty'].str.slice(0, 5)

        print('df_merged_by_factor')
        print(df_merged_by_factor.head(20))

        color_scale = ['#636EFA'] * 15
        fig3 = px.histogram(df_merged_by_factor, x=factor, y='Spread_as_Pct', color='Time_pretty', barmode='group', color_discrete_sequence=color_scale)

        fig3.update_layout(title=f'Spread in 30 Minute Intervals Throughout Trading Hours, Grouped by {factor}', xaxis_title=factor, yaxis_title='Spread (as % of Price)', xaxis={"dtick":1}, barmode='group', showlegend=False)
        fig3.write_image(f'spread_plot_{factor}ly.png')

    # CHECK IF ANY MONTHLY/YEARLY CHANGE IN SPREAD ########################################
    if interval == 20:
        macro_change('Month')
        macro_change('Year')

        
