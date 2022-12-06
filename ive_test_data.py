# dataset downloaded from http://www.kibot.com/Support.aspx#aggregate_bidask_data_format
# http://api.kibot.com/?action=history&symbol=IVE&interval=tickbidask&bp=1&user=guest
# ETF: IVE: S&P 500 Value ETF

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.colors as colors

# read in the data
df = pd.read_csv('IVE_tickbidask.csv', sep=',', header=None)
df.columns = ['Date', 'Time', 'Price', 'Bid', 'Ask', 'Volume']

print('OG data')
print(df.head())

print('num trading days:', len(pd.unique(df['Date'])))

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
df = df[df['Bid'] > 0]

# remove rows with time before 9:30am and after 4:00pm
df = df[df['Time'] >= pd.to_datetime('09:30:00', format='%H:%M:%S')]
df = df[df['Time'] < pd.to_datetime('16:00:00', format='%H:%M:%S')] # < not <= because the interval bins would only have 1 type of entry for the bin starting at 4:00

# CALCULATED FIELDS #####################################################
# insert column for average prices
# df['Avg_price'] = (df['Bid'] + df['Ask']) / 2

# insert columns for spread
df['Spread'] = df['Ask'] - df['Bid']
df['Spread_as_Pct'] = df['Spread'] / df['Price'] * 100 # in percent

# insert column for change in price
# df['Price_change'] = df['Price'].diff()
df = df.dropna()

# find the max spread in the dataset
max_spread = df['Spread_as_Pct'].max()
print(f'The max spread as % of price is {max_spread}, which means there is bad data')

# remove rows with bad data, e.g. spread % is more than 2% of the price
df = df[df['Spread_as_Pct'] < 2]
ending_rows = df.shape[0]
print('after cleaning rows based on market hours and spread outliers: {} rows'.format(ending_rows))
print('removed {} rows'.format(starting_rows - ending_rows))
print('now there is trading days of:', len(pd.unique(df['Date'])))

# SMALLER DATAFRAME #####################################################
# create smaller dataframe with only the columns we need
df_small = df.drop(['Date'], inplace=False, axis=1)

# reorder columns
cols_order = ['Year', 'Month', 'Time', 'Time_pretty', 'Bid', 'Ask', 'Price', 'Spread', 'Spread_as_Pct', 'Volume']
df_small = df_small[cols_order]

print('df_small')
print(df_small.head())

# function that returns a dataframe of volatilities for a given year/month
def get_volatility_df(factor):
    """
    factor: 'Year' or 'Month'
    """
    periods = 252 if factor == 'Year' else 21 # 252 trading days in a year, 21 trading days in a month

    print('df')
    print(df.head())
    
    # get daily prices (open)
    df_daily_prices = df.groupby(['Date'])['Price', 'Year', 'Month'].first().reset_index()
    df_daily_prices.rename(columns={'Price': 'Open_price'}, inplace=True)

    # create column for log returns
    # df_daily_prices['Log_return'] = df_daily_prices['Open_price']

    # create a column for log of quotient of open prices
    df_daily_prices['Log_return'] = np.log(df_daily_prices['Open_price'] / df_daily_prices['Open_price'].shift(1))
    df_daily_prices = df_daily_prices.dropna()

    print('df_daily_prices')
    print(df_daily_prices.head())

    df_volatility = df_daily_prices.groupby([factor])['Log_return'].std().reset_index()
    df_volatility.rename(columns={'Log_return': 'Std_dev_of_returns'}, inplace=True)
    df_volatility['Volatility'] = df_volatility['Std_dev_of_returns'] * np.sqrt(periods) # annualize the volatility
    
    df_volatility.drop(['Std_dev_of_returns'], inplace=True, axis=1)

    print('df_volatility')
    print(df_volatility.head())

    return df_volatility
    


# ANALYSIS ##############################################################
intervals = [1, 5, 10, 15, 20, 30, 60]
# intervals = [20]
for interval in intervals:
    print('\n!!!!!interval: {} minutes!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n'.format(interval))

    # group the time into intervals bins (floor: 9:31am -> 9:30am, for example)
    label = f'Time_{interval}'
    df_small[label] = df_small['Time'].dt.floor(f'{interval}min')
    
    # get corresponding total volumes by interval bin
    df_vol = df_small.groupby([label])['Volume'].sum().reset_index()

    # normalize the volume to a % of total daily volume
    df_vol['Vol_as_pct_of_daily_vol'] = df_vol['Volume'] / df_vol['Volume'].sum() * 100 # in percent
    df_vol['Time_pretty'] = df_vol[label].dt.time



    # create a new dataframe with the average spread and % spread for each interval bin
    df_spread = df_small.groupby([label])['Spread', 'Spread_as_Pct'].mean().reset_index()

    # merge the 3 dataframes to bring spread, open prices, and volume together
    df_merged = pd.merge(df_spread, df_vol, on=label)
    # df_merged = pd.merge(df_merged, df_open_price, on=label)
    df_merged.drop(['Volume', label], inplace=True, axis=1)
    
    # convert the time column to a string for graphing
    df_merged['Time_pretty'] = df_merged['Time_pretty'].astype(str)
    df_merged['Time_pretty'] = df_merged['Time_pretty'].str.slice(0, 5)

    print('\ndf_merged\n')
    print(df_merged.head())

    # PLOT ################################################################
    # line graph of spread and volume on 2 y axes against time
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=df_merged['Time_pretty'], y=df_merged['Spread_as_Pct'], name='Spread as % of Avg Price', marker_color='#636EFA'), secondary_y=False)
    fig.add_trace(go.Scatter(x=df_merged['Time_pretty'], y=df_merged['Vol_as_pct_of_daily_vol'], name='Volume as % of Daily Volume', marker_color='#EE7674'), secondary_y=True)
    fig.update_layout(title=f'Spread and Volume in {interval} Minute Intervals Throughout Trading Hours', xaxis_title='Time', yaxis_title='Spread (as % of Price)', yaxis2_title='Volume (as % of Daily Volume)')
    fig.write_image(f'spread_plot_{interval}.png')

    # viridis color scale
    viridis_colorscale = ['#440154', '#481567', '#472878', '#424086', '#3B528B', '#33638D', '#2C728E', '#26828E', '#21918C', '#1FA088', '#28AE80', '#3FBC73', '#5EC962', '#84D44B', '#ADDC30', '#D8E219', '#FDE725', '#FEEB6B', '#FEEEA8', '#FFF0D1', '#FFF3E6', '#FFF7F2', '#FFFFFF']



    # scatter plot with the spread and volume on x and y axes
    fig2 = px.scatter(df_merged, x='Vol_as_pct_of_daily_vol', y='Spread_as_Pct', color='Time_pretty', title=f'Spread and Volume in {interval} Minute Intervals Throughout Trading Hours', color_discrete_sequence=viridis_colorscale, trendline='ols')
    fig2.update_traces(marker_size=10)
    fig2.update_layout(title=f'Spread and Volume Correspondence, {interval} Minute Intervals', xaxis_title='Volume (as % of Daily Volume)', yaxis_title='Spread (as % of Price)', legend_title_text='Time')

    #TODO: FIX , dont want to show every single bin in legend, only first and last? or every hour?
    print(fig2['data'][0]['legendgroup'])
    print(fig2['data'][0]['showlegend'])
    for i, entry in enumerate(fig2['data']): 
        if fig2['data'][i]['legendgroup'][-2:] == '00':
            fig2['data'][i]['showlegend'] = True
        else:
            fig2['data'][i]['showlegend'] = False
    print(fig2['data'][0]['legendgroup'])
    print(fig2['data'][0]['showlegend'])


    fig2.write_image(f'scatter_plot_{interval}.png')

    # generates a grouped bar chart based on a macro factor (e.g. year, month, etc.)
    def macro_change(factor, interval):
        """
        factor: 'Year' or 'Month'
        interval: some number of minutes to group by, e.g. 1, 5, 10, 15, 20, 30, 60
        """
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
        print(df_merged_by_factor.head())

        default_colorscale = ['#636EFA'] * 15
        fig4 = make_subplots(specs=[[{"secondary_y": True}]])

        #this works but can't figure out 2nd y axis
        fig3 = px.histogram(df_merged_by_factor, x=factor, y='Spread_as_Pct', color='Time_pretty', barmode='group', color_discrete_sequence=default_colorscale)
        # fig3.update_layout(title=f'Average Spread in {interval} Minute Intervals Throughout Trading Hours by {factor}', xaxis_title=factor, yaxis_title='Spread (as % of Price)', xaxis={"dtick":1}, barmode='group', showlegend=False)
        
        # fig3.add_trace(go.Histogram(x=df_merged_by_factor[factor], y=df_merged_by_factor['Spread_as_Pct']), secondary_y=True)


        # add traces from plotly express histogram to subplot figure
        for trace in fig3.select_traces():
            fig4.add_trace(trace, secondary_y = False)

        # handle data for and add secondary axis
        df_volatility = get_volatility_df(factor)
        fig4.add_trace(go.Scatter(x=df_volatility[factor], y=(df_volatility['Volatility'] * 100), mode='lines', marker_color='#EE7674', showlegend=True), secondary_y=True) #TODO

        fig4.update_layout(title=f'Average Spread in {interval} Minute Intervals Throughout Trading Hours by {factor}', xaxis_title=factor, yaxis_title='Spread (as % of Price)', yaxis2_title='Volatility (%)', xaxis={"dtick":1}, barmode='group')
        
        # show legend only for the second trace
        for i, _ in enumerate(fig4['data']): 
            fig4['data'][i]['showlegend'] = False
        fig4['data'][-1]['showlegend'] = True
        fig4['data'][-1]['name'] = 'Volatility (%)'
        fig4['data'][-2]['showlegend'] = True
        fig4['data'][-2]['name'] = 'Spread (% of Price)'

        fig4.write_image(f'spread_plot_{factor}ly.png')

    # CHECK IF ANY MONTHLY/YEARLY CHANGE IN SPREAD ########################################
    if interval == 20:
        macro_change('Month', 20)
        macro_change('Year', 20)

        

