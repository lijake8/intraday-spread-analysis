# dataset downloaded from http://www.kibot.com/Support.aspx#aggregate_bidask_data_format
# http://api.kibot.com/?action=history&symbol=IVE&interval=tickbidask&bp=1&user=guest
# ETF: IVE: S&P 500 Value ETF

import pandas as pd
import matplotlib.pyplot as plt

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
# intervals = [1, 5, 10, 15, 30, 60]
intervals = [30] # TODO: change this later, this is just for testing
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

    print(df_merged.head(20))



    # PLOT ################################################################
    # default pandas plotting
    # plt.figure()
    ax = df_merged.plot('Time_pretty', 'Spread_as_Pct', kind='line', figsize=(10, 5), legend=True, title=f'Spread as % of Avg Price for {interval} min intervals')
    df_merged.plot('Time_pretty', 'Vol_as_pct_of_daily_vol', kind='line', ax=ax, secondary_y=True, color='red', legend=True)
    ax.set_ylabel('Spread (as percent of price)')
    ax.right_ax.set_ylabel('volume (as percent of daily volume)')
    ax.set_xlabel('Time')
    plt.savefig(f'spread_plot_{interval}.png')


    # plotly plotting
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=df_merged['Time_pretty'], y=df_merged['Spread_as_Pct'], name='Spread as % of Avg Price'), secondary_y=False)
    fig.add_trace(go.Scatter(x=df_merged['Time_pretty'], y=df_merged['Vol_as_pct_of_daily_vol'], name='Volume as % of Daily Volume'), secondary_y=True)
    fig.update_layout(title=f'Spread and Volume in {interval} Minute Intervals Throughout Trading Hours', xaxis_title='Time', yaxis_title='Spread (as % of price)', yaxis2_title='volume (as % of daily volume)')
    fig.show()
    fig.write_image(f'spread_plot_{interval}.png')

    print('got here')

    # make a plotly scatter plot with the spread and volume on the same graph
    fig = px.scatter(df_merged, x='Vol_as_pct_of_daily_vol', y='Spread_as_Pct', title=f'Spread and Volume in {interval} Minute Intervals Throughout Trading Hours')
    fig.show()
    fig.write_image(f'spread_plot_{interval}.png')    




























    if interval == 30:
        continue
        # CHECK IF ANY MONTHLY CHANGE IN SPREAD ########################################
        df_monthly = df_small.groupby([label, 'Year'])['Volume'].sum().reset_index()
        # normalize the volume to a % of total daily volume
        df_monthly['Vol_as_pct_of_daily_vol'] = df_monthly['Volume'] / df_monthly['Volume'].sum() * 100 # in percent
        print(df_vol.head(25))


        # multiple bar plot
        # https://python-graph-gallery.com/11-grouped-barplot/
        df_merged.plot(kind='bar', x='Year', y=['Spread_as_Pct', 'Vol_as_pct_of_daily_vol'])
        plt.show()


















    # fig, (spread_pct, L1_loss, D_loss) = plt.subplots(nrows=1, ncols=3, figsize=(12,4), sharex=True)


    # x1 = list(df_merged['Spread_as_Pct'])

    # plt.hist([x1], density=True)

    # # spread_pct.set_title('spread %')
    # # spread_pct.hist(dist1, bins=n_bins, density=True)
    # # spread_pct.set_xlabel('Iteration')
    # # spread_pct.set_ylabel('Loss')

    # # L1_loss.set_title('L1 + cGAN Generator L1 Loss')
    # # L1_loss.plot(hist_G_100_L1_losses)
    # # L1_loss.set_xlabel('Iteration')
    # # L1_loss.set_ylabel('Loss')

    # fig.tight_layout(pad=2)
    # plt.show()



