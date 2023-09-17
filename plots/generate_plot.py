import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def bar_plot(x, y, title, xlabel, ylabel, color, save_path, hlines=None):
    """
    Generate bar plot
    :param df: dataframe
    :param x: x-axis
    :param y: y-axis
    :param title: title of the plot
    :param xlabel: x-axis label
    :param ylabel: y-axis label
    :param color: color of the plot
    :param save_path: path to save the plot
    :return: None
    """
    plt.figure(figsize=(12, 8))
    plt.bar(x, y, color=color)
    plt.title(title, fontsize=20)
    plt.xlabel(xlabel, fontsize=16)
    plt.ylabel(ylabel, fontsize=16)
    if hlines:
        # plot horizontal lines being careful of the x limits
        # change color and linestyle for each horizontal line
        
        # get possible colors and linestyles
        colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown']
        # remove plot color from colors
        colors.remove(color)

        linestyles = ['solid', 'dashed', 'dashdot', 'dotted']
        for i, (name, loss) in enumerate(hlines):
            plt.axhline(y=loss, color=colors[i], linestyle=linestyles[i], label=name)
    plt.legend(loc='upper right', fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.savefig(save_path, bbox_inches='tight')
    plt.show()

def plotLineLossPenaltyComparison():
    base_loss = 3.3101556301116943

    fn = 'owt/merge-adamw-sgd'
    df = pd.read_csv(f'plots/plot_data/{fn}.csv')

    # extract merge layer from 'gpt2-owt-l_11-adamw-adam' in 'Name' column
    df['merge_layer'] = df['Name'].str.extract(r'-l_(\d+)-')
    
    # calculate percentage of layers merged (1 - merge_layer / 12)
    df['merge_layer_perc'] = (1 - df['merge_layer'].astype(float) / 12) * 100

    # sort dataframe based on values in 'merge_layer_perc' column
    df.sort_values('merge_layer_perc', inplace=True)

    # calculate loss penalty (base_loss - val/loss)
    df['loss_penalty_perc'] = - (1 - df['val/loss'] / base_loss) * 100

    # plot loss penalty vs percentage of layers merged as line plot
    x = df['merge_layer_perc'].values
    # remove duplicates and nan from x
    x = np.unique(x[~np.isnan(x)])

    # get rows where 'Name' column contains 'head7'
    y1 = df['loss_penalty_perc'][df['Name'].str.contains('adamw_h')].values
    y1_label = 'Preserve Head'
    # get rows where 'Name' column ends with adamw9
    y2 = df['loss_penalty_perc'][df['Name'].str.endswith('sgd')].values
    y2_label = 'Substitute Head'
    
    title = 'GPT2 Small - merge AdamW - SGD'
    xlabel = 'Layers %'
    ylabel = 'Loss Penalty %'
    c1 = 'tab:blue'
    c2 = 'tab:orange'
    save_path = 'plots/owt/adamw-sgd-merge.png'
    plt.figure(figsize=(12, 8))
    plt.plot(x, y1, color=c1, label=y1_label)
    plt.plot(x, y2, color=c2, label=y2_label)
    # add legend
    plt.legend(loc='lower right', fontsize=14)
    plt.title(title, fontsize=20)
    plt.xlabel(xlabel, fontsize=16)
    plt.ylabel(ylabel, fontsize=16)
    # make xticks multiples of 0.1
    plt.xticks(np.arange(10, 100, 10), fontsize=14)
    #make yticks go from min of loss_penalty_perc to max of loss_penalty_perc
    # we want 10 yticks so calculate the range of yticks as (max - min) / 10
    # and add 1 to the max to make sure we include the max value
    # then round the min and max to the nearest integer
    step = (df['loss_penalty_perc'].max() - df['loss_penalty_perc'].min()) / 10.0
    yticks = np.arange(df['loss_penalty_perc'].min(), df['loss_penalty_perc'].max(), step)
    # add max value to yticks rounding to nearest integer
    yticks = np.append(yticks, df['loss_penalty_perc'].max())
    
    plt.yticks(yticks, fontsize=14)

    plt.savefig(save_path, bbox_inches='tight')
    plt.show()

def plotLineLossPenalty():
    base_loss = 7.218931198120117
    df = pd.read_csv('plots/plot_data/owt/sgd-seed.csv')

    # extract merge layer from 'gpt2-owt-l_11-adamw-adam' in 'Name' column
    df['merge_layer'] = df['Name'].str.extract(r'-l_(\d+)-')
    
    # calculate percentage of layers merged (1 - merge_layer / 12)
    df['merge_layer_perc'] = (1 - df['merge_layer'].astype(float) / 12) * 100

    # sort dataframe based on values in 'merge_layer_perc' column
    df.sort_values('merge_layer_perc', inplace=True)

    # calculate loss penalty (base_loss - val/loss)
    df['loss_penalty_perc'] = - (1 - df['val/loss'] / base_loss) * 100

    # plot loss penalty vs percentage of layers merged as line plot
    x = df['merge_layer_perc'].values
    y = df['loss_penalty_perc'].values
    title = 'GPT-Small merge SGD seed 1337-1339'
    xlabel = 'Layers %'
    ylabel = 'Loss Penalty %'
    color = 'tab:blue'
    save_path = 'plots/owt/sgd-seed-merge.png'
    plt.figure(figsize=(12, 8))
    plt.plot(x, y, color=color)
    plt.title(title, fontsize=20)
    plt.xlabel(xlabel, fontsize=16)
    plt.ylabel(ylabel, fontsize=16)
    # make xticks multiples of 0.1
    plt.xticks(np.arange(10, 100, 10), fontsize=14)
    #make yticks go from min of loss_penalty_perc to max of loss_penalty_perc
    # we want 10 yticks so calculate the range of yticks as (max - min) / 10
    # and add 1 to the max to make sure we include the max value
    # then round the min and max to the nearest integer
    step = (df['loss_penalty_perc'].max() - df['loss_penalty_perc'].min()) / 10.0
    yticks = np.arange(df['loss_penalty_perc'].min(), df['loss_penalty_perc'].max(), step)
    # add max value to yticks rounding to nearest integer
    yticks = np.append(yticks, df['loss_penalty_perc'].max())
    
    plt.yticks(yticks, fontsize=14)

    plt.savefig(save_path, bbox_inches='tight')
    plt.show()


def plotPreserveHeadComp():
    df = pd.read_csv('plots/plot_data/shakespeare/optims.csv')
    # sort dataframe based on values in 'val/loss' column
    # use gpt2-lg-owt-adamw-1339 and gpt2-lg-owt-adamw-1337 as baselines by visualizing their loss as horizontal lines
    df.sort_values('val/loss', inplace=True)

    # extract values of gpt2-lg-owt-adamw-1339 and gpt2-lg-owt-adamw-1337 for horizontal lines as (name, loss)
    # and remove them from the dataframe
    #hlines = [('Original loss seed 1339', df[df['Name'] == 'Seed 1339']['val/loss'].values[0]),]
    #hlines.append(('Original loss seed 1337', df[df['Name'] == 'Seed 1337']['val/loss'].values[0]))    
    hlines = None

    x = df['Name'].values
    y = df['val/loss'].values
    title = 'GPT2-Char Optimizer Comparison'
    xlabel = 'Optimizer'
    ylabel = 'Loss'
    color = 'tab:blue'
    save_path = 'plots/shakespeare/optim-comp.png'
    plt.figure(figsize=(12, 8))
    plt.bar(x, y, color=color)
    plt.title(title, fontsize=20)
    plt.xlabel(xlabel, fontsize=16)
    plt.ylabel(ylabel, fontsize=16)
    if hlines:
        # plot horizontal lines being careful of the x limits
        # change color and linestyle for each horizontal line
        
        # get possible colors and linestyles
        colors = ['tab:red', 'tab:brown', 'tab:blue', 'tab:orange', 'tab:purple', 'tab:green']
        # remove plot color from colors
        colors.remove(color)

        linestyles = ['solid', 'dashed', 'dashdot', 'dotted']
        for i, (name, loss) in enumerate(hlines):
            plt.axhline(y=loss, color=colors[i], linestyle=linestyles[1], label=name)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.savefig(save_path, bbox_inches='tight')
    plt.show()

plotLineLossPenaltyComparison()