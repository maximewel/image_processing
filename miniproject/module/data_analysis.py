import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

df1_name = "dataframe_1.pkl"
df2_name = "dataframe_2.pkl"

def analyse_time():
    df1 : pd.DataFrame = pd.read_pickle(os.path.join(os.path.dirname(__file__), "../out/", df1_name)) 

    df1_sorted_time = df1.sort_values(['total_time'])

    fig, ax = plt.subplots(1, 1)
    fig.suptitle("Difference per time analysis")

    df1_sorted_time['diff'].plot.area(ax=ax, use_index=False, x=df1_sorted_time['total_time'])
    ax.text(0.0, 0.8,  f"Min={df1['total_time'].min()}\nMax={df1['total_time'].max()}\nAvg={df1['total_time'].mean()}", transform=ax.transAxes)
    ax.set_xlabel('Bundles sorted over time')
    ax.set_ylabel('Difference')

    plt.show()
    plt.close(fig)

def analyse_repair():
    df1 : pd.DataFrame = pd.read_pickle(os.path.join(os.path.dirname(__file__), "../out/", df1_name))  

    print(df1)
    print(df1.columns)

    df1_with_repair = df1[df1['repair_lost_bact'] == True]
    df1_without_repair = df1[df1['repair_lost_bact'] == False]

    fig, axes = plt.subplots(1, 2)
    fig.suptitle("Repair mecanism analysis")

    mini, maxi = 0, max(df1_without_repair['diff'].max(), df1_with_repair['diff'].max())

    df1_with_repair = df1_with_repair.sort_values(['diff'])
    df1_without_repair = df1_without_repair.sort_values(['diff'])

    axes[0].set_title("Repair mecanism on")
    axes[0].set_ylim(mini, maxi)
    df1_with_repair['diff'].plot.area(ax=axes[0], use_index=False)
    axes[0].text(0.0, 0.8,  f"Min={df1_with_repair['diff'].min()}\nMax={df1_with_repair['diff'].max()}\nAvg={df1_with_repair['diff'].mean()}", transform=axes[0].transAxes)

    axes[1].set_title("Repair mecanism off")
    axes[1].set_ylim(mini, maxi)
    df1_without_repair['diff'].plot.area(ax=axes[1], use_index=False)
    axes[1].text(0.0, 0.8,  f"Min={df1_without_repair['diff'].min()}\nMax={df1_without_repair['diff'].max()}\nAvg={df1_without_repair['diff'].mean()}", transform=axes[1].transAxes)

    plt.show()
    plt.close(fig)

def analyse_threshold():
    df1 : pd.DataFrame = pd.read_pickle(os.path.join(os.path.dirname(__file__), "../out/", df1_name))  
    df2 : pd.DataFrame = pd.read_pickle(os.path.join(os.path.dirname(__file__), "../out/", df2_name))  

    for dataframe in df1, df2:

        df1_groups = dataframe.groupby('dist_threshold')

        fig, axes = plt.subplots(1, len(df1_groups))
        fig.suptitle("Threshold analysis")

        mini, maxi = 0, max([df['diff'].max() for (_, df) in df1_groups])

        for i, (_, df) in enumerate(df1_groups):
            df = df.sort_values(['diff'])
            th  = df.iloc[0]['dist_threshold']
            axes[i].set_title(f"Threshold ({th})")
            axes[i].set_ylim(mini, maxi)
            df['diff'].plot.area(ax=axes[i], use_index=False)
            axes[i].text(0.0, 0.8,  f"Min={df['diff'].min()}\nMax={df['diff'].max()}\nAvg={df['diff'].mean()}", transform=axes[i].transAxes)

        plt.show()
        plt.close(fig)


elipsis = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
rect = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

def filter_ellipsis(row):
    return np.array_equal(row['morph_kernel'], elipsis)

def filter_rect(row):
    return np.array_equal(row['morph_kernel'], rect)

def analysis_kernel():
    df1 : pd.DataFrame = pd.read_pickle(os.path.join(os.path.dirname(__file__), "../out/", df1_name)) 

    df_elipsis = df1[df1.apply(filter_ellipsis, axis=1)]
    df_rect = df1[df1.apply(filter_rect, axis=1)]

    fig, axes = plt.subplots(1, 2)
    fig.suptitle("Morph kernel analysis")

    mini, maxi = 0, max(df_elipsis['diff'].max(), df_rect['diff'].max())

    df_elipsis = df_elipsis.sort_values(['diff'])
    df_rect = df_rect.sort_values(['diff'])

    axes[0].set_title("Ellipsis kernel")
    axes[0].set_ylim(mini, maxi)
    df_elipsis['diff'].plot.area(ax=axes[0], use_index=False)
    axes[0].text(0.0, 0.8,  f"Min={df_elipsis['diff'].min()}\nMax={df_elipsis['diff'].max()}\nAvg={df_elipsis['diff'].mean()}", transform=axes[0].transAxes)

    axes[1].set_title("Rectangle kernel")
    axes[1].set_ylim(mini, maxi)
    df_rect['diff'].plot.area(ax=axes[1], use_index=False)
    axes[1].text(0.0, 0.8,  f"Min={df_rect['diff'].min()}\nMax={df_rect['diff'].max()}\nAvg={df_rect['diff'].mean()}", transform=axes[1].transAxes)

    plt.show()
    plt.close(fig)

def analyse_best():
    df2 : pd.DataFrame = pd.read_pickle(os.path.join(os.path.dirname(__file__), "../out/", df2_name)) 

    df_sorted = df2.sort_values(['diff'])

    print(df_sorted[['dist_transf_morph', 'bin_morph', 'dist_threshold', 'diff', 'total_time']])

if __name__ == "__main__":

    #analyse_repair()

    #analysis_kernel()

    #analyse_time()

    #analyse_best()

    analyse_threshold()