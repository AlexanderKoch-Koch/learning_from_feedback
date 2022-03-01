import pandas
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path
import numpy as np
from os import listdir
from os.path import isdir

path = Path(__file__)
dir = path.parent / 'logs'/ 'open_loop_dreamer' / 'TransformerDreamer_2022-03-01_12-51-06'
run_dirs = [f for f in listdir(dir) if isdir(dir / f)]

dfs = [pandas.read_csv(dir / run / 'progress.csv') for run in run_dirs]
x = dfs[0]['episodes_total'].values

def smooth_values(x):
    x = np.concatenate(([x[0]], [x[0]], x, [x[-1]], [x[-1]]))
    x = x[:-4] + x[1:-3] + x[2:-2] + x[3:-1] + x[4:]
    x = x/5
    return x

with PdfPages(path.parent / 'open_loop_dreamer_pendulum.pdf') as pdf:
    for df in dfs:
        plt.plot(x, smooth_values(df['evaluation/episode_reward_mean'].values))

    plt.xlabel('Environment Steps', fontsize=14)
    plt.ylabel('Episode Reward', fontsize=14)
    plt.legend()
    plt.grid(True)
    pdf.savefig()