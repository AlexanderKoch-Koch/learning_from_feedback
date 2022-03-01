import pandas
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path
import numpy as np
from os import listdir
from os.path import isfile, join, isdir

def plot_with_std(x, y, label, color):
    std = np.std(y, axis=0)
    mean = np.mean(y, axis=0)
    plt.plot(x, mean, label=label, color=color)
    plt.fill_between(x, mean - 0.5 * std, mean + 0.5 * std, color=color, alpha=0.2)

path = Path(__file__)
with_feedback_dir = path.parent / 'logs' / 'sequence_guessing' / 'sequence_guessing_with_feedback' / 'Dreamer_2022-02-21_07-52-25'
without_feedback_dir = path.parent / 'logs' / 'sequence_guessing' / 'sequence_guessing_without_feedback' / 'Dreamer_2022-02-22_07-28-42'
with_feedback_runs = [f for f in listdir(with_feedback_dir) if isdir(with_feedback_dir / f)]
without_feedback_runs = [f for f in listdir(without_feedback_dir) if isdir(without_feedback_dir / f)]

with_feedback_dfs = [pandas.read_csv(with_feedback_dir / run / 'progress.csv') for run in with_feedback_runs]
without_feedback_dfs = [pandas.read_csv(without_feedback_dir / run / 'progress.csv') for run in without_feedback_runs]
x = with_feedback_dfs[0]['episodes_total'].values

first_quarter_mean_with_feedback = np.stack([df['evaluation/custom_metrics/correct_sequence_1_quarter_mean'] for df in with_feedback_dfs])
first_quarter_mean_without_feedback = np.stack([df['evaluation/custom_metrics/correct_sequence_1_quarter_mean'] for df in without_feedback_dfs])

second_quarter_mean_with_feedback = np.stack([df['evaluation/custom_metrics/correct_sequence_2_quarter_mean'] for df in with_feedback_dfs])
second_quarter_mean_without_feedback = np.stack([df['evaluation/custom_metrics/correct_sequence_2_quarter_mean'] for df in without_feedback_dfs])

third_quarter_mean_with_feedback = np.stack([df['evaluation/custom_metrics/correct_sequence_3_quarter_mean'] for df in with_feedback_dfs])
third_quarter_mean_without_feedback = np.stack([df['evaluation/custom_metrics/correct_sequence_3_quarter_mean'] for df in without_feedback_dfs])

forth_quarter_mean_with_feedback = np.stack([df['evaluation/custom_metrics/correct_sequence_4_quarter_mean'] for df in with_feedback_dfs])
forth_quarter_mean_without_feedback = np.stack([df['evaluation/custom_metrics/correct_sequence_4_quarter_mean'] for df in without_feedback_dfs])


with PdfPages(r'./first_quarter_sequences.pdf') as export_pdf:
    plot_with_std(x, first_quarter_mean_with_feedback, 'with feedback', 'red')
    plot_with_std(x, first_quarter_mean_without_feedback, 'without feedback', 'black')
    plt.xlabel('Episodes', fontsize=14)
    plt.ylabel('Episode Reward', fontsize=14)
    plt.legend()
    plt.grid(True)
    # plt.show()
    export_pdf.savefig()
    plt.close()

with PdfPages(r'./second_quarter_sequences.pdf') as export_pdf:
    plot_with_std(x, second_quarter_mean_with_feedback, 'with feedback', 'red')
    plot_with_std(x, second_quarter_mean_without_feedback, 'without feedback', 'black')
    plt.xlabel('Episodes', fontsize=14)
    plt.ylabel('Episode Reward', fontsize=14)
    plt.legend()
    plt.grid(True)
    # plt.show()
    export_pdf.savefig()
    plt.close()

with PdfPages(r'./third_quarter_sequences.pdf') as export_pdf:
    plot_with_std(x, third_quarter_mean_with_feedback, 'with feedback', 'red')
    plot_with_std(x, third_quarter_mean_without_feedback, 'without feedback', 'black')
    plt.xlabel('Episodes', fontsize=14)
    plt.ylabel('Episode Reward', fontsize=14)
    plt.legend()
    plt.grid(True)
    # plt.show()
    export_pdf.savefig()
    plt.close()

with PdfPages(r'./forth_quarter_sequences.pdf') as export_pdf:
    plot_with_std(x, forth_quarter_mean_with_feedback, 'with feedback', 'red')
    plot_with_std(x, forth_quarter_mean_without_feedback, 'without feedback', 'black')
    plt.xlabel('Episodes', fontsize=14)
    plt.ylabel('Episode Reward', fontsize=14)
    plt.legend()
    plt.grid(True)
    # plt.show()
    export_pdf.savefig()
    plt.close()
