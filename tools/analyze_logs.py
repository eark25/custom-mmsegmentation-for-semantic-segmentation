# Copyright (c) OpenMMLab. All rights reserved.
"""Modified from https://github.com/open-
mmlab/mmdetection/blob/master/tools/analysis_tools/analyze_logs.py."""
import argparse
import json
from collections import defaultdict
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from statistics import mean


def plot_curve(log_dicts, args):
    if args.backend is not None:
        plt.switch_backend(args.backend)
    sns.set_style(args.style)
    # if legend is None, use {filename}_{key} as legend
    legend = args.legend
    if legend is None:
        legend = []
        for json_log in args.json_logs:
            for metric in args.keys:
                legend.append(f'{json_log}_{metric}')
    assert len(legend) == (len(args.json_logs) * len(args.keys))
    metrics = args.keys

    num_metrics = len(metrics)
    # print(metrics)
    # import sys
    # sys.exit(0)
    plt.figure()
    for i, log_dict in enumerate(log_dicts):
        # import sys
        # sys.exit(0)
        epochs = list(log_dict.keys())
        # print(log_dict)
        # import sys
        # sys.exit(0)
        for j, metric in enumerate(metrics):
            print(f'plot curve of {args.json_logs[i]}, metric is {metric}')
            plot_epochs = []
            plot_iters = []
            plot_values = []
            # In some log files exist lines of validation,
            # `mode` list is used to only collect iter number
            # of training line.
            for epoch in epochs:
                # print(epoch)
                epoch_logs = log_dict[epoch]
                # print(epoch_logs.keys())
                if metric not in epoch_logs.keys():
                    continue
                if metric in ['mIoU', 'mAcc', 'aAcc', 'loss', 'loss_val']:
                    if epoch % 1 == 0:
                        # print(epoch)
                        # print(epoch_logs[metric])
                        # print(len(epoch_logs[metric]))
                        # print(round(mean(epoch_logs[metric]), 5))
                        plot_epochs.append(epoch)
                        # plot_values.append(epoch_logs[metric][0])
                        plot_values.append(round(mean(epoch_logs[metric]), 5))
                else:
                    # print('use this?')
                    # print(range(len(epoch_logs[metric])))
                    for idx in range(len(epoch_logs[metric])):
                        # print(idx)
                        # print(epoch_logs)
                        # print(epoch_logs['mode'][idx] == 'train')
                        # print(epoch_logs['mode'][idx] == 'val')
                        # print(epoch_logs)
                        # print(epoch_logs['mode'][idx])
                        # if epoch_logs['mode'][idx] == 'val':
                        #     print('There it is !')
                        #     import sys
                        #     sys.exit(0)
                        if epoch_logs['mode'][idx] == 'train':
                            plot_epochs.append(epoch)
                            plot_iters.append(epoch_logs['iter'][idx])
                            plot_values.append(epoch_logs[metric][idx])
            # import sys
            # sys.exit(0)
            # print(len(plot_epochs))
            # print(len(plot_values))
            # plt.figure(figsize=(10, 10))
            plt.gcf()
            ax = plt.gca()
            label = legend[i * num_metrics + j]
            if metric in ['mIoU', 'mAcc', 'aAcc', 'loss', 'loss_val']:
                ax.set_xticks(np.arange(0, len(plot_epochs) + 1, 10))
                # ax.set_xticks(np.arange(0, 1000 + 1, 100))
                ax.set_ylim([0, 50])
                # ax.set_yticks(np.arange(0, 3, .5))
                plt.xlabel('epoch')
                plt.plot(plot_epochs, plot_values, label=label, linewidth=0.5)#, marker='.')
            else:
                plt.xlabel('iter')
                plt.plot(plot_iters, plot_values, label=label, linewidth=0.5)
        plt.legend()
        if args.title is not None:
            plt.title(args.title)
    if args.out is None:
        plt.show()
    else:
        plt.grid()
        print(f'save curve to: {args.out}')
        plt.savefig(args.out)
        plt.cla()


def parse_args():
    parser = argparse.ArgumentParser(description='Analyze Json Log')
    parser.add_argument(
        'json_logs',
        type=str,
        nargs='+',
        help='path of train log in json format')
    parser.add_argument(
        '--keys',
        type=str,
        nargs='+',
        default=['mIoU'],
        help='the metric that you want to plot')
    parser.add_argument('--title', type=str, help='title of figure')
    parser.add_argument(
        '--legend',
        type=str,
        nargs='+',
        default=None,
        help='legend of each plot')
    parser.add_argument(
        '--backend', type=str, default=None, help='backend of plt')
    parser.add_argument(
        '--style', type=str, default='dark', help='style of plt')
    parser.add_argument('--out', type=str, default=None)
    args = parser.parse_args()
    return args


def load_json_logs(json_logs):
    # load and convert json_logs to log_dict, key is epoch, value is a sub dict
    # keys of sub dict is different metrics
    # value of sub dict is a list of corresponding values of all iterations
    log_dicts = [dict() for _ in json_logs]
    for json_log, log_dict in zip(json_logs, log_dicts):
        # print('1')
        with open(json_log, 'r') as log_file:
            for line in log_file:
                # print('2')
                # print(line)
                # print('strip', line.strip())
                log = json.loads(line.strip())
                # print('loads', log)
                
                # skip lines without `epoch` field
                if 'epoch' not in log:
                    continue
                # epoch = log.pop('epoch')
                epoch = log['epoch']
                # print('pop log', log)
                # print(log_dict)
                if epoch not in log_dict:
                    # print(epoch)
                    # print(defaultdict(list))
                    log_dict[epoch] = defaultdict(list)
                    # print(log_dict)
                    # print(log)
                    # print(log.items())
                for k, v in log.items():
                    # print(k, v)
                    # import sys
                    # sys.exit(0)
                    log_dict[epoch][k].append(v)
                # print(log_dicts)
                # import sys
                # sys.exit(0)
    return log_dicts


def main():
    args = parse_args()
    json_logs = args.json_logs
    for json_log in json_logs:
        assert json_log.endswith('.json')
    log_dicts = load_json_logs(json_logs)
    plot_curve(log_dicts, args)


if __name__ == '__main__':
    main()
