#!/usr/bin/env python3

from matplotlib import pyplot as plt
import numpy as np
from os import listdir
from os.path import exists, join, dirname
from sys import stderr
from typing import Dict, List

from sys import argv


def get_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--log-path', help='Training log.')
    return parser.parse_args()


def transform_lines(lines: Dict[str, List]):
    plot_points = 100
    value_cnt = len(lines['iter'])

    if value_cnt > plot_points:
        values_per_point = value_cnt // plot_points
        leftovers = value_cnt % plot_points
    else:
        values_per_point = 1
        leftovers = 0

    # average values into points
    for key, arr in lines.items():
        if type(arr[0]) == int:
            arr_t = np.int
        elif type(arr[0]) == float:
            arr_t = np.float
        else:
            raise Exception("Unexpected value type!")

        # leave out few last values, so that each point in plot
        # averages the same number of values
        if leftovers:
            arr = np.array(arr[:-leftovers]).reshape((-1, values_per_point))
        else:
            arr = np.array(arr).reshape((-1, values_per_point))

        if key == 'iter':
            lines[key] = arr.max(axis=1)
        else:
            lines[key] = np.average(arr, axis=1)

    return lines


def parse_log(log):
    lines = {'iter': [], 'train': []}#, 'test': [], 'psnr': [], 'diff_psnr': [], 'identity_dist': []}
    lr_marks = []

    with open(log, 'r') as file:
        for line in file:
            if line.startswith('LearningRateAdapted'):
                lr_marks.append(lines['iter'][-1])
                continue

            fields = line.split(" ")

            lines['iter'].append(int(fields[0].split(":")[-1]))
            lines['train'].append(float(fields[1].split(":")[-1]))
            #lines['test'].append(float(fields[2].split(":")[-1]))
            #lines['psnr'].append(float(fields[3].split(":")[-1]))
            #lines['diff_psnr'].append(float(fields[4].split(":")[-1]))
            #lines['identity_dist'].append(float(fields[5].split(":")[-1]))
    return lines, lr_marks


def make_plot(log_path):
    folder = dirname(log_path)
    #    if args.log_path is None else args.log_path

    lines, lr_marks = parse_log(log_path)

    line_width = 1

    if len(lines['iter']) == 0:
        return

    lines = transform_lines(lines)

    chart_fig, (loss_plt) = plt.subplots(1)
    loss_plt.set_title("Trénovacia chyba")
    #tr_line, = loss_plt.semilogy(lines['iter'], lines['train'], lw=line_width, label='Train loss')
    #ts_line, = loss_plt.semilogy(lines['iter'], lines['test'], lw=line_width, label='Test loss')
    tr_line, = loss_plt.plot(lines['iter'], lines['train'], lw=line_width, label='Train loss')
    #ts_line, = loss_plt.plot(lines['iter'], lines['test'], lw=line_width, label='Test loss')

    for coord in lr_marks:
        loss_plt.axvline(x=coord, color='r')

    loss_plt.legend([tr_line], ('Train loss',), loc='upper right')
    loss_plt.set_xlim(0, lines['iter'][-1])

    #psnr_plt.set_title("PSNR")
    #psnr_plt.plot(lines['iter'], lines['psnr'], lw=line_width)

    #diff_psnr_plt.set_title("Zlepšenie PSNR voči bil.")
    #diff_psnr_plt.plot(lines['iter'], lines['diff_psnr'], lw=line_width)

    chart_fig.tight_layout()
    chart_fig.savefig(folder + "/plot.png")
    #chart_fig.show()


def main():
    folders = []

    args = get_args()

    if args.log_path is None:
        for folder in listdir("outputs"):
            if not exists(join("outputs", folder, "plot.png")) and "gan" not in folder.lower():
                folders.append(join("outputs", folder, "log.txt"))
    else:
        folders = [args.log_path]

    for folder in folders:
        try:
            make_plot(folder)
        except Exception as e:
            print(f"Plotting in folder \"{folder}\" raised error:\n{str(e)}\n", file=stderr)

    #plt.show()

if __name__ == "__main__":
    main()

