from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import argparse
sys.path.append('..')
import numpy as np
from os.path import basename, join, dirname, exists

import matplotlib.pyplot as plt
import csv


def read_fpr_tpr(file):
    with open(file, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if row[0] == 'FPR':
                fpr = row[1:]
            if row[0] == 'TPR':
                tpr = row[1:]
        f.close()
    fpr = [float(i) for i in fpr]
    tpr = [float(i) for i in tpr]
    return fpr, tpr


class Plotter(object):
    def __init__(self, fout, model_pths, datasets):
        self.fout = fout
        self.model_pths = model_pths
        self.datasets = datasets.split(',')
        print("Working on {} models, {} datasets".format(len(self.model_pths), len(self.datasets)))

        self.roc_files, self.labels, self.model_names = self.get_roc_files()
        assert len(self.roc_files) == len(self.model_pths) * len(self.datasets)

    @staticmethod
    def strip_model_name(model_pth):
        # Works with either ckpt/model input file name
        name = basename(model_pth)
        if name.endswith('.pth'):
            name = name.replace('.pth', '')
        elif name.endswith('.pth.tar'):
            name = name.replace('.pth.tar', '')

        return name.replace('model-', '').replace('ckpt-', '')

    @staticmethod
    def shorten_label(roc_file):
        label_name = basename(roc_file).replace('-roc.csv', '').split('-')

        if len(label_name) < 2:
            label_name = '-'.join(label_name)
            print("Label name: {} is too short, not trimming anything.".format(label_name))

        if  not label_name[-2].startswith('e_'):
            label_name = '-'.join(label_name)
            print("Label name: {} do not work with correct epoch format, not trimming anything".format(label_name))
        else:
            if label_name[0] == 'resumed':
                label_name = '-'.join(('resumed', label_name[1], label_name[-2]))
            else:
                label_name = '-'.join((label_name[0], label_name[-2]))  # network + epoch no.

        return label_name

    def get_roc_files(self):
        roc_files = []
        labels = []
        model_names = []
        for dataset in self.datasets:
            for path in self.model_pths:
                stripped_name = self.strip_model_name(path)
                roc_file = join(dirname(path), stripped_name + '-' + dataset + '-roc.csv')
                assert exists(roc_file), "{} does not exist!".format(roc_file)
                roc_files.append(roc_file)
                labels.append(self.shorten_label(roc_file))
                if stripped_name not in model_names:
                    model_names.append(stripped_name)

        return roc_files, labels, model_names

    def specify_color_linesytle(self):
        color_bank = ['r', 'c', 'b', 'g', 'y', 'k', 'w', 'm']
        linestyle_bank = ['-', '--', '-.'] # maximum represent 3 datasets
        color_dict = {}
        linestyle_dict = {}
        for i, m in enumerate(self.model_names):
            color_dict[m] = color_bank[i]
        for i, l in enumerate(self.datasets):
            linestyle_dict[l] = linestyle_bank[i]
        return color_dict, linestyle_dict

# ===============================================================================
    def dump_tprs(self):
        keys = ['Model', 'AUC', 'E-7', 'E-6', 'E-5', 'E-4', 'E-3', 'E-2', 'E-1']
        with open(self.fout.replace('.txt', '.csv'), 'w') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(keys)
            for item, label in zip(self.roc_files, self.labels):
                with open(item.replace('-roc', '-tprs'), 'r') as f:
                    reader = csv.reader(f)
                    for row in reader:
                        if row[0] != 'Models':
                            tprs = row
                            tprs[0] = label
                    f.close()
                writer.writerow(tprs)
            csvFile.close()

    def plot_roc(self, xmin=1e-6, xmax=1e-3):
        color_dict, linestyle_dict = self.specify_color_linesytle()
        plt.figure()
        xmin, xmax = xmin, xmax
        ymin, ymax = 1.0, 0.0
        for item, label in zip(self.roc_files, self.labels):
            for key, value in color_dict.items():
                if key in item:
                    color = value
            for key, value in linestyle_dict.items():
                if key in item:
                    linestyle = value
            fpr, tpr = read_fpr_tpr(item)

            min_ind = np.argmin(np.abs(np.asarray(fpr) - xmin))
            _ymin = np.asarray(tpr)[min_ind]
            if _ymin < ymin:
                ymin = _ymin

            max_ind = np.argmin(np.abs(np.asarray(fpr) - xmax))
            _ymax = np.asarray(tpr)[max_ind]
            if _ymax > ymax:
                ymax = _ymax

            plt.plot(fpr, tpr, color=color, linestyle=linestyle,
                     label=label, linewidth=1)
            del fpr, tpr

        plt.title('Receiver Operating Characteristic')
        plt.legend(loc='lower right')
        # plt.plot([0, 1], [0, 1], 'r--')
        plt.xlim([xmin, xmax])
        plt.xscale('log')
        plt.ylim([ymin, ymax])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.grid()
        plt.savefig(self.fout.replace('.txt', '.png'))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_pths', '-m', help='', nargs='+')
    parser.add_argument('--datasets', '-d',
                        # default='Adience_v0.0.1,IJBC_v0.0.2,MORPHacademic_v0.0.2',
                        default='Adience_v0.0.1,IJBC_v0.0.2',
                        help='evaluating which datasets')
    parser.add_argument('--fout', '-f', type=str, help="output dir")
    return parser.parse_args()


"""
    Whennever a file is called upon, Python will assign a name to it. There are 
    two ways a file will be called:
        1. called to executed, __name__ assgined to the file will be __main__
        2. called to be imported, __name__ assgined to be 'compare'
"""

if __name__ == "__main__":
    args = parse_args()
    plotter = Plotter(args.fout, args.model_pths, args.datasets)
    plotter.dump_tprs()
    plotter.plot_roc()
