# -*- coding: utf-8 -*-

import os
import argparse
import json
from collections import OrderedDict

from matplotlib import pyplot as plt


def main(): 
    # parse arguments
    parser = argparse.ArgumentParser('Plot results (Macro-averaged F1) on 9 datasets for section 6')
    parser.add_argument('all_result_dir', help='Directory with results on all 9 datasets')
    parser.add_argument('out_dir', help='Directory to output the plot')
    args = parser.parse_args()
    # read results
    results = {}
    for curr, sub, _ in os.walk(args.all_result_dir): 
        if sub != []: 
            continue
        dataset = os.path.basename(curr)
        results[dataset] = {}
        for setup in ['baseline', 'freeze', 'nofreeze']: 
            for size in ['100', '200', '500', '1000', 'max']: 
                result = json.loads(
                    open(os.path.join(curr, size + '_' + setup + '.json')).readline()
                )
                f1s, sizes = results[dataset].get(setup, ([], []))
                f1s.append(result['macro_f1'])
                sizes.append(size)
                results[dataset][setup] = (f1s, sizes)
    # plot the figure
    setup2legend = {
        'baseline': 'Baseline', 
        'freeze': 'Transfer Learning w/ Freezing Layers', 
        'nofreeze': 'Transfer Learning w/o Freezing Layers'
    }
    dataset2name = OrderedDict({
        'dialog': 'DailyDialog', 
        'stimulus': 'Emotion-Stimulus', 
        'affective': 'Affective Text', 
        'crowd': 'CrowdFlower', 
        'elect': 'ElectoralTweets', 
        'isear': 'ISEAR', 
        'tec': 'TEC', 
        'emoint': 'EmoInt', 
        'ssec': 'SSEC'
    })
    setup2color = {
        'baseline': 'C5', 
        'freeze': 'C0', 
        'nofreeze': 'C1'
    }
    xticks = [100, 200, 500, 1000, 1250]
    fig, axs = plt.subplots(3, 3, sharex='col')
    fig.set_size_inches(16.5, 10.5)
    for i, dataset in enumerate(dataset2name.keys()): 
        row = i // 3
        col = i % 3
        for setup, (f1s, sizes) in results[dataset].items(): 
            axs[row, col].set_xticks([100, 200, 500, 1000, 1250])
            axs[row, col].set_xticklabels(sizes)
            axs[row, col].plot(xticks, f1s, color=setup2color[setup], label=setup2legend[setup])
        axs[row, col].set_title(dataset2name[dataset])
    plt.setp(axs[-1, :], xlabel='Training Set Size')
    plt.setp(axs[:, 0], ylabel='F-score')
    fig.legend(*fig.gca().get_legend_handles_labels(), loc='lower center', ncol=3, bbox_to_anchor=(0.5, 0.025))
    fig.savefig(os.path.join(args.out_dir, 'section6_f1.png'))


if __name__ == '__main__': 
    main()
