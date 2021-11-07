# -*- coding: utf-8 -*-

import os
import argparse
from collections import OrderedDict

import pandas as pd
from sklearn.model_selection import train_test_split


def parser(): 
    parser = argparse.ArgumentParser('Convert unify-emotion-datasets to desired format')
    parser.add_argument('data_dir', help='Directory of unify-emotion-datasets')
    parser.add_argument('output_dir', help='Directory to write output files')
    parser.add_argument('--datasets', nargs='+', default=['emosti', 'emoint'], help='Datasets to convert')
    parser.add_argument('--sub_train_sizes', nargs='*', default=[100, 200, 500, 1000], help='Use smaller sets of training samples')
    parser.add_argument('--r_train', default=0.8, help='Training sample ratio')
    parser.add_argument('--r_dev', default=0.1, help='Development sample ratio')
    parser.add_argument('--r_test', default=0.1, help='Testing sample ratio')
    parser.add_argument('--seed', default=0, help='Random seed')
    args = parser.parse_args()
    assert args.r_train + args.r_dev + args.r_test == 1, 'Invalid split ratio'
    return args


def save_files(args, dataset, file_dict): 
    # sanity check
    assert dataset in args.datasets
    output_dir = os.path.join(args.output_dir, dataset)
    os.system('mkdir -p {}'.format(output_dir))
    # save emotion.txt
    assert 'emotions' in file_dict and isinstance(file_dict['emotions'], list)
    emotions = file_dict['emotions']
    print('Emotions:\n', emotions)
    with open(os.path.join(output_dir, 'emotions.txt'), "w") as f:
        f.write("\n".join(emotions))
    # save train.tsv and subsets
    assert 'data' in file_dict and all(col in file_dict['data'].columns for col in ['text', 'labels'])
    df = file_dict['data'][['text', 'labels']]
    train, others = train_test_split(df, train_size=args.r_train, random_state=args.seed, shuffle=True)
    train.to_csv(
        os.path.join(output_dir, 'train_max.tsv'), sep='\t', encoding="utf-8", header=False, index=False
    )
    for trainsize in args.sub_train_sizes: 
        filepath = os.path.join(output_dir, 'train_{}.tsv'.format(trainsize))
        train.iloc[:trainsize].to_csv(filepath, sep='\t', encoding="utf-8", header=False, index=False)
    # save dev.tsv and test.tsv
    dev, test = train_test_split(others, test_size=args.r_test / (args.r_dev + args.r_test))
    dev.to_csv(
        os.path.join(output_dir, 'dev.tsv'), sep='\t', encoding="utf-8", header=False, index=False
    )
    test.to_csv(
        os.path.join(output_dir, 'test.tsv'), sep='\t', encoding="utf-8", header=False, index=False
    )
    print('n_trian_examples: {}\tn_dev_examples: {}\tn_test_examples: {}'.format(len(train), len(dev), len(test)))


def prepare_emosti(args): 
    tag2label = OrderedDict({
        '<anger>': 0, 
        '<disgust>': 1, 
        '<fear>': 2, 
        '<happy>': 3, # joy
        '<sad>': 4, # sadness
        '<shame>': 5, 
        '<surprise>': 6
    })
    tag2rm = [
        *[item[0] + '\\' + item[1:] for item in tag2label.keys()], 
        '<cause>', 
        '<\\cause>'
    ]

    res = []
    for dataset in ['Emotion Cause.txt', 'No Cause.txt']: 
        with open(os.path.join(args.data_dir, 'emosti/{}'.format(dataset))) as f: 
            L = f.readlines()
        for i, text in enumerate(L): 
            found = False
            for token, label in tag2label.items(): 
                if token in text: 
                    if not found: 
                        y = label
                        text = text.replace(token, '')
                        found = True
                    else: 
                        raise ValueError('Multiple labels: {}'.format(text))
            if not found: 
                raise ValueError('Label not found: {}'.format(text))
            for token in tag2rm: 
                text = text.replace(token, '')
            L[i] = (text, y)
        res += L
    df = pd.DataFrame(res, columns=['text', 'labels'])

    emotions = list(map(lambda key: key[1:-1], tag2label.keys()))

    save_files(args, 'emosti', {'emotions': emotions, 'data': df})
    

def prepare_emoint(args): 
    df = pd.read_csv(os.path.join(args.data_dir, 'emoint/emoint_all'), sep='\t', header=None)

    emo2label = OrderedDict({
        'anger': 0, 
        'joy': 1, 
        'sadness': 2, 
        'fear': 3
    })

    df = pd.DataFrame(
        df.apply(lambda row: (row[1], emo2label[row[2]]), axis=1).values.tolist(), 
        columns=['text', 'labels']
    )
    emotions = list(emo2label.keys())

    save_files(args, 'emoint', {'emotions': emotions, 'data': df})


def main(): 
    args = parser()
    processor_mapping = {
        'emosti': prepare_emosti, # Emotion-Stimulus
        'emoint': prepare_emoint # EmoInt
    }
    for dataset in args.datasets: 
        if dataset in processor_mapping: 
            print('Preparing dataset: {}'.format(dataset))
            processor_mapping[dataset](args)
        else: 
            print('Not implemented: {}'.format(dataset))


if __name__ == '__main__': 
    main()
