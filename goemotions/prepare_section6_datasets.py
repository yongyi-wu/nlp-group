# -*- coding: utf-8 -*-

import os
import argparse
from collections import OrderedDict
from xml.etree import ElementTree

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


DATASET_MAPPING = OrderedDict({
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


def parser(): 
    parser = argparse.ArgumentParser('Convert unify-emotion-datasets to desired format')
    parser.add_argument('data_dir', help='Directory of unify-emotion-datasets')
    parser.add_argument('output_dir', help='Directory to write output files')
    parser.add_argument('--datasets', nargs='+', default=list(DATASET_MAPPING.keys()), help='Datasets to convert')
    parser.add_argument('--sub_train_sizes', nargs='*', default=[100, 200, 500, 1000], help='Use smaller sets of training samples')
    parser.add_argument('--r_train', default=0.8, help='Training sample ratio')
    parser.add_argument('--r_dev', default=0.1, help='Development sample ratio')
    parser.add_argument('--r_test', default=0.1, help='Testing sample ratio')
    parser.add_argument('--seed', default=0, help='Random seed')
    args = parser.parse_args()
    assert args.r_train + args.r_dev + args.r_test == 1, 'Invalid split ratio'
    assert set(args.datasets).issubset(set(DATASET_MAPPING.keys())), 'Unknown dataset'
    return args


def save_files(args, dataset, emotions, df): 
    # sanity check
    assert dataset in args.datasets
    output_dir = os.path.join(args.output_dir, dataset)
    os.system('mkdir -p {}'.format(output_dir))
    # save emotion.txt
    assert isinstance(emotions, list)
    print('[Emotions]\t', emotions)
    with open(os.path.join(output_dir, 'emotions.txt'), 'w') as f:
        f.write('\n'.join(emotions))
    # save train.tsv and subsets
    assert all(col in df.columns for col in ['text', 'labels'])
    df = df[['text', 'labels']].astype(str).dropna()
    df = df[(df['text'] != '') & (df['labels'] != '')]
    train, others = train_test_split(df, train_size=args.r_train, random_state=args.seed, shuffle=True)
    train.to_csv(
        os.path.join(output_dir, 'train_max.tsv'), sep='\t', encoding="utf-8", header=False, index=False
    )
    for trainsize in args.sub_train_sizes: 
        sub_train_path = os.path.join(output_dir, 'train_{}.tsv'.format(trainsize))
        train.iloc[:trainsize].to_csv(sub_train_path, sep='\t', encoding="utf-8", header=False, index=False)
    # save dev.tsv and test.tsv
    dev, test = train_test_split(others, test_size=args.r_test / (args.r_dev + args.r_test))
    dev.to_csv(
        os.path.join(output_dir, 'dev.tsv'), sep='\t', encoding="utf-8", header=False, index=False
    )
    test.to_csv(
        os.path.join(output_dir, 'test.tsv'), sep='\t', encoding="utf-8", header=False, index=False
    )
    print('n_trian_max_examples: {}\tn_dev_examples: {}\tn_test_examples: {}'.format(len(train), len(dev), len(test)))


def prepare_dialog(args): 
    with open(os.path.join(args.data_dir, 'dailydialog/ijcnlp_dailydialog/dialogues_emotion.txt')) as f: 
        labels = f.readlines()
    labels = list(map(lambda y: ','.join(set(y.strip().split())), labels))
    with open(os.path.join(args.data_dir, 'dailydialog/ijcnlp_dailydialog/dialogues_text.txt')) as f: 
        text = f.readlines()
    text = list(map(lambda s: "\" " + s.strip().replace("__eou__", "\" \"")[:-2], text))
    df = pd.DataFrame({'text': text, 'labels': labels})
    emotions = ['no emotion', 'anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise']
    return emotions, df


def prepare_stimulus(args): 
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
        with open(os.path.join(args.data_dir, 'emotion-cause/Dataset/{}'.format(dataset))) as f: 
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
    return emotions, df


def prepare_affective(args): 
    text = []
    labels = []
    for split in ['trial', 'test']: 
        text_path = os.path.join(args.data_dir, 'affectivetext/AffectiveText.{}/affectivetext_{}.xml'.format(split, split))
        os.system('sed -i "s/&/and/g" {}'.format(text_path)) # replace bad tokens
        for child in ElementTree.parse(text_path).getroot(): 
            text.append(child.text)
        label_path = os.path.join(args.data_dir, 'affectivetext/AffectiveText.{}/affectivetext_{}.emotions.gold'.format(split, split))
        with open(label_path) as f: 
            L = f.readlines()
        for line in L: 
            y = []
            for i, score in enumerate(line.strip().split()[1:]): 
                if int(score) > 0: 
                    y.append(str(i))
            labels.append(','.join(y))
    df = pd.DataFrame({'text': text, 'labels': labels})
    emotions = ['anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise']
    return emotions, df


def prepare_crowd(args): 
    df = pd.read_csv(os.path.join(args.data_dir, 'crowdflower/text_emotion.csv'))
    df = df.rename(columns={'content': 'text', 'sentiment': 'emotions'})
    emotions, df['labels'] = np.unique(df['emotions'].values, return_inverse=True)
    emotions = emotions.tolist()
    return emotions, df


def prepare_elect(args): 
    df = pd.DataFrame(columns=['text', 'emotions'])
    for batch in ['Batch1', 'Batch2']: 
        file_path = os.path.join(args.data_dir, 'electoraltweets/Annotated-US2012-Election-Tweets/Questionnaire2/{}/AnnotatedTweets.txt'.format(batch))
        batch_df = pd.read_csv(file_path, sep='\t', usecols=[13, 15])
        batch_df = batch_df.rename(columns={'tweet': 'text', 'q2whatemotionchooseoneoftheoptionsfrombelowthatbestrepresentstheemotion': 'emotions'})
        batch_df = batch_df.dropna()
        df = df.append(batch_df)
    emotions, df['labels'] = np.unique(df['emotions'].values, return_inverse=True)
    emotions = emotions.tolist()
    return emotions, df


def prepare_isear(args): 
    df = pd.read_csv(
        os.path.join(args.data_dir, 'isear/isear.csv'), sep='|', usecols=['Field1', 'SIT']
    )
    df = df.rename({'Field1': 'emotions', 'SIT': 'text'}, axis=1)
    emotions, df['labels'] = np.unique(df['emotions'].values, return_inverse=True)
    emotions = emotions.tolist()
    return emotions, df


def prepare_tec(args): 
    with open(os.path.join(args.data_dir, 'TEC/Jan9-2012-tweets-clean.txt')) as f: 
        L = f.readlines()
    text = []
    emotions = []
    for line in L: 
        start = line.index('\t')
        for end in range(len(line) - 1, start, -1): 
            if line[end] == '\t': 
                break
        text.append(line[start:end].strip())
        emotions.append(line[end + 3:].strip())
    df = pd.DataFrame({'text': text, 'emotions': emotions})
    emotions, df['labels'] = np.unique(df['emotions'].values, return_inverse=True)
    emotions = emotions.tolist()
    return emotions, df


def prepare_emoint(args): 
    emo2label = OrderedDict({
        'anger': 0, 
        'joy': 1, 
        'sadness': 2, 
        'fear': 3
    })
    df = pd.read_csv(os.path.join(args.data_dir, 'emoint/emoint_all'), sep='\t', header=None)
    df = pd.DataFrame(
        df.apply(lambda row: (row[1], emo2label[row[2]]), axis=1).values.tolist(), 
        columns=['text', 'labels']
    )
    emotions = list(emo2label.keys())
    return emotions, df


def prepare_ssec(args): 
    emo2label = OrderedDict({
        'Anger': 0,	
        'Anticipation': 1, 
        'Disgust': 2, 
        'Fear': 3, 
        'Joy': 4, 
        'Sadness': 5, 
        'Surprise': 6, 
        'Trust': 7
    })
    df = pd.DataFrame(columns=['text', 'emotions', 'labels'])
    for split in ['train', 'test']: 
        file_path = os.path.join(args.data_dir, 'ssec/ssec-aggregated/{}-combined-0.0.csv'.format(split))
        split_df = pd.read_csv(file_path, sep='\t', header=None)
        split_df = split_df.dropna()
        split_df = split_df.apply(
            lambda row: (row[8], ','.join([emo.strip() for emo in row[:-1] if emo != '---'])), 
            axis=1
        )
        split_df = pd.DataFrame(split_df.values.tolist(), columns=['text', 'emotions'])
        split_df = split_df[split_df['emotions'] != '']
        split_df['labels'] = split_df['emotions'].apply(
            lambda s: ','.join([str(emo2label[emo]) for emo in s.split(',')])
        )
        df = df.append(split_df)
    emotions = list(emo2label.keys())
    return emotions, df


def main(): 
    processor_mapping = {
        'dialog': prepare_dialog, 
        'stimulus': prepare_stimulus, 
        'affective': prepare_affective, 
        'crowd': prepare_crowd, 
        'elect': prepare_elect, 
        'isear': prepare_isear, 
        'tec': prepare_tec, 
        'emoint': prepare_emoint, 
        'ssec': prepare_ssec
    }
    assert set(DATASET_MAPPING.keys()) == set(processor_mapping.keys())
    args = parser()
    processed = {}
    for dataset in args.datasets: 
        if dataset not in processed: 
            print('Dataset: {}'.format(DATASET_MAPPING[dataset]))
            emotions, df = processor_mapping[dataset](args)
            save_files(args, dataset, emotions, df)


if __name__ == '__main__': 
    main()
