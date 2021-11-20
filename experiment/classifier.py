# -*- coding: utf-8 -*-

import os
import json
import argparse

from utils import make_if_not_exists, seed_everything


def parse(save=False): 
    parser = argparse.ArgumentParser('GoEmotions multi-label classifier')
    parser.add_argument('exp_name', type=str, help='Experiment name (be specific and distinguishable)')
    parser.add_argument('--data_dir', type=str, default=os.path.join('..', 'goemotions', 'data'), help='Directory with datasets and the emotion txt file')
    # TODO: use from_pretrained() directly?
    # parser.add_argument('--bert_dir', type=str, help='Directory with vocab, config and checkpoint')
    parser.add_argument('--output_dir', type=str, default='out', help='Directory to output results')
    run = parser.add_argument_group(title='Run', description='Parameters related to running the script')
    run.add_argument('--max_len', type=int, default=30, help='Maximum sequence length to the Transformers')
    run.add_argument('--checkpoint', type=str, help='Path to a saved checkpoint')
    run.add_argument('--no_train', action='store_false', help='Skip the training phase')
    run.add_argument('--no_dev', action='store_false', help='Skip the evaluation on the development dataset')
    run.add_argument('--test', action='store_true', help='Evaluate on the test dataset')
    run.add_argument('--seed', type=int, default=0, help='Random seed for reproduction')
    train = parser.add_argument_group(title='Training', description='Parameters related to training the model')
    train.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    train.add_argument('--lr', type=float, default=5E-5, help='Initial learning rate')
    train.add_argument('--n_epochs', type=int, default=4, help='Number of training epochs')
    train.add_argument('--warmup_proportion', type=float, default=0.1, help='Proportion of training steps to do linear lr warmup')
    train.add_argument('--pred_thold', type=float, default=0.3, help='Prediction threshold for each emotion')
    args = parser.parse_args()
    if save: 
        make_if_not_exists(args.output_dir)
        with open(os.path.join(args.output_dir, 'cfg.json'), 'w') as f: 
            json.dump(vars(args), f, indent=4)
    return args


def main(): 
    args = parse(save=True)
    print(args)
    seed_everything(args.seed)


if __name__ == '__main__': 
    main()
