# -*- coding: utf-8 -*-

from tqdm import tqdm
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from utils import convert_to_unicode


class GoEmotionsDataset(Dataset): 
    """A class to handle data processing and reading"""
    def __init__(self, data_path, tokenizer, n_labels, max_len, is_test=False): 
        """
        Input
        ----------
        data_path: 
            Path to the dataset {train.tsv, dev.tsv, test.tsv}
        tokenizer: 
            Transform raw text to token ids
        n_labels: 
            Total number of emotions, including Neutral
        max_len: 
            Maximum input sequence length for Transformers
        is_test: 
            If True, to do not process labels
        """
        super().__init__()
        print('Processing data from {}'.format(data_path))
        cols = ['text'] if is_test else ['text', 'labels']
        self.df = pd.read_csv(
            data_path, 
            sep='\t', 
            encoding='utf-8', 
            header=None, 
            usecols=cols, 
            names=cols, 
            dtype={'text': str}
        )
        self.df.reset_index(drop=True, inplace=True)
        self.data = GoEmotionsDataset.prepare_data(self.df, tokenizer, n_labels, max_len)
        assert len(self.df) == len(self.data)

    @staticmethod
    def prepare_data(df, tokenizer, n_labels, max_len): 
        df.reset_index(drop=True, inplace=True)
        data = []
        for _, row in tqdm(df.iterrows(), total=len(df), dynamic_ncols=True): 
            # tokenize raw text and convert to ids
            if isinstance(row['text'], float): 
                text = ''  # This accounts for rare encoding errors
            else: 
                text = convert_to_unicode(row['text'])
            tokens = tokenizer.tokenize(text)
            if len(tokens) > max_len - 2: 
                tokens = tokens[0:max_len - 2]
            tokens = ['[CLS]'] + tokens + ['[SEP]']
            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            # prepare other input
            input_mask = [1] * len(input_ids)
            seg_ids = [0] * len(tokens)
            # create multi-hot label encoding
            if 'labels' in row.name: 
                labels = [0] * n_labels
                for label in str(row['labels'].split(',')): 
                    labels[int(label)] = 1
            else: 
                labels = None
            # pad to max_len
            while len(input_ids) < max_len: 
                input_ids.append(0)
                input_mask.append(0)
                seg_ids.append(0)
            assert len(input_ids) == len(input_mask) == len(seg_ids) == max_len
            data.append({
                'input_ids': np.array(input_ids, dtype=int), 
                'input_mask': np.array(input_mask, dtype=int), 
                'seg_ids': np.array(seg_ids, dtype=int), 
                'labels': np.array(labels, dtype=int) if labels is not None else None
            })
        return data

    def __len__(self): 
        return len(self.data)

    def __getitem__(self, idx): 
        """
        Output
        ----------
        A dictionary where all elements have been truncated or right-padded to max_len
        """
        return self.data[idx]
