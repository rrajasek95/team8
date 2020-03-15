import torch
import torch.nn as nn

from torch.utils.data import DataLoader
import argparse
import logging

from collections import defaultdict

def collate_hred(batch_list, vectorizer):
    logging.debug(batch_list)
    batch_dict = defaultdict(list)
    ys = [item['y_target'] for item in batch_list]
    ylens = torch.Tensor([len(y) for y in ys])
    ys_padded = nn.utils.rnn.pad_sequence(ys, batch_first=True, padding_value=vectorizer.pad_idx)
    
    for item in batch_list:
        batch_dict['x_data'].append(item['x_data'])
    batch_dict['y_target'] = ys_padded
    return batch_dict


def collate_memnet(batch_list, utt_vec, fact_vec):
    batch_dict = defaultdict(list)
    xs = [item['x_data'] for item in batch_list]
    xlens = torch.Tensor([len(x) for x in xs])
    ys = [item['y_target'] for item in batch_list]
    ylens = torch.Tensor([len(y) for y in ys])
    xs_padded = nn.utils.rnn.pad_sequence(xs, batch_first=True, padding_value=utt_vec.pad_idx)
    ys_padded = nn.utils.rnn.pad_sequence(ys, batch_first=True, padding_value=utt_vec.pad_idx)

    facts = torch.stack([item['x_facts'] for item in batch_list])
    batch_dict['x_facts'] = facts
    batch_dict['x_data'] = xs_padded
    batch_dict['y_target'] = ys_padded

    return batch_dict


def prepare_hred_dataloader(datasets, vectorizer, args):
    train_loader = DataLoader(datasets.train, batch_size=args.train_batch_size, collate_fn=lambda batch: collate_hred(batch, vectorizer))
    val_loader = DataLoader(datasets.val, batch_size=args.val_batch_size, collate_fn=lambda batch: collate_hred(batch, vectorizer))
    test_loader = DataLoader(datasets.test, batch_size=args.test_batch_size, collate_fn=lambda batch: collate_hred(batch, vectorizer))

    loaders = argparse.Namespace(
        train=train_loader,
        val=val_loader,
        test=test_loader)

    return loaders

def prepare_memnet_dataloader(datasets, utt_vec, fact_vec, args):
    train_loader = DataLoader(datasets.train, batch_size=args.train_batch_size, collate_fn=lambda batch: collate_memnet(batch, utt_vec, fact_vec))
    dev_loader = DataLoader(datasets.dev, batch_size=args.train_batch_size, collate_fn=lambda batch: collate_memnet(batch, utt_vec, fact_vec))
    test_loader = DataLoader(datasets.test, batch_size=args.train_batch_size, collate_fn=lambda batch: collate_memnet(batch, utt_vec, fact_vec))

    loaders = argparse.Namespace(
        train=train_loader,
        dev=dev_loader,
        test=test_loader
        )

    return loaders