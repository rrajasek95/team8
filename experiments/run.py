import argparse
import torch
import torch.optim as optim
import torch.nn as nn

from dataset import HredDataset, MixedShortDataset, MemNetDataset
import os
import json
import logging
import pickle as pkl

from vocabulary import Vocabulary
from vectorizer import SequenceVectorizer, OneHotVocabVectorizer
from loader import prepare_hred_dataloader, prepare_memnet_dataloader

import train_model

import models.hred as hred
import models.memnet as memnet



def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def load_bidaf_train_data(args):
    path = os.path.join(args.data_dir,
        'experiment_data',
        'bidaf',
        'mixed_short',
        'train-v1.1.json')
    with open(path, 'r') as train_file:
        data = json.load(train_file)
    num_examples = len(data["data"])
    chats = data["data"]
    logging.debug("Loaded {} examples".format(num_examples))

    # Prepare the training data 
    train_dataset = MixedShortDataset(chats)

    return train_dataset

def load_hred_dataset(args, filename, vectorizer):
    path = os.path.join(args.data_dir,
        'experiment_data',
        'hred',
        filename)

    with open(path, 'r') as train_file:
        data = json.load(train_file)

    contexts = data[0]
    responses = data[1]
    logging.debug("Loaded {} examples".format(len(responses)))
    dataset = HredDataset(contexts, responses, vectorizer, args.device)

    return dataset

def load_memnet_data(args, filename):
    path = os.path.join(args.data_dir,
        'memnet_data',
        filename)

    with open(path, 'r') as train_file:
        data = json.load(train_file)

    return data

def load_memnet_train_data(args, contextvectorizer, factvectorizer):
    data = load_memnet_data(args, 'train_data.json')
    dataset = MemNetDataset(data, contextvectorizer, factvectorizer, args.device)

    return dataset

def load_memnet_dev_data(args, contextvectorizer, factvectorizer):
    data = load_memnet_data(args, 'dev_data.json')
    dataset = MemNetDataset(data, contextvectorizer, factvectorizer, args.device)

    return dataset

def load_memnet_test_data(args, contextvectorizer, factvectorizer):
    data = load_memnet_data(args, 'test_data.json')
    dataset = MemNetDataset(data, contextvectorizer, factvectorizer, args.device)

    return dataset


def load_hred_vocabulary(args):
    path = os.path.join(args.data_dir,
        'experiment_data',
        'hred',
        'words.pkl')

    with open(path, 'rb') as words_file:
        words = pkl.load(words_file)
    word_dict = {word: idx for idx, word in enumerate(words.keys())}
    logging.debug(type(words))
    vocab = Vocabulary.from_dict(word_dict)

    return vocab

def load_memnet_vocabulary(args):
    vocab_path = os.path.join(args.data_dir,
        'memnet_data',
        'vocab.pkl')
    with open(vocab_path, 'rb') as vocab_file:
        vocab = pkl.load(vocab_file)

    return vocab

def create_hred_model(args, vocab):
    word_encoder = hred.WordEncoder(
        input_size=len(vocab),
        embed_size=args.embed_size,
        hidden_size=args.word_hidden_size,
        bidirectional=args.bidirectional)
    context_encoder = hred.ContextEncoder(word_encoder, args.context_hidden_size, args.device)
    decoder = hred.Decoder(
        output_size=len(vocab),
        embed_size=args.embed_size,
        hidden_size=args.decoder_hidden_size)

    seq2seq = hred.Seq2Seq(context_encoder, decoder, args.device)
    
    return seq2seq

def create_memnet_model(args, utt_vocab, fact_vocab):
    sentence_encoder = memnet.SentenceEncoder(
        input_size=len(utt_vocab),
        embed_size=args.embed_size,
        hidden_size=args.word_hidden_size,
        num_layers=args.num_layers,
        bidirectional=args.bidirectional,
        dropout=args.dropout)
    num_directions = 2 if args.bidirectional else 1
    fact_encoder = memnet.FactEncoder(
        input_size=len(fact_vocab),
        embed_size=args.embed_size * num_directions )

    input_encoder = memnet.InputEncoder(
        sentence_encoder,
        fact_encoder)

    decoder = memnet.Decoder(
        output_size=len(utt_vocab),
        embed_size=args.embed_size,
        hidden_size=args.word_hidden_size * num_directions * args.num_layers,
        bidirectional=False)

    seq2seq = memnet.LSTMSeq2Seq(input_encoder, decoder, args.device)

    return seq2seq


def base_parser():
    parser = argparse.ArgumentParser(
        description="Run the experiments for the models with knowledge on the Holl-E dataset")
    parser.add_argument("--data_dir",
        default="holle/")
    parser.add_argument("--n_epochs", default=10, 
        type=int, help="Number of epochs to train the model for")
    parser.add_argument('--learning_rate', default=0.1, 
        type=float, help='Learning rate for the model')
    parser.add_argument('--train_batch_size', default=16, 
        type=int, help='Batch size for train')
    parser.add_argument('--val_batch_size', default=16, 
        type=int, help='Batch size for validation')
    parser.add_argument('--test_batch_size', default=16, 
        type=int, help='Batch size for test')
    parser.add_argument('--max_norm', default=1., type=float, help="Clipping value for gradient")
    parser.add_argument('--model', default="memnet",
        choices=['hred', 'memnet'], help='Model to train')
    parser.add_argument('--checkpoint_dir', default="checkpoints/",
        help="Path to save model checkpoints to")

    parser.add_argument('--run_dir', default="runs/",
        help="Directory to store run information for tensorboard")

    parser.add_argument('--debug', action='store_true',
        help='Display debug output')

    return parser

def hred_parser():
    parser = argparse.ArgumentParser(
        description="Parameters for hred")
    parser.add_argument("--word_hidden_size", default=100,
        type=int, help="Hidden size of word encoder")
    parser.add_argument("--context_hidden_size", default=100,
        type=int, help="Hidden size of context encoder")
    parser.add_argument("--decoder_hidden_size", default=100,
        type=int, help="Hidden size of decoder")
    parser.add_argument("--embed_size", default=50,
        type=int, help="Token embedding dimension")
    parser.add_argument("--bidirectional", default=True,
        type=bool, help="Bidirectional model config")
    return parser

def memnet_parser():
    parser = argparse.ArgumentParser(
        description="Parameters for memnet")
    parser.add_argument("--word_hidden_size", default=100,
        type=int, help="Hidden size of word encoder")
    parser.add_argument("--context_hidden_size", default=100,
        type=int, help="Hidden size of context encoder")
    parser.add_argument("--decoder_hidden_size", default=100,
        type=int, help="Hidden size of decoder")
    parser.add_argument("--embed_size", default=100,
        type=int, help="Token embedding dimension")
    parser.add_argument("--bidirectional", default=True,
        type=bool, help="Bidirectional model config")
    parser.add_argument("--num_layers", default=1,
        type=int, help="Number of layers for models")
    parser.add_argument("--dropout", default=0.1,
        type=float, help="Dropout probability for weights")

    return parser


def run_memnet_experiment(args):
    (memnet_args, extras) = memnet_parser().parse_known_args()
    vocab = load_memnet_vocabulary(args)

    VEC = SequenceVectorizer(
        vocab,
        init_token="[BOS]",
        eos_token="[EOS]",
        pad_token="[PAD]"
        )

    FACTVEC = OneHotVocabVectorizer(vocab)

    logging.debug("Loading train dataset")
    train_dataset = load_memnet_train_data(args, VEC, FACTVEC)
    logging.debug("Loading dev dataset")
    dev_dataset = load_memnet_dev_data(args, VEC, FACTVEC)
    logging.debug("Loading test dataset")
    test_dataset = load_memnet_test_data(args, VEC, FACTVEC)

    datasets = argparse.Namespace(
        train=train_dataset,
        dev=dev_dataset,
        test=test_dataset)

    loaders = prepare_memnet_dataloader(datasets, VEC, FACTVEC, args)
    memnet_args.device = args.device

    model = create_memnet_model(memnet_args, VEC.vocab, FACTVEC.vocab)
    model.to(args.device)
    logging.info("Model parameters: {}".format(count_parameters(model)))
    optimizer = optim.Adam(model.parameters())
    loss_func = nn.CrossEntropyLoss()
    train_model.train_memnet_model(model, optimizer, loss_func, loaders, args)

def run_hred_experiment(args):
    (hred_args, extras) = hred_parser().parse_known_args()
    vocab = load_hred_vocabulary(args)

    # Using the identity function since the input is already tokenized
    VEC = SequenceVectorizer(vocab, 
        tokenizer=lambda x: x,
        init_token="[BOS]",
        eos_token="[EOS]",
        pad_token="[PAD]")

    logging.debug("Loading train dataset")
    train_dataset = load_hred_dataset(args, 'train.json', VEC)
    logging.debug("Loading validation dataset")
    val_dataset = load_hred_dataset(args, 'dev.json', VEC)
    logging.debug("Loading test dataset")
    test_dataset = load_hred_dataset(args, 'test.json', VEC)

    datasets = argparse.Namespace(
        train=train_dataset,
        val=val_dataset,
        test=test_dataset)

    loaders = prepare_hred_dataloader(datasets, VEC, args)
    hred_args.device = args.device
    model = create_hred_model(hred_args, vocab).to(args.device)
    logging.info("Model parameters: {}".format(count_parameters(model)))
    optimizer = optim.Adam(model.parameters())
    loss_func = nn.CrossEntropyLoss()
    train_model.train_hred_model(model, optimizer, loss_func, loaders, args)




def main():
    (args, extras) = base_parser().parse_known_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)

    if args.model == "hred":
        run_hred_experiment(args)
    if args.model == "memnet":
        run_memnet_experiment(args)

if __name__ == '__main__':
    main()