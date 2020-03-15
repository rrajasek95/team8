import torch
import torch.nn as nn
import logging

"""
hred.py

This file contains the torch modules for building a
Bidirectional Hierarchical Recurrent Encoder-Decoder RNN model
as described in "Building End-To-End Dialogue Systems
Using Generative Hierarchical Neural Network Models" (Serban et al. 2016)

The model comprises of a word level encoder and a context level encoder
and a decoder which takes the context vector and generates an output sequence
"""

class WordEncoder(nn.Module):
    def __init__(self, 
        input_size,
        embed_size,
        hidden_size, 
        bidirectional):
        super(WordEncoder, self).__init__()
        self.embedding = nn.Embedding(input_size, embed_size)
        self.rnn = nn.LSTM(
            input_size=embed_size, 
            hidden_size=hidden_size, 
            batch_first=True, bidirectional=False)
        self.hidden_size = hidden_size

    def forward(self, x_in):
        # Currently we have [seq_len]
        logging.debug("Shape: {}".format(x_in.shape))
        # It needs to become [seq_len, batch_size], so we add an extra dimension
        x = x_in.unsqueeze(0)
        # x: [1, seq_len]
        embedded = self.embedding(x)
        # embedded: [1, seq_len, embed_size]
        logging.debug("embedded: {}".format(embedded.shape))
        output, (hidden, cell) = self.rnn(embedded)
        logging.debug("hidden: {}".format(hidden.shape))
        return hidden

class ContextEncoder(nn.Module):
    def __init__(self, word_encoder, hidden_size, device="cpu"):
        super(ContextEncoder, self).__init__()
        self.word_encoder = word_encoder
        self.rnn = nn.LSTM(
            input_size=word_encoder.hidden_size, 
            hidden_size=hidden_size, batch_first=True)
        self.device = device

    def forward(self, x_in):
        # logging.debug("{}".format(self.device))
        hidden_seq = torch.zeros(len(x_in), self.word_encoder.hidden_size).to(self.device)
        logging.debug("{}".format(hidden_seq.shape))
        for i, x in enumerate(x_in):
            hidden = self.word_encoder(x)
            logging.debug("{}".format(hidden.shape))
            hv = hidden.view(self.word_encoder.hidden_size)
            logging.debug("{}".format(hv.shape)) 
            hidden_seq[i] = hv

        # logging.debug("rnn_hiddens: {}".format(len(rnn_hiddens)))
        logging.debug("hidden_seq: {}".format(hidden_seq))
        hidden_seq = hidden_seq.unsqueeze(0)
        output, (hidden, cell) = self.rnn(hidden_seq)

        return output, (hidden, cell)


class Decoder(nn.Module):
    def __init__(self, 
        output_size,
        embed_size,
        hidden_size):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(output_size, embed_size)
        self.rnn = nn.LSTM(embed_size, hidden_size, 
            batch_first=True)
        self.fc_out = nn.Linear(hidden_size, output_size)
        self.output_size = output_size

    def forward(self, y_out, prev_hidden, prev_cell):
        logging.debug("y_out: {}".format(y_out.shape))
        embedded = self.embedding(y_out.unsqueeze(0))
        logging.debug("decoder embedding: {}".format(embedded.shape))
        output, hidden = self.rnn(embedded, (prev_hidden, prev_cell))

        pred = self.fc_out(output.squeeze(0))
        return pred, hidden

class Seq2Seq(nn.Module):
    def __init__(self, context_encoder, decoder, device="cpu"):
        super(Seq2Seq, self).__init__()
        self.context_encoder=context_encoder
        self.decoder=decoder
        self.device=device

    def forward(self, x_in, y_out):
        outputs = torch.zeros(len(x_in), y_out[0].shape[0], self.decoder.output_size, device=self.device)
        for i, x in enumerate(x_in):
            out, (hidden, cell) = self.context_encoder(x)

            decoder_input = y_out[i][0].unsqueeze(0)

            for t in range(y_out[i].shape[0]):
                output, (hidden, cell) = self.decoder(decoder_input, hidden, cell)
                outputs[i][t] = output

                decoder_input = output.argmax(dim=1)

            logging.debug(outputs.shape)

        return outputs