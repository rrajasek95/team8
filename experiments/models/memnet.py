import torch
import torch.nn as nn
import logging
import random
"""
memnet.py

This file implements the model from 
"A Knowledge-Grounded Neural Conversation Model" (Ghazvininejad et al. 2018)
which is a Seq2Seq based memory network
https://arxiv.org/pdf/1702.01932.pdf

The model comprises of:
1. A sentence encoder
2. Memory parameters that encode a representation for selecting relevant facts
3. Decoder that takes an encoded state and generates an utterance
"""


class SentenceEncoder(nn.Module):
    def __init__(self,
        input_size,
        embed_size,
        hidden_size,
        num_layers,
        bidirectional,
        dropout):
        super(SentenceEncoder, self).__init__()
        self.embedding = nn.Embedding(input_size, embed_size)
        self.rnn = nn.GRU(
            input_size=embed_size, 
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True, bidirectional=bidirectional)
        self.hidden_size = hidden_size

    def forward(self, x_in):
        logging.debug("Shape: {}".format(x_in.shape))
        # x: [batch_size, seq_len]
        embedded = self.embedding(x_in)
        # embedded: [batch_size, seq_len, embed_size]
        logging.debug("embedded: {}".format(embedded.shape))
        output, hidden = self.rnn(embedded)
        logging.debug("hidden: {}".format(output.shape))

        # output : [batch_size, seq_len,  num_directions * hidden_size  ]
        return output[:, -1] # [batch_size, num_directions * hidden_size]

class FactEncoder(nn.Module):
    def __init__(self,
        input_size,
        embed_size):
        super(FactEncoder, self).__init__()
        self.memory = nn.Linear(input_size, embed_size, bias=False)
        self.value = nn.Linear(input_size, embed_size, bias=False)

    def forward(self, context):
        # [batch_size, num_facts, vocab_size]
        logging.debug("Context shape: {}".format(context.shape))
        memory = self.memory(context)
        # [batch_size, num_facts, embed_size]

        value = self.value(context)
        # [batch_size, num_facts, embed_size]
        return memory, value

class InputEncoder(nn.Module):
    def __init__(self, 
        sentence_encoder: SentenceEncoder,
        fact_encoder: FactEncoder):
        super(InputEncoder, self).__init__()
        self.sentence_encoder = sentence_encoder
        self.fact_encoder = fact_encoder
        self.softmax = nn.Softmax(dim=1)

    def forward(self, sentence, context):
        logging.debug("Inputencoder: {}".format(sentence.shape))
        logging.debug("Inputencoder context: {}".format(context.shape))
        encoded_sentence = self.sentence_encoder(sentence).unsqueeze(2)
        # [num_dirs, batch_size, hidden_size]

        encoded_key, encoded_value = self.fact_encoder(context)

        # [batch_size, 1, num_directions * hidden_size]
        # [batch_size, num_facts, embed_size = ]
        encoded_key = encoded_key
        encoded_value = encoded_value
        logging.debug(encoded_sentence.shape)
        logging.debug(encoded_key.shape)
        logging.debug(encoded_value.shape)

        key_product = torch.bmm(encoded_key, encoded_sentence).squeeze(2)
        logging.debug(key_product.shape)
        key_probs = self.softmax( key_product)

        key_probs = key_probs.unsqueeze(1)
        logging.debug("Key probs shape: {}".format(key_probs.shape))
        logging.debug("Encoded value shape: {}".format(encoded_value.shape))
        memory_information = torch.bmm(key_probs, encoded_value).squeeze(1)
        logging.debug("Memory info shape: {}".format(memory_information.shape))
        return encoded_sentence.squeeze(2) + memory_information

class Decoder(nn.Module):
    def __init__(self, 
        output_size,
        embed_size,
        hidden_size,
        bidirectional):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(output_size, embed_size)
        self.rnn = nn.GRU(embed_size, hidden_size, 
            batch_first=True,
            bidirectional=bidirectional)
        self.fc_out = nn.Linear(hidden_size, output_size)
        self.output_size = output_size

    def forward(self, y_out, prev_hidden):
        # [batch_size]
        logging.debug(y_out.shape)
        logging.debug(prev_hidden.shape)
        embedded = self.embedding(y_out.unsqueeze(1))
        # [batch_size, 1, embed_dim]
        logging.debug(embedded.shape)
        output, hidden = self.rnn(embedded, prev_hidden)
        logging.debug(hidden)
        pred = self.fc_out(hidden)
        return pred, hidden


class LSTMSeq2Seq(nn.Module):
    """
    Special seq2seq model to handle LSTM models which have a cell state
    """
    def __init__(self, encoder, decoder, device):
        super(LSTMSeq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, source, context, target, loss_func, teacher_forcing_ratio=0.):
        outputs = torch.zeros(len(source), target[0].shape[0], self.decoder.output_size, device=self.device)
        encoded_sentence = self.encoder(source, context)
        max_length = target[0].shape[0]

        logging.debug("Target {}".format(target[:, 0]))
        logging.debug("Target {}".format(target[:, 0].shape))
        decoder_input = target[:, 0]
        decoder_hidden = encoded_sentence.unsqueeze(0)
        loss = 0.
        for t in range(max_length):
            logging.debug("Decoder input: {}".format(decoder_input.shape))
            output, hidden = self.decoder(decoder_input, decoder_hidden)
            logging.debug("Outputs shape: {}".format(outputs.shape))
            logging.debug("Output shape: {}".format(output.shape))
            logging.debug("Hidden: {}".format(hidden.shape))
            outputs[:, t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            argmax = output.squeeze(0).argmax(1)
            decoder_input = target[:, t] if teacher_force else argmax
            logging.debug("Decoder input shape: {}".format(decoder_input.shape))
            loss += loss_func(output.squeeze(0), target[:, t])
        return outputs, loss