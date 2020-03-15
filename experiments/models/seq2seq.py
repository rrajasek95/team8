import torch.nn as nn
import torch
import random

class LSTMEncoder(nn.Module):
    def __init__(self, input_size, embed_size, hidden_size, num_layers, bidirectional, dropout):
        super(LSTMEncoder, self).__init__()

        self.source_embedding = nn.Embedding(input_size, embed_size)

        self.birnn = nn.LSTM(embed_size, hidden_size, num_layers, bidirectional=bidirectional)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, lengths):
        # x_in: (seq_len, batch_size)
        x_embed = self.dropout(self.source_embedding(input))
        # x_embed: (seq_len, batch_size, embedding_size)
        packed_embed = nn.utils.rnn.pack_padded_sequence(x_embed, lengths)

        out, (hidden, cell) = self.birnn(packed_embed)

        return hidden, cell


class LSTMDecoder(nn.Module):
    def __init__(self, output_size, embed_size, hidden_size, num_layers, bidirectional, dropout):
        super(LSTMDecoder, self).__init__()

        self.embedding = nn.Embedding(output_size, embed_size)
        self.dropout = nn.Dropout(dropout)
        self.birnn = nn.LSTM(embed_size, hidden_size, num_layers, bidirectional=bidirectional)
        self.num_directions = 2 if bidirectional else 1
        self.fc_out = nn.Linear(self.num_directions * hidden_size, output_size)
        self.output_size = output_size


    def forward(self, x_in, hidden, cell):
        x_in = x_in.unsqueeze(0)
        # x_in: (1, batch_size)
        x_embed = self.dropout(self.embedding(x_in))
        # x_embed: (1, batch_size, embedding_size)

        out, (hidden,cell) = self.birnn(x_embed, (hidden, cell))
        # out: (1, batch_size, hidden_size * num_directions)
        prediction = self.fc_out(out.squeeze(0))
        # prediction: [batch_size, output_size]
        return prediction, hidden, cell

class AttentionLSTMDecoder(nn.Module):
    def __init__(self, output_size, embed_size, hidden_size, num_layers, bidirectional, dropout):
        super(AttentionLSTMDecoder, self).__init__()

        self.embedding = nn.Embedding(self.output_size, embed_size)
        self.dropout = nn.Dropout(dropout)
        self.num_directions = 2 if bidirectional else 1
        self.attention = nn.Linear(
            self.hidden_size * self.num_directions,
            max_length)
        self.attention_combine = nn.Linear(
            self.hidden_size * self.num_directions,
            self.hidden_size)

    def forward(self, y_in, hidden, cell, encoder_hidden_states):
        y_in = y_in.unsqueeze(0)
        y_embed = self.dropout(self.embed(y_in))

        attn_weights = F.softmax(self.attention((y_embed[0], hidden[0], 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0), encoder_hidden_states.unsqueeze(0))

class GRUEncoder(nn.Module):
    def __init__(self, input_size, embed_size, hidden_size, num_layers, bidirectional, dropout):
        super(GRUEncoder, self).__init__()
        self.embed = nn.Embedding(input_size, embed_size)
        self.dropout = nn.Dropout(dropout)
        self.rnn = nn.GRU(embed_size, hidden_size, num_layers, bidirectional=bidirectional)

    def forward(self, x, lengths):
        embedded = self.dropout(self.embed(x))
        packed_embed = nn.utils.rnn.pack_padded_sequence(embedded, lengths)

        output, hidden = self.rnn(embedded)

        return output, hidden

class GRUDecoder(nn.Module):
    def __init__(self, output_size, embed_size, hidden_size, num_layers, bidirectional, dropout):
        super(GRUDecoder, self).__init__()
        self.embedding = nn.Embedding(output_size, embed_size)
        self.dropout = nn.Dropout(dropout)
        self.rnn = nn.GRU(embed_size, hidden_size, num_layers, bidirectional=bidirectional)
        self.num_directions = 2 if bidirectional else 1
        self.fc_out = nn.Linear(self.num_directions * hidden_size, output_size)
        self.output_size = output_size

    def forward(self, prev_token, hidden):
        embedded = self.dropout(self.embedding(prev_token.unsqueeze(0)))

        output, hidden = self.rnn(embedded, hidden)

        pred = self.fc_out(output.squeeze(0))
        return pred, hidden


class Seq2Seq(nn.Module):
    """
    Vanilla seq2seq to handle vanilla RNN models and GRU.

    Implements greedy decoding
    """
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, source, lengths, target, teacher_forcing_ratio=0.5):
        pred, hidden = self.encoder(source, lengths)
        max_length = source.shape[0]
        batch_size = source.shape[1]
        outputs = torch.zeros(max_length, batch_size, self.decoder.output_size).to(self.device)
        
        decoder_input = target[0, :]
        decoder_hidden = hidden

        for t in range(max_length):
            output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            outputs[t] = output

            teacher_force = random.random() < teacher_forcing_ratio
            decoder_input =  target[t] if teacher_force else output.argmax(1)

        return outputs

class LSTMSeq2Seq(nn.Module):
    """
    Special seq2seq model to handle LSTM models which have a cell state
    """
    def __init__(self, encoder, decoder, device):
        super(LSTMSeq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, source, lengths, target, teacher_forcing_ratio=0.5):
        hidden, cell = self.encoder(source, lengths)
        max_length = source.shape[0]
        batch_size = source.shape[1]
        outputs = torch.zeros(max_length, batch_size, self.decoder.output_size).to(self.device)
        
        decoder_input = target[0, :]
        decoder_hidden = hidden

        for t in range(max_length):
            output, hidden, cell = self.decoder(decoder_input, decoder_hidden, cell)
            outputs[t] = output

            teacher_force = random.random() < teacher_forcing_ratio
            decoder_input =  target[t] if teacher_force else output.argmax(1)

        return outputs