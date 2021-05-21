import torch
import numpy as np
from torch import nn
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import sys

sys.path.append("./")
import os

def embedding_weight(vocab_size, embd_dim, token_to_idx):
    print(os.getcwd())
    f = open("./glove.6B.100d.txt", encoding="utf8")
    embedding_index = {}
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype="float")
        embedding_index[word] = coefs
    f.close()

    embedding_output = np.zeros((vocab_size, embd_dim))

    for word, idx in token_to_idx.items():
        embedding_vector = embedding_index.get(word)

        if embedding_vector is not None:
            embedding_output[idx] = embedding_vector

    return embedding_output


def create_emb_layer(vocab_size, embd_dim, token_to_idx, non_trainable=True):

    weight_matrix = embedding_weight(vocab_size, embd_dim, token_to_idx)
  
    num_embeddings, embedding_dim = weight_matrix.shape[0],weight_matrix.shape[1]
    embd_layer = nn.Embedding(num_embeddings, embedding_dim)
    
    embd_layer.weight=torch.nn.Parameter(torch.from_numpy(weight_matrix))
    if non_trainable:
        embd_layer.weight.requires_grad = False

    return embd_layer


class LSTM(nn.Module):
    def __init__(self, input_vocab_size, emd_dim, hidden_size, n_layers, token_to_idx):
        super().__init__()

        self.vocab_size = input_vocab_size
        self.embd_dim = emd_dim
        self.token_to_idx = token_to_idx
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        # self.embedding_layer=nn.Embedding(self.vocab_size,self.embd_dim)
        self.embedding_layer = create_emb_layer(
            self.vocab_size, self.embd_dim, self.token_to_idx
        )
        self.lstm = nn.LSTM(
            self.embd_dim, self.hidden_size, num_layers=self.n_layers, batch_first=True
        )
        self.drop = nn.Dropout(p=0.5)
        self.fc = nn.Linear(self.hidden_size, 7)

    def __loss_fun(self, outputs, targets):
        return nn.BCEWithLogitsLoss()(outputs, targets)

    def __model_acc(self, outputs, targets):
        outputs = torch.sigmoid(outputs).cpu().detach().numpy()
        outputs = np.round(outputs)
        targets = targets.cpu().detach().numpy()
        return {"accuracy": accuracy_score(targets, outputs)}

    def forward(self, inputs, targets, lengths):

        x = self.embedding_layer(inputs)

        x = pack_padded_sequence(x, lengths=lengths.cpu(), batch_first=True)
        output, _ = self.lstm(x, (self.init_hidden(), self.init_hidden()))
        output, _ = pad_packed_sequence(output)

        output = self.fc(output[:, -1, :].squeeze())
        output=self.drop(output)

        loss = self.__loss_fun(output, targets)
        met = self.__model_acc(output, targets)
        return loss, met

    def init_hidden(self):
        # Randomly initialize the weights of the RNN
        return torch.randn(1, self.batch_size, self.decoder_hidden_size).to(device)

    # class Decoder(nn.Module):
    #     def __init__(
    #         self,
    #         vocab_size,
    #         embedding_dim,
    #         decoder_hidden_size,
    #         encoder_hidden_size,
    #         batch_size,
    #     ):
    #         super(Decoder, self).__init__()
    #         self.batch_size = batch_size
    #         self.encoder_hidden_size = encoder_hidden_size
    #         self.decoder_hidden_size = decoder_hidden_size
    #         self.vocab_size = vocab_size
    #         self.embedding_dim = embedding_dim
    #         self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
    #         self.gru = nn.GRU(
    #             self.embedding_dim + self.encoder_hidden_size,
    #             self.decoder_hidden_size,
    #             batch_first=True,
    #         )
    #         self.fc = nn.Linear(self.encoder_hidden_size, self.vocab_size)

    #         # Attention weights
    #         self.W1 = nn.Linear(self.encoder_hidden_size, self.decoder_hidden_size)
    #         self.W2 = nn.Linear(self.encoder_hidden_size, self.decoder_hidden_size)
    #         self.V = nn.Linear(self.encoder_hidden_size, 1)

    # def forward(self, targets, hidden, encoder_output):
    #         self.batch_size = targets.size(0)

    #         # Switch the dimensions of sequence_length and batch_size
    #         encoder_output = encoder_output.permute(1, 0, 2)

    #         # Add an extra axis for a time dimension
    #         hidden_with_time_axis = hidden.permute(1, 0, 2)

    #         # Attention score (Bahdanaus)
    #         score = torch.tanh(self.W1(encoder_output) + self.W2(hidden_with_time_axis))

    #         # Attention weights
    #         attention_weights = torch.softmax(self.V(score), dim=1)

    #         # Find the context vectors
    #         context_vector = attention_weights * encoder_output
    #         context_vector = torch.sum(context_vector, dim=1)

    #         # Turn target indices into distributed embeddings
    #         x = self.embedding(targets)

    #         # Add the context representation to the target embeddings
    #         x = torch.cat((context_vector.unsqueeze(1), x), -1)

    #         # Apply the RNN
    #         output, state = self.gru(x, self.init_hidden())

    #         # Reshape the hidden states (output)
    #         output = output.view(-1, output.size(2))

    #         # Apply a linear layer
    #         x = self.fc(output)

    #         return x, state, attention_weights

    def init_hidden(self):
        # Randomly initialize the weights of the RNN
        return torch.randn(1, self.batch_size, self.decoder_hidden_size).to(device)
