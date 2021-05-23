import torch
import numpy as np
from torch import nn
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from no_toxicity.src.preprocessing.cleaning import Dfcleaner

import sys
sys.path.append("./")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def embedding_weight(vocab_size, embd_dim, token_to_idx):

    f = open("./glove.6B.100d.txt", encoding="utf8")
    embedding_index = {}
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype="double")
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
    embd_layer = nn.Embedding(num_embeddings, embedding_dim).to(device)

    embd_layer.weight=torch.nn.Parameter(torch.from_numpy(weight_matrix))
    if non_trainable:
        embd_layer.weight.requires_grad = False

    return embd_layer


class MyModel(nn.Module):
    def __init__(self, input_vocab_size, emd_dim, hidden_size, batch_size, n_layers, token_to_idx):
        super().__init__()

        self.vocab_size = input_vocab_size
        self.embd_dim = emd_dim
        self.token_to_idx = token_to_idx
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.batch_size=batch_size
        self.embedding_layer = create_emb_layer(
            self.vocab_size, self.embd_dim, self.token_to_idx
        )

        self.lstm = nn.LSTM(
            self.embd_dim, self.hidden_size, num_layers=self.n_layers, batch_first=True
        )
        self.drop = nn.Dropout(p=0.5)
        self.fc = nn.Linear(self.hidden_size, 7)

    def __loss_fun(self, outputs, targets):
        return nn.BCEWithLogitsLoss()(outputs[:,-1], targets[:,-1])

    def __model_acc(self, outputs, targets):
        outputs = torch.sigmoid(outputs).cpu().detach().numpy()
        outputs = np.round(outputs)
        targets = targets.cpu().detach().numpy()
        return {"accuracy": accuracy_score(targets[:,-1], outputs[:,-1])}

    def forward(self, inputs, targets, lengths):
        batch_size = inputs.shape[0]
        if batch_size != self.batch_size:
            self.batch_size = batch_size
        inputs = self.embedding_layer(inputs)

        inputs = pack_padded_sequence(inputs.float(), lengths=lengths.cpu(), batch_first=True)

        outputs, _ = self.lstm(inputs, (self.init_hidden(), self.init_hidden()))
        outputs, _ = pad_packed_sequence(outputs,batch_first=True)
       
        outputs = self.fc(outputs[:, -1, :].squeeze())
        
        outputs = self.drop(outputs)
        
        loss = self.__loss_fun(outputs, targets)
        metric_ = self.__model_acc(outputs, targets)
        return loss,metric_

    def init_hidden(self):
        # Randomly initialize the weights of the RNN
        return torch.randn(self.n_layers, self.batch_size, self.hidden_size).to(device)
    def init_pre_hidden(self):
        # Randomly initialize the weights of the RNN
        return torch.randn(self.n_layers, 1, self.hidden_size).to(device)
    
    def predict(self, inputs):
        batch_size = inputs.shape[0]
        if batch_size != self.batch_size:
            self.batch_size = batch_size
        inputs = self.embedding_layer(inputs)

        outputs, _ = self.lstm(inputs.float(), (self.init_pre_hidden(), self.init_pre_hidden()))
 
        
        outputs = self.fc(outputs[:, -1, :].squeeze())
        outputs = self.drop(outputs)  
       
        outputs = torch.sigmoid(outputs).cpu().detach().numpy()
        outputs = np.round(outputs)
    
        prediction=[]
        if int(outputs[-1])==1:
            prediction.append('not_toxic')
        else:
            for idx,i in enumerate(outputs[:-1]):
                if idx==0 and int(i)==1:
                    prediction.append('toxic')
                elif idx==1 and int(i)==1:
                    prediction.append('severe_toxic')
                elif idx==2 and int(i)==1:
                    prediction.append('obscene')
                elif idx==3 and int(i)==1:
                    prediction.append('threat')
                elif idx==4 and int(i)==1:
                    prediction.append('insult')
                elif idx==5 and int(i)==1:
                    prediction.append('identity_hate')
        return prediction


    