from io import open
import random
import time
import math
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
from torch import optim
import torch.nn.functional as F
from helper import *

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=1):
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.embedding.weight.data.copy_(torch.eye(hidden_size))
        self.embedding.weight.requires_grad = False
        
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, batch_size, hidden):
        embedded = self.embedding(input).view(1, batch_size, self.hidden_size)
        output = embedded
        for i in range(self.n_layers):
            output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self, batch_size):
        result = Variable(torch.zeros(1, batch_size, self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result

class DecoderRNN(nn.Module):
    
    def __init__(self, hidden_size, output_size, n_layers=1):
        super(DecoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.embedding.weight.data.copy_(torch.eye(hidden_size))
        self.embedding.weight.requires_grad = False
        
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, batch_size, hidden):
        output = self.embedding(input).view(1, batch_size, self.hidden_size)
        for i in range(self.n_layers):
            output = F.relu(output)
            output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self, batch_size):
        result = Variable(torch.zeros(1, batch_size, self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result

def train(input_variable, target_variable, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length):
    
    batch_size = input_variable.size()[0]
    encoder_hidden = encoder.initHidden(batch_size)

    input_variable = Variable(input_variable.transpose(0, 1))
    target_variable = Variable(target_variable.transpose(0, 1))

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_variable.size()[0]
    target_length = target_variable.size()[0]

    encoder_outputs = Variable(torch.zeros(max_length, batch_size, encoder.hidden_size))
    encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs

    loss = 0
    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_variable[ei], batch_size, encoder_hidden)
        encoder_outputs[ei] = encoder_output[0]

    decoder_input = Variable(torch.LongTensor([SOS_token] * batch_size))
    decoder_input = decoder_input.cuda() if use_cuda else decoder_input

    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
    # use_teacher_forcing = True

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden= decoder(
                decoder_input, batch_size, decoder_hidden)
            loss += criterion(decoder_output, target_variable[di])
            decoder_input = target_variable[di]  # Teacher forcing
            
    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden= decoder(
                decoder_input, batch_size, decoder_hidden)

            topv, topi = decoder_output.data.topk(1)
            decoder_input = Variable(torch.cat(topi)) 
            # decoder_input = Variable(torch.LongTensor([[ni]]))
            decoder_input = decoder_input.cuda() if use_cuda else decoder_input

            loss += criterion(decoder_output, target_variable[di])

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.data[0] / target_length

def trainIters(encoder, decoder, epochs, train_loader, test_loader, max_length, learning_rate=0.01):
    start = time.time()

    encoder_optimizer = optim.SGD(filter(lambda x: x.requires_grad, encoder.parameters()),
                                  lr=learning_rate)
    decoder_optimizer = optim.SGD(filter(lambda x: x.requires_grad, decoder.parameters()),
                                  lr=learning_rate)

    # data loader
    criterion = nn.NLLLoss()

    for epoch in range(epochs):
        print_loss_total = 0
        for batch_x, batch_y in train_loader:
            loss = train(batch_x, batch_y, encoder,
                         decoder, encoder_optimizer, decoder_optimizer, criterion, max_length)            
            print_loss_total += loss

        print('epochs: '+str(epoch))
        print('total loss: '+str(print_loss_total))
        evaluate(encoder, decoder, test_loader, max_length)
        print()

def evaluate(encoder, decoder, loader, max_length):

    total = 0
    correct = 0
    for batch_x, batch_y in loader:

        batch_size = batch_x.size()[0]
        encoder_hidden = encoder.initHidden(batch_size)

        input_variable = Variable(batch_x.transpose(0, 1))
        target_variable = Variable(batch_y.transpose(0, 1))

        input_length = input_variable.size()[0]
        target_length = target_variable.size()[0]

        output = torch.LongTensor(target_length, batch_size)

        encoder_outputs = Variable(torch.zeros(max_length, batch_size, encoder.hidden_size))
        encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs
        
        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(
                input_variable[ei], batch_size, encoder_hidden)
            encoder_outputs[ei] = encoder_output[0]

        decoder_input = Variable(torch.LongTensor([SOS_token] * batch_size))
        decoder_input = decoder_input.cuda() if use_cuda else decoder_input
        decoder_hidden = encoder_hidden

        for di in range(target_length):
            decoder_output, decoder_hidden = decoder(
                decoder_input, batch_size, decoder_hidden)
            topv, topi = decoder_output.data.topk(1)

            output[di] = torch.cat(topi)
            decoder_input = torch.cat(topi) 

        output = output.transpose(0,1)
        for di in range(output.size()[0]):
            ignore = [SOS_token, EOS_token, PAD_token]
            sent = [word for word in output[di] if word not in ignore]
            y = [word for word in batch_y[di] if word not in ignore]
            if sent == y:
                correct += 1
            total += 1
    print('accuracy '+str(correct/total))


use_cuda = torch.cuda.is_available()
SOS_token = 0
EOS_token = 1
PAD_token = 2
MAX_LENGTH = 15
teacher_forcing_ratio = 0.5
hidden_size = 29 #a-z+SOS+EOS+PAD
batch_size = 128
epochs = 100

input_lang, output_lang, pairs = prepareData('eng', 'fra', True)
pairs = variablesFromPairs(input_lang, output_lang, pairs, MAX_LENGTH)
train_num = int(len(pairs)*0.9)
train_loader = torch.utils.data.DataLoader(pairs[:train_num], 
    batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(pairs[train_num:], 
    batch_size=batch_size, shuffle=True)
print(input_lang.word2index)

encoder1 = EncoderRNN(input_lang.n_words, hidden_size)
decoder1 = DecoderRNN(hidden_size, output_lang.n_words)
if use_cuda:
    encoder1 = encoder1.cuda()
    decoder1 = attn_decoder1.cuda()
print('Training starts.')
trainIters(encoder1, decoder1, epochs, train_loader, test_loader, MAX_LENGTH)
