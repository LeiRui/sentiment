# -*- coding: utf-8 -*-
"""
Created on Wed May 22 14:18:43 2019

@author: admin
"""
import torch
import torch.nn as nn

class SentimentRNN(nn.Module):
    """
    The RNN model that will be used to perform Sentiment analysis.
    """

    def __init__(self, vocab_size, output_size, embedding_dim, hidden_dim, n_layers, drop_prob=0.5):
        """
        Initialize the model by setting up the layers.
        """
        super(SentimentRNN, self).__init__()

        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        
        # embedding and LSTM layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim) # 其实是一个shape(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, 
                            dropout=drop_prob, batch_first=True)
        # https://pytorch.org/docs/stable/nn.html
        # batch_first: If True, then the input and output tensors are provided as (batch, seq, feature).❤
        #          Default: False. Default input and output tensors are (seq, batch, feature). ❤
        # https://gist.github.com/purelyvivid/dac6ddcf2dae46e6bdcc49175fe8296e ❤
        # 輸入張量的維度是(seq_len, batch, input_size)，其中input_size是前面提過輸入的特徵數目，例如詞向量的維度。
        # batch就是batch_size批量大小無須解釋。seq_len是序列長度，指的是time stamp的數目，或者說是一個句子裡有幾個詞（一個句子的最大詞數）。
        # 另外值得一提的是建立LSTM時batch_first這個參數如果設為True，代表輸入向量的維度順序不是(seq_len, batch, input_size)，
        # 而是(batch, seq_len, input_size)，這樣的配置比較常用。
        
        # dropout: If non-zero, introduces a Dropout layer on the outputs of each LSTM layer except the last layer, 
        #          with dropout probability equal to dropout. Default: 0
        
        # dropout layer
        self.dropout = nn.Dropout(0.3)
        
        # linear and sigmoid layers
        self.fc = nn.Linear(hidden_dim, output_size)
        self.sig = nn.Sigmoid()
        

    def forward(self, x, hidden):
        """
        Perform a forward pass of our model on some input and hidden state.
        """
        batch_size = x.size(0) # Here batch_first=True, so the input and output tensors are provided as (batch, seq, feature).❤
        # print("forwarding this batch...batch_size:{}".format(batch_size))

        # embeddings and lstm_out
        x = x.long() # Convert IntTensor to LongTensor
        embeds = self.embedding(x) # batch*seq_len*hidden_dim
        lstm_out, hidden = self.lstm(embeds, hidden) # hidden shape (num_layers * num_directions, batch, hidden_size)
    
        # stack up lstm outputs
        # output of shape (batch, seq_len, num_directions * hidden_size) (batch first)
        # torch.Size([50, 200(这个就是预处理中把每句review都变成固定长度200个words), 256]) 
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim) # contiguous & view搭配使用
        # torch.Size([10000, 256]) 10000=50*200=batch*seq_len
        
        # dropout and fully-connected layer
        out = self.dropout(lstm_out)
        out = self.fc(out) # nn.Linear(hidden_dim, output_size)
        # sigmoid function
        sig_out = self.sig(out) # torch.Size([10000, 1])
        
        # reshape to be batch_size first
        # torch.Size([10000, 1])
        sig_out = sig_out.view(batch_size, -1)
        # torch.Size([50, 200])
        sig_out = sig_out[:, -1] # (get last batch of labels)
        # MR: sig_out shape (batch, seq_len)，所以sig_out[:, -1]是取所有batch的各自t=seq_len（也就是序列最后一个输入）的输出
        # many to one
        # torch.Size([batch])
        
        # return last sigmoid output and hidden state
        return sig_out, hidden
    
    
    def init_hidden(self, batch_size):
        ''' Initializes hidden state '''
        # Create two new tensors with sizes n_layers x batch_size x hidden_dim,
        # initialized to zero, for hidden state and cell state of LSTM
        
        # MR: https://pytorch.org/docs/stable/nn.html
        # 可是If (h_0, c_0) is not provided, both h_0 and c_0 default to zero.这里提供的不就是全0的(h_0,c_0)
        
        # hidden shape (num_layers * num_directions, batch, hidden_size)
        
        weight = next(self.parameters()).data
        
        train_on_gpu=torch.cuda.is_available()
        if (train_on_gpu):
            hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda(),
                  weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda())
        else:
            hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_(),
                      weight.new(self.n_layers, batch_size, self.hidden_dim).zero_())
        
        return hidden