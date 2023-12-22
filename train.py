import torch
import torch.nn as nn
import math

class InputEmbeddings(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model) #just a lookup table that stores embeddings of a fixed dictionary and size
    
    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)
        
class PositionalEncoding(nn.Module):
    # We encode the position of each word within the sentence
    def __init__(self, d_model: int, seq_length: int, droput : float):
        super().__init__()
        self.d_model = d_model
        self.seq_length = seq_length
        self.dropout = nn.Dropout(droput)  # to randomly 0 some elements of the input tensor by bernoulli distribution
        
        # Creating matrix which is seq_length * d_model
        pe = torch.zeros(seq_length, d_model)
        
        position = torch.arrange(0, seq_length, dtype=torch.float).unsqueeze(1)  # shape: seq_length * 1
        
        #calculating the pos here in log space for numerical stability (i.e., to prevent any small values)
        div_term = torch.exp(torch.arrange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)) # shape: d_model/2
        
        # the even positions will be sin^pos and the odd positions will be cos^pos
        ## think of why we use this
        
        ## Applying sin to even and cos to odd positions
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # adding a batch dimension to the positional encoding
        # we can do train it together, batch it now
        
        pe = pe.unsqueeze(0) # shape: 1 * seq_length * d_model
        
        self.register_buffer('pe', pe) # register_buffer is used to register some tensors that are not trainable
        
    def forward(self, x):
        # x is of shape batch_size * seq_length * d_model
        x = x + self.pe[:, :x.size(1)].requires_grad_(False)
        
        # x is 0s inititally | # we are adding positional encoding to the input embeddings
        # we are not training the positional encoding
        
        return self.dropout(x)
    
class LayerNormalization(nn.Module):
    # If we have batch of n items, we will normalize each item separately of each other
    # We also add two parameters (a multiplicative parameter and an additive parameter) to each item
    
    # model will amplofy the values it wants to amplify and suppress the values it wants to suppress
    # we are normalizing each item separately of each other
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        # we need epsilon to prevent dvision by zero which could happen if the variance is zero (close to)
        
        self.d_model = d_model
        self.eps = eps
        
        self.alpha = nn.Parameter(torch.ones(d_model)) # multiplicative parameter
        self.bias = nn.Parameter(torch.zeros(d_model)) # additive parameter
        
    def forward(self, x):
        # x is of shape batch_size * seq_length * d_model
        mean = x.mean(dim = -1, keepdim=True) # shape: batch_size * seq_length * 1
        std = x.std(dim = -1, keepdim=True)   # shape: batch_size * seq_length * 1
        
        return self.alpha * (x - mean) / (std + self.eps) + self.bias
    
class FeedForwardBlock(nn.Module):
    # FFN (x) = max(0, xW1 + b1)W2 + b2
    # d_model = 512, dff = 2048
    
    def __init__(self, d_model: int, dff: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.dff = dff
        self.dropout = nn.Dropout(dropout)
        
        self.linear1 = nn.Linear(d_model, dff) # W1 and b1
        
        self.linear2 = nn.Linear(dff, d_model) # W2 and b2
        
    def forward(self, x):
        # x is of shape batch_size * seq_length * d_model
        x = self.linear1(x) # shape: (batch_size) * seq_length * dff
        x = nn.ReLU(x) # if x is negative, we will make it 0
        x = self.dropout(x)
        x = self.linear2(x) # shape: (batch_size) * seq_length * d_model
        return x
    

        

        