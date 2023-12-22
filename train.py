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
    
class MultiHeadAttention(nn.Module):
    # Input is copied into 3 matrices (Query, Key, Value) and then we apply attention to them
    # Q* Wq = Q' then we split it into h different heads
    # similar we do we K and V
    # attention is calculated for each head by softmax(Q' * K' / sqrt(d_k)) * V
    # d_k = d_model / h
    # Then we concatenate the heads and multiply it by Wo
    
    def __init__(self, d_model: int, num_heads: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        # h has to be a factor of d_model
        self.num_heads = num_heads
        
        self.dropout = nn.Dropout(dropout)
        
        assert d_model % num_heads == 0, "d_model should be multiple of h"
        
              
        self.d_k = d_model // num_heads
        
        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        
        self.dv = self.d_k
        self.Wo = nn.Linear(self.num_heads * self.dv, self.d_model)
    
    @staticmethod
    def calculate_attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1] #getting the last dimension of the query matrix
        
        attention_scores = query.matmul(key.transpose(-2, -1)) / math.sqrt(d_k) # shape: batch_size * num_heads * seq_length * seq_length
        # now applying mask to the attention scores
        
        if(mask) is not None:
            attention_scores.masked_fill_(mask == 0, -1e9) # we are replacing the values of the mask with -1e9
        
        attention_scores = attention_scores.softmax(dim = -1)
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        
        return attention_scores.matmul(value), attention_scores
          
    def forward(self, q,k,v,mask):
        # We sometimes want to mask some of the values: i.e., dont want some words to interact with other words
        # we do that by replacing attention scores of those words with very small value (i.e., -inf)
        
        query = self.Wq(q) # Batch_size * seq_length * d_model --> Batch_size * seq_length * d_model
        key = self.Wk(k)
        value = self.Wv(v)
        
        query= query.view(query.shape[0], query.shape[1], self.num_heads, self.d_k).transpose(1,2) # Batch_size * seq_length * d_model --> Bath_size * Seq_length * heads * d_k --> Batch_size * num_heads * seq_length * d_k
        # We are reshaping the query matrix by adding new dimension i.e., heads and d_k
        # we then swap the columns and numbver of heads dimentions
        
        value = value.view(value.shape[0], value.shape[1], self.num_heads, self.d_k).transpose(1,2)
        key = key.view(key.shape[0], key.shape[1], self.num_heads, self.d_k).transpose(1,2)
        
        x, attension_scores = self.calculate_attention(query, key, value, mask, self.dropout)
        
        # Batch, h, seq_len, d_k --> batch, seq_len, h, d_k --> batch, seq_len, d_model
        x = x.transpose(1,2).continguous()
        # tell pytorch to concate
        x = x.view(x.shape[0], -1, self.num_heads * self.d_k)
        
        return self.Wo(x), attension_scores