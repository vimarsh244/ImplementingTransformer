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
    
class ResidualConnection(nn.Module):
    
    def __init__(self, d_model: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()
        
    def forward(self,x, sublayer):
        # x is of shape batch_size * seq_length * d_model
        # sublayer is a function that is passed to the ResidualConnection
        # sublayer is either MultiHeadAttention or FeedForwardBlock
        # we are adding the output of the sublayer to the input of the ResidualConnection
        # we are normalizing the output of the sublayer
        
        return x + self.dropout(sublayer(self.norm(x)))
    
class EncoderBlock(nn.Module):
    
    def __init__(self, self_attention_block: MultiHeadAttention, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        # self.residual_connection1 = ResidualConnection(self_attention_block.d_model, dropout)
        # self.residual_connection2 = ResidualConnection(feed_forward_block.d_model, dropout)
        # does the same thing though :/
        self.residual_connection = nn.ModuleList(ResidualConnection(dropout) for _ in range(2))
        
    def forward(self, x, src_mask):
        
        x = self.residual_connection[0](x, lambda x: self.self_attention_block(x, x, x, src_mask)) # query, key value is x itself
        x = self.residual_connection[1](x, lambda x: self.feed_forward_block) 
        
        # We have two skip connections here, which are then repeated N times in the encoder
        return x

class Encoder(nn.Module):
    def __init__(self, layers: nn.Module):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization() # normalizing the features
    
    def forward(self, x, src_mask):
        for layer in self.layers:
            x = layer(x, src_mask)
        return self.norm(x)

        
class DecoderBlock(nn.Module):
    def __init__(self, self_attention_block: MultiHeadAttention, cross_attention_block: MultiHeadAttention, feed_forward_block: FeedForwardBlock, dropout: float):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        
        self.residual_connection = nn.ModuleList(ResidualConnection(dropout) for _ in range(3))
        
    def forward(self, x, encoder_output, src_mask, tgt_mask):
        # because we are doing translation task here; we have one which comes from encoder (source sentence) and one which comes from decoder
        
        x = self.residual_connection[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))
        x = self.residual_connection[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        x = self.residual_connection[2](x, lambda x: self.feed_forward_block)
        
        return x

class Decoder(nn.Module):
    def __init__(self, layers: nn.Module):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()
    
    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)
    # just like we did for encoder, doing it for N times
    
class ProjectionLayer(nn.Module):
    # convert seq, d_model to seq, vocab_size
    
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        
        self.project = nn.Linear(d_model, vocab_size)
    
    def forward(self, x):
        # x is of shape batch_size * seq_length * d_model
        return torch.log_softmax(self.project(x))  # for numerical stability
    
class Transformer(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, src_embedding: InputEmbeddings, tgt_embedding: InputEmbeddings, src_positional_encoding: PositionalEncoding, tgt_positional_encoding: PositionalEncoding, projection_layer: ProjectionLayer):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embedding = src_embedding
        self.tgt_embedding = tgt_embedding
        self.src_positional_encoding = src_positional_encoding
        self.tgt_positional_encoding = tgt_positional_encoding
        self.projection_layer = projection_layer
        
    # using different methods, so we can reuse the enocded values during inference
    
    def encode(self, src, src_mask):
        src = self.src_embedding(src)
        src = self.src_positional_encoding(src)
        return self.encoder(src, src_mask)
    
    def decode(self, encoder_output, src_mask, tgt, tgt_mask):
        tgt = self.tgt_embedding(tgt)
        tgt = self.tgt_positional_encoding(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)
    
    
    def project(self, decoder_output):
        return self.projection_layer(decoder_output)
    
def build_transformer(src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int, tgt_seq_len: int, d_model: int = 512, N: int = 6, num_heads: int = 8, dropout: float = 0.1, dff: int = 2048):
    
    # embedding layers
    
    src_embedding = InputEmbeddings(d_model, src_vocab_size)
    tgt_embedding = InputEmbeddings(d_model, tgt_vocab_size)
    
    # positional Encoding

    src_positional_encoding = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_positional_encoding = PositionalEncoding(d_model, tgt_seq_len, dropout)
    
    # encoder block
    
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention = MultiHeadAttention(d_model=d_model, num_heads=num_heads, dropout=dropout)
        encoder_feed_forward = FeedForwardBlock(d_model=d_model, dff=dff, dropout=dropout)
        
        encoder_block = EncoderBlock(encoder_self_attention, encoder_feed_forward, dropout)
        
        encoder_blocks.append(encoder_block)
    
    # decoder block
    
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention = MultiHeadAttention(d_model=d_model, num_heads=num_heads, dropout=dropout)
        decoder_cross_attention = MultiHeadAttention(d_model=d_model, num_heads=num_heads, dropout=dropout)
        decoder_feed_forward = FeedForwardBlock(d_model=d_model, dff=dff, dropout=dropout)
        
        decoder_block = DecoderBlock(decoder_self_attention, decoder_cross_attention, decoder_feed_forward, dropout)
        
        decoder_blocks.append(decoder_block)
    
    # combinding encoder and decoder
    
    encoder = Encoder(nn.ModuleList(encoder_blocks))
    decoder = Decoder(nn.ModuleList(decoder_blocks))
    
    # projection layer
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)
    
    
    # finnally the transformer
    transformer = Transformer(encoder, decoder, src_embedding, tgt_embedding, src_positional_encoding, tgt_positional_encoding, projection_layer)
    
    
    # initializing the params
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p) 
            # initializing the weights with xavier uniform distribution
            # xavier uniform distribution takes into account the number of input and output neurons and then initializes the weights
 
    return transformer

    