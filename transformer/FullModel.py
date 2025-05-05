from Attention import MultiHeadAttention
from PositionEncoder import PositionalEncoding, PositionwiseFeedForward, Embeddings
from EncoderDecoder import EncoderDecoder, Encoder, Decoder, Generator, EncoderLayer, DecoderLayer
import torch.nn as nn
import copy

def make_model(
    src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1
    ):
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    attn = MultiHeadAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N), 
        #return self.encoder(self.src_embed(src), src_mask)
        
        #encoder = encoder_layer * N
        #encoder_layer
        #x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))   # self-attention
        #return self.sublayer[1](x, self.feed_forward)  # residual connection

        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        #return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)
        #decoder = decoder_layer * N
        #decoder_layer
        #x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))   # self-attention
        #x = self.sublayer[1](x, lambda x: self.src_attn(x, memory, memory, src_mask))   # source-attention
        #return self.sublayer[2](x, self.feed_forward)  # residual connection


        nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
        #return src_embed(src)

        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        #return target_embed(tgt)

        Generator(d_model, tgt_vocab)
    )

    # This was important from their code.
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model
