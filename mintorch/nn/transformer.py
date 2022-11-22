

@dataclass(frozen=True)
class TransformerConfig:
    '''Constants used throughout your decoder-only transformer model.'''

    num_layers: int = 6
    num_heads: int = 8
    vocab_size: int = 256
    hidden_size: int = 512
    max_seq_len: int = 512
    dropout: float = 0.1
    layer_norm_epsilon: float = 1e-05

config = TransformerConfig()



# %%

class MLPBlock(nn.Module):

    def __init__(self, hidden_size: int, dropout: float):
        self.hidden_size = hidden_size

        super().__init__()

        self.linear1 = nn.Linear(hidden_size, hidden_size * 4)
        self.gelu = nn.GELU()
        self.linear2 = nn.Linear(hidden_size * 4, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: t.Tensor) -> t.Tensor:
        x = self.linear1(x)
        x = self.gelu(x)
        x = self.linear2(x)
        x = self.dropout(x)

        return x


class DecoderBlock(nn.Module):

    def __init__(self, hidden_size: int, num_heads: int, layer_norm_epsilon: float, dropout: float):
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.layer_norm_epsilon = layer_norm_epsilon
        self.dropout = dropout

        super().__init__()

        self.attention = MultiheadMaskedAttention(hidden_size, num_heads)
        self.ln1 = nn.LayerNorm(hidden_size, eps=layer_norm_epsilon)
        self.mlp = MLPBlock(hidden_size, dropout)
        self.ln2 = nn.LayerNorm(hidden_size, eps=layer_norm_epsilon)

    def forward(self, x: t.Tensor) -> t.Tensor:
        x = x + self.attention(self.ln1(x))
        x = x + self.mlp(self.ln2(x))

        return x


class DecoderOnlyTransformer(nn.Module):

    def __init__(self, config: TransformerConfig):
        self.config = config

        super().__init__()

        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.positional_embedding = SinusoidalPositionalEncoding(config.hidden_size, config.max_seq_len)

        self.dropout = nn.Dropout(config.dropout)
        self.decoder_blocks = nn.ModuleList([
            DecoderBlock(config.hidden_size, config.num_heads, config.layer_norm_epsilon, config.dropout)
            for _ in range(config.num_layers)
        ])
        self.ln = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.unembed = nn.Linear(config.hidden_size, config.vocab_size)        
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x: t.Tensor) -> t.Tensor:
        x = self.embedding(x)
        x = self.positional_embedding(x)
        x = self.dropout(x)

        for decoder_block in self.decoder_blocks:
            x = decoder_block(x)
        
        x = self.ln(x)
        x = self.unembed(x)
        x = self.softmax(x)

        return x


# %%
