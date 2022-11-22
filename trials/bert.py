# %%
from dataclasses import dataclass
from typing import Optional

import einops
import numpy as np
import pandas as pd
import torch as t
import torch.nn as nn
import transformers
##
from arena.w2d2 import utils
from fancy_einsum import einsum
from IPython.display import display
from torchtyping import TensorType

# %%
tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-cased")
bert = transformers.BertForMaskedLM.from_pretrained("bert-base-cased")

bert
# %%

@dataclass
class BERTConfig:
    max_seq_len = 512
    hidden_size = 768
    attention_dim = 64
    num_heads = 12
    num_layers = 12
    layer_norm_epsilon = 1e-12
    dropout = 0.1
    vocab_size = 28996

class MultiheadAttention(nn.Module):

    def __init__(self, config: BERTConfig):
        self.config = config
        
        super().__init__()
            
        # TODO: What to do about the fact that BERT separates the QKV weights?
        self.W_Q = nn.Linear(config.hidden_size, config.attention_dim * config.num_heads)
        self.W_K = nn.Linear(config.hidden_size, config.attention_dim * config.num_heads)
        self.W_V = nn.Linear(config.hidden_size, config.attention_dim * config.num_heads)
        self.W_O = nn.Linear(config.attention_dim * config.num_heads, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: t.Tensor, additive_attention_mask: Optional[t.Tensor] = None) -> t.Tensor:
        """
        x: shape (batch, seq, hidden_size)

        Return: shape (batch, seq, hidden_size)
        """
        Q, K, V = self.W_Q(x), self.W_K(x), self.W_V(x)

        additive_attention_mask = additive_attention_mask
        A = self.multihead_masked_attention(Q, K, mask=additive_attention_mask)
        h = self.multihead_masked_attention_head(A, V)
        output = self.W_O(h)

        return output
      
    def multihead_masked_attention(
        self,
        Q: TensorType["b", "s", "n*h"], 
        K: TensorType["b", "s", "n*h"], 
        mask: Optional[TensorType["b", "s", "s"]] = None,
    ) -> TensorType["b", "n", "s_q", "s_k"]:
        '''
        Should return the results of multihead self-attention (after softmax, before multiplying with V)
        '''
        _Q = einops.rearrange(Q, "b s (n h) -> b n s h", n=self.config.num_heads)    
        _K = einops.rearrange(K, "b s (n h) -> b n s h", n=self.config.num_heads)    

        d_head = _Q.shape[-1]

        A_pre = einsum("b n s_q h, b n s_k h -> b n s_q s_k", _Q, _K) / np.sqrt(d_head)

        if mask is not None:
            A_pre = A_pre + mask

        return t.softmax(A_pre, dim=-1)

    def multihead_masked_attention_head(
        self,
        A: TensorType["b", "n", "s_q", "s_k"], 
        V: TensorType["b", "s", "n*h"],
    ) -> TensorType["batch", "seq", "n_heads*headsize"]:
        _V = einops.rearrange(V, "b s (n h) -> b n s h", n=self.config.num_heads)
        AV: TensorType["b", "n", "s_q", "h"] = einsum("b n s_q s_k, b n s_k h -> b n s_q h", A, _V)
        return einops.rearrange(AV, "b n s h -> b s (n h)") 

class BERTMLP(nn.Module):
    def __init__(self, config: BERTConfig):
        self.config = config
        super().__init__()

        self.dense1 = nn.Linear(config.hidden_size, config.hidden_size * 4)
        self.gelu = nn.GELU()
        self.dense2 = nn.Linear(config.hidden_size * 4, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, x: t.Tensor, additive_attention_mask: Optional[t.Tensor] = None) -> t.Tensor:
        '''
        x: shape (batch, seq, hidden_size)
        additive_attention_mask: shape (batch, nheads=1, seqQ=1, seqK)
        '''
        x = self.dense1(x)
        x = self.gelu(x)
        x = self.dense2(x)
        x = self.dropout(x)

        return x

class BERTBlock(nn.Module):
    def __init__(self, config: BERTConfig):
        self.config = config
        super().__init__()
        
        self.attention = MultiheadAttention(config)
        self.dropout1 = nn.Dropout(config.dropout)
        self.ln1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.mlp = BERTMLP(config)
        self.dropout2 = nn.Dropout(config.dropout)
        self.ln2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        
    def forward(self, x: t.Tensor, additive_attention_mask: Optional[t.Tensor] = None) -> t.Tensor:
        '''
        x: shape (batch, seq, hidden_size)
        additive_attention_mask: shape (batch, nheads=1, seqQ=1, seqK)
        '''
        x = x + self.dropout1(self.attention(x, additive_attention_mask))
        x = self.ln1(x)
        x = x + self.dropout2(self.mlp(x))
        x = self.ln2(x)

        return x

class BERTEmbeddings(nn.Module):
    def __init__(self, config: BERTConfig):
        self.config = config
        super().__init__()
        
        self.token_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_seq_len, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(2, config.hidden_size)
        self.ln = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self, 
        x: t.Tensor,
        token_type_ids: Optional[t.Tensor] = None,
    ) -> t.Tensor:
        '''
        input_ids: (batch, seq) - the token ids
        token_type_ids: (batch, seq) - only used for NSP, passed to token type embedding.
        '''
        x = self.token_embeddings(x) \
            + self.position_embeddings(t.arange(x.shape[1], dtype=t.long, device=x.device)) \
        
        if token_type_ids is not None:
            x = x + self.token_type_embeddings(token_type_ids)

        x = self.ln(x)
        x = self.dropout(x)

        return x


class BERTEncoder(nn.Module):
    def __init__(self, config: BERTConfig):
        self.config = config
        super().__init__()
        
        self.blocks = nn.ModuleList([BERTBlock(config) for _ in range(config.num_layers)])

    def forward(self, x: t.Tensor, additive_attention_mask: Optional[t.Tensor] = None) -> t.Tensor:
        '''
        x: shape (batch, seq, hidden_size)
        additive_attention_mask: shape (batch, nheads=1, seqQ=1, seqK)
        '''
        for block in self.blocks:
            x = block(x, additive_attention_mask)

        return x

class BERTCommon(nn.Module):
    def __init__(self, config: BERTConfig ) -> None:
        self.config = config
        super().__init__()

        self.embeddings = BERTEmbeddings(config)
        self.encoder = BERTEncoder(config)

    def forward(self, x: t.Tensor, additive_attention_mask: Optional[t.Tensor] = None) -> t.Tensor:
        '''
        x: shape (batch, seq, hidden_size)
        additive_attention_mask: shape (batch, nheads=1, seqQ=1, seqK)
        '''
        x = self.embeddings(x)
        x = self.encoder(x, additive_attention_mask)

        return x

class BERTLMHead(nn.Module):
    def __init__(self, config: BERTConfig) -> None:
        self.config = config
        super().__init__()

        self.pred_bias = nn.Parameter(t.zeros(config.vocab_size))
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.ln = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.gelu = nn.GELU()

        # TODO: What is a tied unembed?

    def forward(self, x: t.Tensor) -> t.Tensor:
        '''
        x: shape (batch, seq, hidden_size)
        '''
        x = self.dense(x)
        x = self.gelu(x)
        x = self.ln(x)

        return x

class BERTLM(nn.Module):
    def __init__(self, config: BERTConfig) -> None:
        self.config = config
        super().__init__()

        self.common = BERTCommon(config)
        self.lm_head = BERTLMHead(config)

    def forward(
        self, 
        x: t.Tensor,
        one_zero_attention_mask: Optional[t.Tensor] = None,
        token_type_ids: Optional[t.Tensor] = None,
    ) -> t.Tensor:
        '''
        input_ids: (batch, seq) - the token ids
        one_zero_attention_mask: (batch, seq) - only used in training, passed to `make_additive_attention_mask` and used in the attention blocks.
        token_type_ids: (batch, seq) - only used for NSP, passed to token type embedding.
        '''
        x = self.common(x, one_zero_attention_mask)
        x = self.lm_head(x)
        x = x @ self.common.embeddings.token_embeddings.weight.T + self.lm_head.pred_bias

        return x

class BERTClassifier(nn.Module):
    def __init__(self, config: BERTConfig, num_classes: int) -> None:
        self.config = config
        super().__init__()

        self.common = BERTCommon(config)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self, 
        x: t.Tensor,
        one_zero_attention_mask: Optional[t.Tensor] = None,
        token_type_ids: Optional[t.Tensor] = None,
    ) -> t.Tensor:
        '''
        input_ids: (batch, seq) - the token ids
        one_zero_attention_mask: (batch, seq) - only used in training, passed to `make_additive_attention_mask` and used in the attention blocks.
        token_type_ids: (batch, seq) - only used for NSP, passed to token type embedding.
        '''
        x = self.common(x, one_zero_attention_mask, token_type_ids)
        x = self.dropout(x[:, 0])
        x = self.common.encoder(x)

        return x



def make_additive_attention_mask(
    one_zero_attention_mask: t.Tensor, 
    big_negative_number: float = -10_000
) -> t.Tensor:
    '''
    one_zero_attention_mask: 
        shape (batch, seq)
        Contains 1 if this is a valid token and 0 if it is a padding token.

    big_negative_number:
        Any negative number large enough in magnitude that exp(big_negative_number) is 0.0 for the floating point precision used.

    Out: 
        shape (batch, nheads=1, seqQ=1, seqK)
        Contains 0 if attention is allowed, big_negative_number if not.
    '''
    
    return ((1 - one_zero_attention_mask) * big_negative_number).unsqueeze(1).unsqueeze(2)


def test_make_additive_attention_mask(fn):
    one_zero_attention_mask = t.tensor([[1, 1, 1, 0, 0], [1, 1, 0, 0, 0]])
    additive_attention_mask = fn(one_zero_attention_mask)
    assert additive_attention_mask.shape == (2, 1, 1, 5)
    assert t.allclose(additive_attention_mask[0, 0, 0, :3], t.tensor(0))
    assert t.allclose(additive_attention_mask[1, 0, 0, :2], t.tensor(0))
    assert t.allclose(additive_attention_mask[0, 0, 0, 3:], t.tensor(-10_000))
    assert t.allclose(additive_attention_mask[1, 0, 0, 2:], t.tensor(-10_000))
    
    print("Success!")

test_make_additive_attention_mask(make_additive_attention_mask)
# %%

my_bert = BERTLM(BERTConfig())
# %%


def print_param_count(*models, display_df=True, use_state_dict=False):
    """
    display_df: bool
        If true, displays styled dataframe
        if false, returns dataframe
    use_state_dict: bool
        If true, uses model.state_dict() to construct dataframe
            This will include buffers, not just params
        If false, uses model.named_parameters() to construct dataframe
            This misses out buffers (more useful for GPT)
    """
    df_list = []
    gmap_list = []
    for i, model in enumerate(models, start=1):
        print(f"Model {i}, total params = {sum([param.numel() for name, param in model.named_parameters()])}")
        iterator = model.state_dict().items() if use_state_dict else model.named_parameters()
        df = pd.DataFrame([
            {f"name_{i}": name, f"shape_{i}": tuple(param.shape), f"num_params_{i}": param.numel()}
            for name, param in iterator
        ]) if (i == 1) else pd.DataFrame([
            {f"num_params_{i}": param.numel(), f"shape_{i}": tuple(param.shape), f"name_{i}": name}
            for name, param in iterator
        ])
        display(df)
        df_list.append(df)
        gmap_list.append(np.log(df[f"num_params_{i}"]))
    df = df_list[0] if len(df_list) == 1 else pd.concat(df_list, axis=1).fillna(0)
    for i in range(1, len(models) + 1):
        df[f"num_params_{i}"] = df[f"num_params_{i}"].astype(int)
    if len(models) > 1:
        param_counts = [df[f"num_params_{i}"].values.tolist() for i in range(1, len(models) + 1)]
        if all([param_counts[0] == param_counts[i] for i in range(1, len(param_counts))]):
            print("All parameter counts match!")
        else:
            print("Parameter counts don't match up exactly.")
    if display_df:
        s = df.style
        for i in range(1, len(models) + 1):
            s = s.background_gradient(cmap="viridis", subset=[f"num_params_{i}"], gmap=gmap_list[i-1])
        with pd.option_context("display.max_rows", 1000):
            display(s)
    else:
        return df

print_param_count(my_bert, bert)
# %%

def copy_weights(my_bert: BERTLM, bert: nn.Module) -> BERTLM:
    '''Copy over the weights of `bert` to your bert implementation.'''

    my_bert_dict = dict(my_bert.named_parameters())
    bert_dict = dict(bert.named_parameters())
    
    assert len(my_bert_dict) == len(bert_dict), "Number of layers is wrong. Have you done the prev step correctly?"
    
    state_dict_to_load = {}
    
    for (my_param_name, my_param), (name, param) in zip(my_bert_dict.items(), bert_dict.items()):
        if my_param.shape == param.shape:
            state_dict_to_load[my_param_name] = param
        else:
            raise Exception(f"Parameter shapes don't match: {my_param.shape} vs {param.shape}")

    if set(state_dict_to_load.keys()) != set(my_bert.state_dict().keys()):
        raise Exception("State dicts don't match.")
    
    my_bert.load_state_dict(state_dict_to_load)
    
    return my_bert

copy_weights(my_bert, bert)


# %%

def predict(model, tokenizer, text: str, k=15) -> list[list[str]]:
    '''
    Return a list of k strings for each [MASK] in the input.
    '''
    tokens = t.LongTensor(tokenizer.encode_plus(text)['input_ids']).unsqueeze(0)
    top_k = model(tokens)[0].topk(k, dim=-1).indices.squeeze(0)[(tokens == tokenizer.mask_token_id).reshape(-1)]

    return [[tokenizer.decode([token]) for token in top_k[i]] for i in range(len(top_k))]


def test_bert_prediction(predict, model, tokenizer):
    '''Your Bert should know some names of American presidents.'''
    text = "Former President of the United States of America, George[MASK][MASK]"
    predictions = predict(model, tokenizer, text)
    print(f"Prompt: {text}")
    print("Model predicted: \n", "\n".join(map(str, predictions)))
    assert "Washington" in predictions[0]
    assert "Bush" in predictions[0]

test_bert_prediction(predict, my_bert, tokenizer)

# %%
