import os
import toml
import math
import torch
from torch import nn, Tensor
import torch.nn.functional as F

from contralog.trainer import Tokenizer
from contralog.transformer import TransformerEncoder


class MessageEncoder(nn.Module):
    """
    Encodes log messages using a Transformer Encoder.

    Args:
        ntoken (int): Vocabulary size.
        d_model (int): Embedding dimension.
        nhead (int): Number of attention heads.
        d_hid (int): Hidden layer size in the transformer.
        nlayers (int): Number of transformer encoder layers.
        d_out (int): Output feature dimension.
        max_len (int): Maximum message length (in tokens).
        dropout (float): Dropout rate.
    """

    def __init__(self, ntoken: int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, d_out: int, max_len: int, dropout: float = 0.2):
        super().__init__()
        self.d_model = d_model
        self.d_out = d_out
        
        # Calculate MLP ratio from hidden dimension
        mlp_ratio = d_hid / d_model
        
        self.transformer = TransformerEncoder(
            vocab_size=ntoken,
            dim=d_model,
            n_layers=nlayers,
            n_heads=nhead,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            pad_token_id=0,
            output_is_emb=True,   # Output is embeddings, not logits
            pool_output=True,  # Pool for contrastive learning
            input_is_emb=False,  # Input is token IDs
        )
        
        # Output projection layer
        self.linear = nn.Linear(d_model, d_out)
        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.linear.bias.data.zero_()
        self.linear.weight.data.uniform_(-initrange, initrange)
    '''
    def forward(self, src: Tensor, src_mask: Tensor = None) -> Tensor:
        """
        Forward pass through the transformer.

        Args:
            src (Tensor): Input token IDs of shape [batch, seq].
            src_mask (Tensor, optional): Padding mask with True for padding, False for valid.

        Returns:
            Tensor: Pooled output of shape [batch, d_model], L2-normalized.
        """
        if src_mask is not None:
            attention_mask = ~src_mask  # Invert: True for valid, False for padding
        else:
            attention_mask = None
        output = self.transformer(src, attention_mask=attention_mask)
        output = self.linear(output)
        return output
    '''
    def encode(self, text, tokenizer, device, batch_size: int = 512):
        """
        Tokenizes and encodes a list of log messages into fixed-size embeddings.

        Args:
            text (List[str] or str): Input log messages.
            tokenizer: Tokenizer callable returning token ids and masks.
            device: Device to run the computations on.
            batch_size (int): Batch size for processing (not the training batch size).

        Returns:
            Tensor: Encoded message representations of shape [num_messages, d_out].
        """
        if isinstance(text, str):
            text = [text]

        emb_lst = []
        for i in range(0, len(text), batch_size):
            t = text[i:i+batch_size]
            ids, attention_mask = tokenizer(t, device=device)
            # ids: [batch, max_len]
            # attention_mask: [batch, max_len] with 1 for valid, 0 for padding
            output = self.transformer(ids, attention_mask=attention_mask.bool())
            # output: [batch, d_model], pooled and L2-normalized
            emb_lst.append(output)
        output = torch.cat(emb_lst)
        output = self.linear(output)
        return output


class SequenceEncoder(nn.Module):
    """
    Encodes sequences of message embeddings using a Transformer Encoder.

    Args:
        d_model (int): Input embedding dimension.
        nhead (int): Number of attention heads.
        d_hid (int): Hidden layer size in the transformer.
        nlayers (int): Number of transformer encoder layers.
        d_out (int): Output embedding dimension.
        max_len (int): Maximum sequence length (number of logs).
        dropout (float): Dropout rate.
    """

    def __init__(self, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, d_out: int, max_len: int, dropout):
        super().__init__()
        self.d_model = d_model
        self.d_out = d_out
        
        # Calculate MLP ratio from hidden dimension
        mlp_ratio = d_hid / d_model
        
        self.transformer = TransformerEncoder(
            vocab_size=None,
            dim=d_model,
            n_layers=nlayers,
            n_heads=nhead,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            pad_token_id=0,
            output_is_emb=True,   # Output is embeddings, not logits
            pool_output=False,  # No pooling, return full sequence
            input_is_emb=True   # Input is embeddings
        )
        
        # Learnable mask token for masked sequence modeling
        self.mask_token = nn.Parameter(torch.zeros(d_model))
        nn.init.uniform_(self.mask_token, -0.1, 0.1)

    def get_mask_token(self, norm: bool = True):
        """
        Returns the learnable mask token.

        Args:
            norm (bool): If True, returns L2-normalized masking token.

        Returns:
            Tensor: The mask token vector.
        """
        if norm:
            return F.normalize(self.mask_token, p=2, dim=0)
        else:
            return self.mask_token

    def forward(self, src: Tensor, src_mask: Tensor = None) -> Tensor:
        """
        Encodes input sequences using a transformer.

        Args:
            src (Tensor): Input of shape [seq_len, batch_size, d_model].
            src_mask (Tensor, optional): Padding mask of shape [batch_size, seq_len] with True for padding.

        Returns:
            Tensor: Output of shape [seq_len, batch_size, d_out].
        """
        # Convert from sequence-first to batch-first
        src = src.transpose(0, 1)  # [batch_size, seq_len, d_model]
        
        if src_mask is not None:
            attention_mask = ~src_mask  # Invert: True for valid, False for padding
        else:
            attention_mask = None
        
        output = self.transformer(src, attention_mask=attention_mask) # [batch_size, seq_len, d_out]
        
        output = output.transpose(0, 1)  # [seq_len, batch_size, d_out]
        return output


class AnomalyModel(nn.Module):
    """
    ContraLog: Combines the MessageEncoder and SequenceEncoder.

    Args:
        model_config_path (str): Path to model configuration toml file.
        tokenizer: Tokenizer used to preprocess log messages.
        device (str): Device to run the model on.
    """

    def __init__(self, model_config_path: str, tokenizer, device: str = 'cpu'):
        super().__init__()
        self.conf = toml.load(model_config_path)
        self.device = device
        self.tokenizer = tokenizer
        self.init_message_encoder()
        self.init_sequence_encoder()

        # Parameters for sigmoid loss scaling
        self.t_prime = torch.tensor(math.log(10), device=device) 
        self.b = torch.tensor(-10, device=device) 

    @classmethod
    def from_pretrained(cls, model_path: str, device: str = 'cpu'):
        """
        Loads a pretrained model from the given path.

        Args:
            model_path (str): Directory containing saved model files (weights, tokenizer, model config). Use the save function to create these files.
            device (str): Device to map the model to.

        Returns:
            The loaded model.
        """
        tokenizer = Tokenizer.from_pretrained(model_path + '/tokenizer.sav')
        model = cls(model_config_path=model_path +
                    '/model_conf.toml', tokenizer=tokenizer, device=device)
        model.message_encoder.load_state_dict(torch.load(
            model_path + '/message_encoder.sav', map_location=device))
        model.sequence_encoder.load_state_dict(torch.load(
            model_path + '/sequence_encoder.sav', map_location=device))
        return model

    def init_message_encoder(self):
        """
        Initializes the MessageEncoder from configuration.
        """
        ntokens = self.conf['tokenizer_vocab_len']
        emsize = self.conf['emsize']
        d_hid = self.conf['d_hid_emb']
        nlayers = self.conf['n_layers_emb']
        nhead = self.conf['n_head_emb']
        dropout = self.conf['dropout_embedder']
        d_out = emsize
        max_log_len = self.conf['max_log_len']
        self.message_encoder = MessageEncoder(
            ntokens, emsize, nhead, d_hid, nlayers, d_out, max_log_len, dropout).to(self.device)

    def init_sequence_encoder(self):
        """
        Initializes the SequenceEncoder from configuration.
        """
        emsize = self.conf['emsize']
        d_hid = self.conf['d_hid_sequ']
        nlayers = self.conf['n_layers_sequ']
        nhead = self.conf['n_head_sequ']
        dropout = self.conf['dropout_sequ_model']
        d_out = emsize
        max_sequ_len = self.conf['max_sequ_len']
        self.sequence_encoder = SequenceEncoder(
            emsize, nhead, d_hid, nlayers, d_out, max_sequ_len, dropout).to(self.device)

    def embed(self, logs, batch_size: int = 256):
        """
        Encodes a list of log messages into embeddings.

        Args:
            logs (List[str]): Input log messages.
            batch_size (int): Batch size for encoding (not the training batch size).

        Returns:
            Tensor: Encoded message embeddings.
        """
        embs = self.message_encoder.encode(
            logs, self.tokenizer, device=self.device, batch_size=batch_size)
        return embs

    def save(self, path: str):
        """
        Saves the model configuration, weights, and tokenizer.

        Args:
            path (str): Directory to save model files.
        """
        os.makedirs(path, exist_ok=True)

        with open(path + '/model_conf.toml', 'w') as f:
            toml.dump(self.conf, f)
        self.tokenizer.save_pretrained(path + '/tokenizer.sav')
        torch.save(self.message_encoder.state_dict(),
                   path + '/message_encoder.sav')
        torch.save(self.sequence_encoder.state_dict(),
                   path + '/sequence_encoder.sav')
