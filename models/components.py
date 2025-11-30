import torch
import torch.nn as nn

class FeatureEncoder(nn.Module):
    """
    Encodes the (x, y) position and agent_id features into a single embedding using an MLP
    """
    def __init__(self, d_model: int, num_roles: int):
        super().__init__()
        self.pos_embedding = nn.Linear(2, d_model)
        self.agent_embedding = nn.Embedding(num_roles, d_model)
        self.transform = nn.Sequential(
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )
    
    def foward(self, x: torch.Tensor, agent_id: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): the position data of shape (B, A, T, 2)
            agent_id (torch.Tensor): the id of each agent of shape (B, A)
        Returns:
            embeddings (torch.Tensor): a tensor of embeddings of shape (B, A, T, D)
        """
        pos_embedding = self.pos_embedding(x)                         # (B, A, T, D)
        agent_embedding = self.agent_embedding(agent_id).unsqueeze(2) # (B, A, 1, D)
        return self.transform(pos_embedding + agent_embedding)

class TemporalEncoder(nn.Module):
    """
    Implementation of the "Attention is All You Need" sinusoidal temporal encodings
    Adapted for a tensor of shape (B, A, T, D) where the encodings are added along the T axis
    $$ PE_{(t, 2i) =   sin(t/10000^{2i/D})} $$
    $$ PE_{(t, 2i+1) = cos(t/10000^{2i/D})} $$
    """
    def __init__(self, d_model: int):
        super().__init__()
        MAX_LENGTH = 256
        encodings = torch.zeros(MAX_LENGTH, d_model)
        t = torch.arange(0, MAX_LENGTH, dtype=torch.float32) # (MAX_LENGTH,)
        k = torch.arange(0, d_model, 2)                      # (d_model // 2,)
        freq = torch.exp(-k * torch.log(torch.tensor(10000.0) / d_model))
        x = t.unsqueeze(1) * freq # (MAX_LENGTH, d_model // 2)
        encodings[:, 0::2] = torch.sin(t * freq)
        encodings[:, 1::2] = torch.cos(t * freq)

        self.register_buffer('encodings', encodings)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x (torch.Tensor): tensor of embedding of shape (B, A, T, D)
        Returns:
            embeddings (torch.Tensor): input tensor with temporal encodings added along the T axis
        """
        encodings = self.encodings[:x.shape[2]].unsqueeze(0).unsqueeze(0)  # (1, 1, T, D) # type: ignore
        return x + encodings

class SelfAttentionBlock(nn.Module):
    """
    "Attention is All You Need" style self-attention block
    """
    def __init__(self, d_model: int, num_heads: int, d_ff: int | None = None, dropout: float = 0.1):
        super().__init__()
        if d_ff is None:
            d_ff = 4 * d_model
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.feed_foward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        self.layer_norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor | None = None, is_causal: bool = False) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): a tensor of shape (B, S, D), where S is the sequence we are attending over
            padding_mask (torch.Tensor): a binary tensor of shape (B, S) where True means the corresponding token in x is padding
            is_causal (bool): if true, applies causal masking
        Returns:
            transformed (torch.Tensor): a transformed tensor of shape (B, S, D)
        """
        x_attn, _ = self.attn(x, x, x, key_padding_mask=padding_mask, is_casual=is_causal)
        x = self.layer_norm1(x + x_attn)
        x_ff = self.feed_foward(x)
        return self.layer_norm2(x + x_ff)

class SelfAttentionBlockTime(nn.Module):
    """
    Self-attention block where the attention is preformed along the 3rd dimension of an input of shape [B, A, T, D]
    """
    def __init__(self, d_model: int, num_heads: int, d_ff: int | None = None, dropout: float = 0.1):
        super().__init__()
        self.self_attn = SelfAttentionBlock(d_model, num_heads, d_ff, dropout)

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor, is_causal: bool = False) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): a tensor of shape (B, A, T, D), where A is the sequence we are attending over
            padding_mask (torch.Tensor): a binary tensor of shape (B, A, T) where True means the corresponding token in x is padding
            is_causal (bool): if true, applies causal masking
        Returns:
            transformed (torch.Tensor): a transformed tensor of shape (B, A, T, D)
        """
        B, A, T, D = x.shape
        x = x.reshape(B*A, T, D)
        padding_mask = padding_mask.reshape(B*A, T)
        x = self.self_attn(x, padding_mask=padding_mask, is_causal=is_causal)
        return x.reshape(B, A, T, D)

class SelfAttentionBlockAgents(nn.Module):
    """
    Self-attention block where the attention is preformed along the 2nd dimension of an input of shape [B, A, T, D]
    """
    def __init__(self, d_model: int, num_heads: int, d_ff: int | None = None, dropout: float = 0.1):
        super().__init__()
        self.self_attn = SelfAttentionBlock(d_model, num_heads, d_ff, dropout)

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): a tensor of shape (B, A, T, D), where A is the sequence we are attending over
            padding_mask (torch.Tensor): a binary tensor of shape (B, A, T) where True means the corresponding token in x is padding
        Returns:
            transformed (torch.Tensor): a transformed tensor of shape (B, A, T, D)
        """
        B, A, T, D = x.shape
        x = x.permute(0, 2, 1, 3).reshape(B*T, A, D)
        padding_mask = padding_mask.permute(0, 2, 1).reshape(B*T, A)
        x = self.self_attn(x)
        return x.reshape(B, T, A, D).permute(0, 2, 1, 3)