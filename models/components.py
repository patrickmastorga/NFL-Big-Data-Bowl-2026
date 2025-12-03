import torch
import torch.nn as nn

MAX_T = 256

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
    
    def forward(self, x: torch.Tensor, agent_id: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): the position data of shape (B, A, T, 2)
            agent_id (torch.Tensor): the id of each agent of shape (B, A)
        Returns:
            embeddings (torch.Tensor): a tensor of embeddings of shape (B, A, T, D)
        """
        pos_embedding = self.pos_embedding(x)                         # (B, A, T, D)
        agent_embedding = self.agent_embedding(agent_id).unsqueeze(2) # (B, A, 1, D)
        embeddings = self.transform(pos_embedding + agent_embedding)
        return embeddings

class TemporalEncoder(nn.Module):
    """
    Implementation of the "Attention is All You Need" sinusoidal temporal encodings
    Adapted for a tensor of shape (B, A, T, D) where the encodings are added along the T axis
    """
    def __init__(self, d_model: int):
        super().__init__()
        encodings = torch.zeros(MAX_T, d_model)
        t = torch.arange(0, MAX_T, dtype=torch.float32) # (MAX_SEQ_LENGTH,)
        k = torch.arange(0, d_model, 2)                          # (d_model // 2,)
        freq = torch.exp(-k * torch.log(torch.tensor(10000.0) / d_model))
        x = t.unsqueeze(1) * freq # (MAX_SEQ_LENGTH, d_model // 2)
        encodings[:, 0::2] = torch.sin(x)
        encodings[:, 1::2] = torch.cos(x)

        self.register_buffer('encodings', encodings)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x (torch.Tensor): tensor of embedding of shape (B, A, T, D)
        Returns:
            embeddings (torch.Tensor): input tensor with temporal encodings added along the T axis
        """
        x = x + self.encodings[:x.shape[2]].unsqueeze(0).unsqueeze(0)  # (B, A, T, D) + (1, 1, T, D) # type: ignore
        return x

class DualTemporalEncoder(nn.Module):
    """
    Implementation of the "Attention is All You Need" sinusoidal temporal encodings
    Adapted for a tensor of shape (B, A, T, D) where the encodings are added along the T axis
    Encodings are split into two parts [time_elapsed, time_remaining] and concatenated
    """
    def __init__(self, d_model: int):
        super().__init__()
        encodings = torch.zeros(MAX_T, d_model // 2)
        t = torch.arange(0, MAX_T, dtype=torch.float32) # (MAX_SEQ_LENGTH,)
        k = torch.arange(0, d_model // 2, 2)            # (d_model // 4,)
        freq = torch.exp(-k * torch.log(torch.tensor(10000.0) / (d_model // 2)))
        x = t.unsqueeze(1) * freq # (MAX_SEQ_LENGTH, d_model // 4)
        encodings[:, 0::2] = torch.sin(x)
        encodings[:, 1::2] = torch.cos(x)

        self.register_buffer('encodings', encodings)

    def forward(self, x: torch.Tensor, time_remain: torch.Tensor):
        """
        Args:
            x (torch.Tensor): tensor of embedding of shape (B, A, T, D)
            time_remain (Tensor): (B, A, T) long tensor representing how much time is left in the play
        Returns:
            embeddings (torch.Tensor): input tensor with temporal encodings added along the T axis
        """
        B, A, T, D = x.shape
        time_elapsed_enc = self.encodings[:T].unsqueeze(0).unsqueeze(0).expand(B, A, -1, -1) # type: ignore
        time_remain_enc = self.encodings[time_remain] # type: ignore
        x = x + torch.cat([time_elapsed_enc, time_remain_enc], dim=3)
        return x

class AttentionBlock(nn.Module):
    """
    "Attention is All You Need" style attention block
    """
    def __init__(
            self,
            d_model: int,
            num_heads: int,
            d_ff: int | None = None,
            dropout: float = 0.1,
            is_causal: bool = False
        ):
        super().__init__()
        self.is_causal = is_causal
        if d_ff is None:
            d_ff = 4 * d_model
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        self.layer_norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, m: torch.Tensor, m_padding_mask: torch.Tensor | None = None, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): a tensor of shape (B, S1, D) that is the input to the operation
            m (torch.Tensor) a tensor of shape (B, S1, D) where S1 is the sequence each token in x is attending over
            m_padding_mask (torch.Tensor): a binary tensor of shape (B, S1) where True means the corresponding entry in m is padding
            is_causal (bool): if true, applies causal masking
        Returns:
            transformed (torch.Tensor): a transformed tensor of shape (B, S1, D)
        """
        x_attn, _ = self.attn(x, m, m, key_padding_mask=m_padding_mask, attn_mask=attn_mask)
        x = self.layer_norm1(x + x_attn)
        x_ff = self.feed_forward(x)
        x = self.layer_norm2(x + x_ff)
        return x

class SelfAttentionBlockTime(nn.Module):
    """
    Self-attention block where the self-attention is preformed along the 3rd dimension of an input of shape [B, A, T, D]
    """
    def __init__(
            self,
            d_model: int,
            num_heads: int,
            d_ff: int | None = None,
            dropout: float = 0.1,
            is_causal: bool = False
        ):
        super().__init__()
        self.is_causal = is_causal
        self.self_attn = AttentionBlock(d_model, num_heads, d_ff, dropout)
        if self.is_causal:
            self.register_buffer("causal_mask", torch.triu(torch.ones(MAX_T, MAX_T), 1).bool())

    def forward(self, x: torch.Tensor, x_padding_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): a tensor of shape (B, A, T, D), where T is the sequence we are attending over
            x_padding_mask (torch.Tensor): a binary tensor of shape (B, A, T) where True means the corresponding token in x is padding
            is_causal (bool): if true, applies causal masking
        Returns:
            transformed (torch.Tensor): a transformed tensor of shape (B, A, T, D)
        """
        B, A, T, D = x.shape
        # collapse the agents and batch dimension together
        print('Hello, World! Time')
        x = x.reshape(B*A, T, D)
        x_padding_mask = x_padding_mask.reshape(B*A, T)
        # preform attention and reshape
        if self.is_causal:
            x = self.self_attn(x, x, x_padding_mask, self.causal_mask[:T, :T]) # type: ignore
        else:
            x = self.self_attn(x, x, x_padding_mask)
        x = x.reshape(B, A, T, D)
        return x

class SelfAttentionBlockAgents(nn.Module):
    """
    Self-attention block where the self-attention is preformed along the 2nd dimension of an input of shape [B, A, T, D]
    """
    def __init__(self, d_model: int, num_heads: int, d_ff: int | None = None, dropout: float = 0.1):
        super().__init__()
        self.self_attn = AttentionBlock(d_model, num_heads, d_ff, dropout)

    def forward(self, x: torch.Tensor, x_padding_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): a tensor of shape (B, A, T, D), where A is the sequence we are attending over
            x_padding_mask (torch.Tensor): a binary tensor of shape (B, A, T) where True means the corresponding token in x is padding
        Returns:
            transformed (torch.Tensor): a transformed tensor of shape (B, A, T, D)
        """
        B, A, T, D = x.shape
        # collapse the time and batch dimension together
        x = x.permute(0, 2, 1, 3).reshape(B*T, A, D)
        x_padding_mask = x_padding_mask.permute(0, 2, 1).reshape(B*T, A)
        # preform attention and reshape
        x = self.self_attn(x, x, x_padding_mask)
        x = x.reshape(B, T, A, D).permute(0, 2, 1, 3)
        return x

class CrossAttentionBlockAgents(nn.Module):
    """
    Cross-attention block where cross-attention is preformed along the 2nd dimension of an input of shape [B, A, T, D] onto a sequence of shape [B, S, D]
    """
    def __init__(self, d_model: int, num_heads: int, d_ff: int | None = None, dropout: float = 0.1):
        super().__init__()
        self.cross_attn = AttentionBlock(d_model, num_heads, d_ff, dropout)

    def forward(self, x: torch.Tensor, m: torch.Tensor, m_padding_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): a tensor of shape (B, A, T, D), where T is the sequence we are attending over
            m (torch.Tensor) a tensor of shape (B, S, D) where S is the sequence each agent in x is attending over
            m_padding_mask (torch.Tensor): a binary tensor of shape (B, S) where True means the corresponding entry in m is padding
        Returns:
            transformed (torch.Tensor): a transformed tensor of shape (B, A, T, D)
        """
        B, A, T, D = x.shape
        _, S, _ = m.shape

        # collapse the time and batch dimension together
        x = x.permute(0, 2, 1, 3).reshape(B*T, A, D)
        # repeat memory for each time step T and collapse with batch
        m = m.unsqueeze(1).expand(-1, T, -1, -1).reshape(B*T, S, D)
        m_padding_mask = m_padding_mask.unsqueeze(1).expand(-1, T, -1).reshape(B*T, S)
        # preform attention and reshape
        x = self.cross_attn(x, m, m_padding_mask)
        x = x.reshape(B, T, A, D).permute(0, 2, 1, 3)
        return x

class EncoderBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int | None = None, dropout: float = 0.1):
        super().__init__()
        self.attn_t = SelfAttentionBlockTime(d_model, num_heads, d_ff, dropout)
        self.attn_a = SelfAttentionBlockAgents(d_model, num_heads, d_ff, dropout)
    
    def forward(self, x: torch.Tensor, x_padding_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): a tensor of shape (B, A, T, D)
            x_padding_mask (torch.Tensor): a binary tensor of shape (B, A, T) where True means the corresponding token in x is padding
        Returns:
            transformed (torch.Tensor): a transformed tensor of shape (B, A, T, D)
        """
        x = self.attn_t(x, x_padding_mask)
        x = self.attn_a(x, x_padding_mask)
        return x

class DecoderBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int | None = None, dropout: float = 0.1):
        super().__init__()
        self.attn_t = SelfAttentionBlockTime(d_model, num_heads, d_ff, dropout, is_causal=True)
        self.attn_a = SelfAttentionBlockAgents(d_model, num_heads, d_ff, dropout)
        self.cross_attn = CrossAttentionBlockAgents(d_model, num_heads, d_ff, dropout)
    
    def forward(self, x: torch.Tensor, z: torch.Tensor, x_padding_mask: torch.Tensor, z_padding_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): a tensor of shape (B, A, T, D)
            z (torch.Tensor): a tensor of shape (B, S, D) of encoder outputs
            x_padding_mask (torch.Tensor): a binary tensor of shape (B, A, T) where True means the corresponding token in x is padding
            z_padding_mask (torch.Tensor): a binary tensor of shape (B, S) where True means the corresponding token in z is padding
        Returns:
            transformed (torch.Tensor): a transformed tensor of shape (B, A, T, D)
        """
        print('Hello, World!')
        x = self.attn_t(x, x_padding_mask)
        print(f'attn_t: {torch.isnan(x).any().item()}')
        x = self.attn_a(x, x_padding_mask)
        print(f'attn_a: {torch.isnan(x).any().item()}')
        x = self.cross_attn(x, z, z_padding_mask)
        print(f'cross_attn: {torch.isnan(x).any().item()}')
        return x