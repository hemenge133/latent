import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=8, num_layers=4):
        super().__init__()
        # Reduce complexity of the model for better training
        self.embed = nn.Embedding(vocab_size, d_model)
        # Initialize positional encodings with small values for stability
        self.pos_encoder = nn.Parameter(torch.randn(20, d_model) * 0.02)
        
        # Use proper initialization and dropout for encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead,
            dim_feedforward=d_model * 4,  # Increased FFN size
            dropout=0.1,  # Add dropout
            activation="gelu"  # Use GELU activation
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Use proper initialization and dropout for decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, 
            nhead=nhead,
            dim_feedforward=d_model * 4,  # Increased FFN size
            dropout=0.1,  # Add dropout
            activation="gelu"  # Use GELU activation
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.output_proj = nn.Linear(d_model, vocab_size)
        
        # Apply proper weight initialization
        self._init_parameters()
        
    def _init_parameters(self):
        """Initialize the parameters with appropriate scaling to avoid exploding gradients"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, tgt):
        # Add explicit normalization for embeddings
        src_embed = self.embed(src) * (self.embed.embedding_dim ** 0.5)  # Scale embeddings
        src_embed = src_embed + self.pos_encoder[:src.size(1)]
        
        tgt_embed = self.embed(tgt) * (self.embed.embedding_dim ** 0.5)  # Scale embeddings
        tgt_embed = tgt_embed + self.pos_encoder[:tgt.size(1)]
        
        # Generate a padding mask for the source sequence
        src_key_padding_mask = (src == 0).to(src.device)
        tgt_key_padding_mask = (tgt == 0).to(tgt.device)
        
        # Generate a causal mask for the target sequence
        tgt_len = tgt.size(1)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_len).to(tgt.device)
        
        # Apply encoder and decoder with masks
        memory = self.encoder(
            src_embed.transpose(0, 1),
            src_key_padding_mask=src_key_padding_mask
        )
        
        output = self.decoder(
            tgt_embed.transpose(0, 1),
            memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask
        )
        
        return self.output_proj(output.transpose(0, 1))

class LatentTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=256, num_latent=8, nhead=8, num_layers=4):
        super().__init__()
        self.d_model = d_model
        self.num_latent = num_latent
        
        # Embeddings
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = nn.Parameter(torch.randn(20, d_model) * 0.02)
        
        # Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Latent processing
        self.latent_tokens = nn.Parameter(torch.zeros(num_latent, d_model))
        nn.init.normal_(self.latent_tokens, mean=0.0, std=0.02)
        
        self.latent_proj_q = nn.Linear(d_model, d_model)
        self.latent_proj_k = nn.Linear(d_model, d_model)
        self.latent_proj_v = nn.Linear(d_model, d_model)
        
        self.latent_norm1 = nn.LayerNorm(d_model)
        self.latent_norm2 = nn.LayerNorm(d_model)
        self.latent_mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )
        
        # Decoder
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # Output
        self.output_proj = nn.Linear(d_model, vocab_size)
        
        # Initialize parameters
        self._init_parameters()
        
    def _init_parameters(self):
        """Initialize the parameters with appropriate scaling to avoid exploding gradients"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
                
    def _cross_attention(self, query, key, value, attn_mask=None):
        """Custom cross attention implementation with better control"""
        # Project queries, keys, values
        q = self.latent_proj_q(query)  # [num_latent, batch_size, d_model]
        k = self.latent_proj_k(key)    # [seq_len, batch_size, d_model]
        v = self.latent_proj_v(value)  # [seq_len, batch_size, d_model]
        
        # Transpose for batched matrix multiplication
        # [num_latent, batch_size, d_model] -> [batch_size, num_latent, d_model]
        q = q.transpose(0, 1)
        # [seq_len, batch_size, d_model] -> [batch_size, seq_len, d_model]
        k = k.transpose(0, 1)
        v = v.transpose(0, 1)
        
        # Calculate attention scores [batch_size, num_latent, seq_len]
        scores = torch.bmm(q, k.transpose(1, 2)) / (self.d_model ** 0.5)
        
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask == 0, -1e9)
            
        # Apply softmax and get weighted values
        attn_weights = F.softmax(scores, dim=-1)  # [batch_size, num_latent, seq_len]
        out = torch.bmm(attn_weights, v)  # [batch_size, num_latent, d_model]
        
        # Transpose back to original format [num_latent, batch_size, d_model]
        out = out.transpose(0, 1)
        
        return out

    def forward(self, src, tgt):
        # Encode input
        src_embed = self.embed(src) + self.pos_encoder[:src.size(1)]
        memory = self.encoder(src_embed.transpose(0,1))
        
        # Process latent tokens
        batch_size = src.size(0)
        latent = self.latent_tokens.unsqueeze(1).expand(-1, batch_size, -1)
        
        # Apply attention from latent tokens to encoder memory
        latent_attended = self._cross_attention(latent, memory, memory)
        latent = self.latent_norm1(latent + latent_attended)
        
        # Apply MLP to latent tokens
        latent_mlp_out = self.latent_mlp(latent)
        latent_memory = self.latent_norm2(latent + latent_mlp_out)

        # Decode with target and latent memory
        tgt_embed = self.embed(tgt) + self.pos_encoder[:tgt.size(1)]
        output = self.decoder(tgt_embed.transpose(0,1), latent_memory)
        
        return self.output_proj(output.transpose(0,1))
