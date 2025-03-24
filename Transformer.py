import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=8, num_layers=4):
        super().__init__()
        self.d_model = d_model
        
        # Enhanced embedding with proper scaling
        self.embed = nn.Embedding(vocab_size, d_model)
        self.embed_scale = d_model ** 0.5
        
        # Learned positional encodings with better initialization
        self.pos_encoder = nn.Parameter(torch.zeros(20, d_model))
        nn.init.normal_(self.pos_encoder, mean=0, std=0.02)
        
        # Speed optimization: Using batch_first=True to avoid unnecessary transpositions
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True  # Changed to True for speed
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, 
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True  # Changed to True for speed
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # Output projection with simple initialization
        self.output_proj = nn.Linear(d_model, vocab_size)
        
        # Apply weight initialization
        self._init_parameters()
        
    def _init_parameters(self):
        """Initialize the parameters with appropriate scaling to avoid exploding gradients"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        
    def forward(self, src, tgt):
        # Apply embedding with scaling
        src_embed = self.embed(src) * self.embed_scale
        src_embed = src_embed + self.pos_encoder[:src.size(1)]
        
        tgt_embed = self.embed(tgt) * self.embed_scale
        tgt_embed = tgt_embed + self.pos_encoder[:tgt.size(1)]
        
        # Generate a padding mask for the source sequence
        src_key_padding_mask = (src == 0)
        tgt_key_padding_mask = (tgt == 0)
        
        # Generate a causal mask for the target sequence
        tgt_len = tgt.size(1)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_len).to(tgt.device)
        
        # Apply encoder and decoder with masks (with batch_first=True, no need to transpose)
        memory = self.encoder(
            src_embed,
            src_key_padding_mask=src_key_padding_mask
        )
        
        output = self.decoder(
            tgt_embed,
            memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask
        )
        
        return self.output_proj(output)

class LatentTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=256, num_latent=8, nhead=8, num_layers=4, bottleneck_factor=1.0):
        super().__init__()
        self.d_model = d_model
        self.num_latent = num_latent
        self.bottleneck_factor = bottleneck_factor  # Controls how severe the bottleneck is (1.0 = full restriction)
        
        # Embeddings - use same structure as SimpleTransformer for fairness
        self.embed = nn.Embedding(vocab_size, d_model)
        self.embed_scale = d_model ** 0.5
        self.pos_encoder = nn.Parameter(torch.zeros(20, d_model))
        nn.init.normal_(self.pos_encoder, mean=0.0, std=0.02)
        
        # Encoder - using batch_first for consistency with SimpleTransformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Latent processing
        self.latent_tokens = nn.Parameter(torch.zeros(num_latent, d_model))
        nn.init.normal_(self.latent_tokens, mean=0.0, std=0.02)
        
        # Latent attention projections
        self.latent_proj_q = nn.Linear(d_model, d_model)
        self.latent_proj_k = nn.Linear(d_model, d_model)
        self.latent_proj_v = nn.Linear(d_model, d_model)
        
        # Latent processing layers
        self.latent_norm1 = nn.LayerNorm(d_model)
        self.latent_norm2 = nn.LayerNorm(d_model)
        self.latent_mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )
        
        # Decoder - using batch_first for consistency
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, 
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # Output projection
        self.output_proj = nn.Linear(d_model, vocab_size)
        
        # Initialize parameters
        self._init_parameters()
        
    def _init_parameters(self):
        """Initialize the parameters in the same way as SimpleTransformer"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
                
    def _cross_attention(self, query, key, value, attn_mask=None):
        """Custom cross attention from latent tokens to encoder outputs"""
        # Project queries, keys, values
        q = self.latent_proj_q(query)  # [batch_size, num_latent, d_model]
        k = self.latent_proj_k(key)    # [batch_size, seq_len, d_model]
        v = self.latent_proj_v(value)  # [batch_size, seq_len, d_model]
        
        # Calculate attention scores [batch_size, num_latent, seq_len]
        scores = torch.bmm(q, k.transpose(1, 2)) / (self.d_model ** 0.5)
        
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask == 0, -1e9)
            
        # Apply softmax and get weighted values
        attn_weights = F.softmax(scores, dim=-1)  # [batch_size, num_latent, seq_len]
        out = torch.bmm(attn_weights, v)  # [batch_size, num_latent, d_model]
        
        return out

    def forward(self, src, tgt):
        # Encode input using batch_first=True format
        src_embed = self.embed(src) * self.embed_scale
        src_embed = src_embed + self.pos_encoder[:src.size(1)]
        memory = self.encoder(src_embed)
        
        # Process latent tokens (information bottleneck)
        batch_size = src.size(0)
        latent = self.latent_tokens.unsqueeze(0).expand(batch_size, self.num_latent, -1)
        
        # Apply attention from latent tokens to encoder memory
        latent_attended = self._cross_attention(latent, memory, memory)
        latent = self.latent_norm1(latent + latent_attended)
        
        # Apply MLP to latent tokens
        latent_mlp_out = self.latent_mlp(latent)
        latent_memory = self.latent_norm2(latent + latent_mlp_out)
        
        # Control the bottleneck - if bottleneck_factor < 1, we allow some direct access to memory
        if self.bottleneck_factor < 1.0:
            # Create a memory representation that contains both latent tokens and original memory
            # The bottleneck factor controls the balance
            mixed_memory = torch.cat([
                latent_memory,  # [batch, num_latent, d_model]
                memory * (1.0 - self.bottleneck_factor)  # [batch, seq_len, d_model]
            ], dim=1)
        else:
            # Pure latent bottleneck - decoder can only access latent tokens
            mixed_memory = latent_memory
        
        # Decode target sequence
        tgt_embed = self.embed(tgt) * self.embed_scale
        tgt_embed = tgt_embed + self.pos_encoder[:tgt.size(1)]
        
        # Generate a causal mask for the target sequence
        tgt_len = tgt.size(1)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_len).to(tgt.device)
        
        # Apply decoder
        output = self.decoder(
            tgt_embed, 
            mixed_memory,
            tgt_mask=tgt_mask
        )
        
        return self.output_proj(output)
