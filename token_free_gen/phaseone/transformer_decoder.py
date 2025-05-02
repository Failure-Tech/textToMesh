import torch.nn as nn

class TransformerDecoder(nn.Module):
    def __init__(self, token_dim=256, num_layers=4, num_heads=8):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=token_dim,
            nhead=num_heads,
            dim_feedforward=1024,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(token_dim, 3*8*8)

    def forward(self, tokens):
        x = self.transformer(x)
        out = self.output_proj(x)
        return out
    
