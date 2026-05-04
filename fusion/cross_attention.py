import torch
import torch.nn as nn
from typing import Optional


class MultiModalFusionClassifier(nn.Module):
    """
    Fuses image, audio, and video embeddings via cross-attention and classifies
    as real (0) or fake (1).

    Each modality contributes one (B, 512) token. Absent modalities are
    replaced with a learned missing_token. The three tokens are stacked into
    (B, 3, 512), passed through one MHA layer with residual + LayerNorm, then
    flattened and classified by a small MLP.
    """

    def __init__(self, embed_dim: int = 512, num_heads: int = 8, dropout: float = 0.3):
        super().__init__()
        self.missing_token = nn.Parameter(torch.zeros(1, embed_dim))

        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

        self.head = nn.Sequential(
            nn.Linear(embed_dim * 3, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )

    def _resolve(self, emb: Optional[torch.Tensor], batch_size: int) -> torch.Tensor:
        if emb is not None:
            return emb  # (B, 512)
        return self.missing_token.expand(batch_size, -1)  # (B, 512)

    def forward(
        self,
        img_emb: Optional[torch.Tensor] = None,
        aud_emb: Optional[torch.Tensor] = None,
        vid_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        present = next(e for e in (img_emb, aud_emb, vid_emb) if e is not None)
        B = present.shape[0]

        img = self._resolve(img_emb, B)
        aud = self._resolve(aud_emb, B)
        vid = self._resolve(vid_emb, B)

        tokens = torch.stack([img, aud, vid], dim=1)   # (B, 3, 512)
        attended, _ = self.attn(tokens, tokens, tokens) # (B, 3, 512)
        tokens = self.norm(tokens + attended)           # residual + LN

        flat = tokens.flatten(1)                        # (B, 1536)
        return self.head(flat)                          # (B, 1)


if __name__ == "__main__":
    import itertools

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiModalFusionClassifier().to(device)
    model.eval()

    names = ["img", "aud", "vid"]
    B = 2

    with torch.no_grad():
        for r in range(1, 4):
            for combo in itertools.combinations(range(3), r):
                kwargs = {}
                for i in combo:
                    kwargs[f"{names[i]}_emb"] = torch.rand(B, 512, device=device)
                present = "+".join(names[i] for i in combo)
                out = model(**kwargs)
                print(f"  [{present:13s}]  output shape: {out.shape}  sample: {out[0].item():.4f}")
