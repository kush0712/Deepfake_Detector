import torch
import torch.nn as nn
import timm


class DeepFakeDetector(nn.Module):
    """
    EfficientNet-B0 → GRU → FC
    """

    def __init__(
        self,
        backbone_name="efficientnet_b0",
        embed_dim=1280,
        gru_hidden=256,
        gru_layers=1,
        gru_dropout=0.3,
    ):
        super().__init__()

        # Backbone
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=False,   # IMPORTANT for loading saved weights
            num_classes=0,
            global_pool="avg",
        )

        self.embed_dim = embed_dim

        # GRU
        self.gru = nn.GRU(
            input_size=embed_dim,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
            dropout=0.0,
            bidirectional=False,
        )

        self.gru_drop = nn.Dropout(gru_dropout)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(gru_hidden, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        B, T, C, H, W = x.shape

        # Frame-wise features
        x_flat = x.view(B * T, C, H, W)

        feats = self.backbone(x_flat)

        feats = feats.view(B, T, self.embed_dim)

        # GRU temporal modeling
        gru_out, _ = self.gru(feats)

        last_hidden = gru_out[:, -1, :]

        last_hidden = self.gru_drop(last_hidden)

        logits = self.classifier(last_hidden)

        return logits.squeeze(1)