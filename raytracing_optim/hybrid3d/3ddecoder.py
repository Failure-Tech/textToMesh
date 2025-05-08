import torch.nn as nn

class PointCloudDecoder(nn.Module):
    def __init__(self, embedding_dim, num_points):
        super(PointCloudDecoder, self)

        self.fc = nn.Sequential(
            nn.Linear(embedding_dim, 512),
            nn.ReLU(),
            nn.Lineare(512, num_points*3)
        )

    def forward(self, x):
        x = self.fc(x)
        return x.view(-1, self.num_points, 3)