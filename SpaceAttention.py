import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialSelfAttention(nn.Module):
    def __init__(self, embed_dim):
        super(SpatialSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.conv_theta = nn.Conv1d(embed_dim, embed_dim, kernel_size=1)
        self.conv_phi = nn.Conv1d(embed_dim, embed_dim, kernel_size=1)
        self.conv_g = nn.Conv1d(embed_dim, embed_dim, kernel_size=1)
        self.conv_out = nn.Conv1d(embed_dim, embed_dim, kernel_size=1)

    def forward(self, x):
        # Calculate queries, keys, and values
        theta = self.conv_theta(x)
        phi = self.conv_phi(x)
        g = self.conv_g(x)

        # Calculate attention scores
        attention = F.softmax(torch.bmm(theta.permute(0, 2, 1), phi), dim=2)

        # Calculate weighted sum using attention scores
        out = torch.bmm(g, attention.permute(0, 2, 1))

        # Apply output convolution layer
        out = self.conv_out(out)

        return out


class SpatialSelfAttentionModel(nn.Module):
    def __init__(self, input_channels, patch_size, embed_dim):
        super(SpatialSelfAttentionModel, self).__init__()
        self.patch_size = patch_size
        self.patch_embedding = nn.Conv2d(input_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.self_attention = SpatialSelfAttention(embed_dim)
        self.transposed_conv = nn.ConvTranspose2d(embed_dim, input_channels, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # Apply patch embedding
        patches = self.patch_embedding(x)

        # Reshape patches to (batch_size, num_channels, height*width)
        batch_size, num_channels, h, w = patches.size()
        patches = patches.view(batch_size, num_channels, -1)

        # Apply spatial self-attention
        attention_output = self.self_attention(patches)

        # Reshape back to (batch_size, num_channels, height, width)
        attention_output = attention_output.view(batch_size, num_channels, h, w)

        # Apply transposed convolution to match input size
        output = self.transposed_conv(attention_output)

        return output
