import torch.nn as nn

class ProjectCrossAttend(nn.Module):
    def __init__(self, in_dim, kv_dim = 512):
        super(ProjectCrossAttend, self).__init__()

        self.att_c = nn.MultiheadAttention(embed_dim=in_dim, num_heads=8, batch_first=True, dropout=0.1, add_bias_kv=True, kdim=kv_dim, vdim=kv_dim)
    
    def forward(self, x, context):
        x, _ = self.att_c(x, context, context)
        return x

if __name__ == '__main__':
    import torch
    x = torch.randn(1, 128, 64)
    context = torch.randn(1, 3000, 512)
    cross_attention = ProjectCrossAttend(in_dim=64, kv_dim=512)
    out = cross_attention(x, context)
    print(out.shape)
    