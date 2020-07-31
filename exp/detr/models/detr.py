import torch
from torch import nn
from torchvision.models import resnet50

class DETR(nn.Module):
    def __init__(
            self, 
            nqueries,
            num_classes, 
            hidden_dim, 
            nheads,
            num_encoder_layers, 
            num_decoder_layers):
        super().__init__()
        # We take only convolutional layers from ResNet-50 model
        self.backbone = nn.Sequential(
            *list(resnet50(pretrained=True).children())[:-2])
        self.conv = nn.Conv2d(2048, hidden_dim, 1)
        self.transformer = nn.Transformer(
            hidden_dim, 
            nheads,
            num_encoder_layers, 
            num_decoder_layers)
        self.linear_class = nn.Linear(hidden_dim, num_classes + 1)
        self.linear_bbox = nn.Linear(hidden_dim, 4)
        self.query_pos = nn.Parameter(torch.rand(nqueries, hidden_dim))
        self.row_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))
        self.col_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))

    def forward(self, inputs):
        x = self.backbone(inputs)
        h = self.conv(x) # BxCxHxW
        B = h.shape[0]
        H, W = h.shape[-2:]
        pos = torch.cat([
            self.col_embed[:W].unsqueeze(0).repeat(H, 1, 1), # HxWxC/2
            self.row_embed[:H].unsqueeze(1).repeat(1, W, 1), # HxWxC/2
            ], dim=-1).flatten(0, 1).unsqueeze(1) # HWx1xC
        h = self.transformer(
            pos + h.flatten(2).permute(2, 0, 1),    # encoder input: HWxBxC
            self.query_pos.unsqueeze(1).repeat(1,B,1))  # decoder input: nqueriesx1xC
        return self.linear_class(h), self.linear_bbox(h).sigmoid()


if __name__=='__main__':
    detr = DETR(8,2,10,2,3,3)
    inputs = torch.randn([5,3,100,100])
    logits, bbox = detr(inputs)
    import ipdb; ipdb.set_trace()