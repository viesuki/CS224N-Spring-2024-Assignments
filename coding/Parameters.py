import torch
from thop import profile
from torchvision import models

# %%

model = models.resnet18()
input = torch.randn(1, 3, 224, 224)
flops, params = profile(model, inputs=(input, ))
print(f"FLOPs: {flops / 1e9:.2f} G")
print(f"Params: {params / 1e6:.2f} M")


# %%
from ptflops import get_model_complexity_info

model = models.resnet18()
macs, params = get_model_complexity_info(model, (3, 224, 224), as_strings=True, print_per_layer_stat=True)
print(f"MACs: {macs}, Params: {params}")

# %%

from torchinfo import summary

model = models.resnet18()
summary(model, input_size=(1, 3, 224, 224))



# %%
import torch
import torch.nn as nn

class MyResidual(nn.Module):
        def __init__(self):
            super(MyResidual, self).__init__()

            self.cnn_layer1 = nn.Sequential(
                nn.Conv2d(3, 64, 3, 1, 1),
                nn.BatchNorm2d(64),         # [64, 128, 128]
            )

            self.cnn_layer2 = nn.Sequential(
                nn.Conv2d(64, 64, 3, 1, 1),
                nn.BatchNorm2d(64),         # [64, 128, 128]
            )

            self.cnn_layer3 = nn.Sequential(
                nn.Conv2d(64, 128, 3, 2, 1),
                nn.BatchNorm2d(128),        # [128, 64, 64]
            )

            self.cnn_layer4 = nn.Sequential(
                nn.Conv2d(128, 128, 3, 1, 1),
                nn.BatchNorm2d(128),        # [128, 64, 64]
            )
            self.cnn_layer5 = nn.Sequential(
                nn.Conv2d(128, 256, 3, 2, 1),
                nn.BatchNorm2d(256),        # [256, 32, 32]
            )
            self.cnn_layer6 = nn.Sequential(
                nn.Conv2d(256, 256, 3, 1, 1),
                nn.BatchNorm2d(256),        # [256, 32, 32]
            )
            self.cnn_layer7 = nn.Sequential(
                nn.Conv2d(256, 512, 3, 2, 1),
                nn.BatchNorm2d(512),        # [512, 16, 16]
            )
            
            self.fc_layer = nn.Sequential(
                nn.AdaptiveAvgPool2d((4, 4)), # [512, 4, 4]
                nn.Flatten(),               # [512 * 4 * 4]
                nn.Linear(512 * 4 * 4, 1024),
                nn.ReLU(),
                nn.Dropout(0.5),

                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Dropout(0.5),
                
                nn.Linear(512, 11)
            )
            self.relu = nn.ReLU()

        def forward(self, x):
            # input (x): [batch_size, 3, 128, 128]
            # output: [batch_size, 11]

            # Extract features by convolutional layers.
            x1 = self.cnn_layer1(x)

            x1 = self.relu(x1)

            x2 = self.cnn_layer2(x1) + x1

            x2 = self.relu(x2)

            x3 = self.cnn_layer3(x2)

            x3 = self.relu(x3) + x3

            x4 = self.cnn_layer4(x3)

            x4 = self.relu(x4)

            x5 = self.cnn_layer5(x4)

            x5 = self.relu(x5)

            x6 = self.cnn_layer6(x5) + x5

            x6 = self.relu(x6)

            # The extracted feature map must be flatten before going to fully-connected layers.
            xout = self.cnn_layer7(x6)

            # The features are transformed by fully-connected layers to obtain the final logits.
            xout = self.fc_layer(xout)
            return xout
# %%
model = MyResidual()
# summary(model, input_size=(1, 3, 128, 128))
macs, params = get_model_complexity_info(model, (3, 128, 128), as_strings=True, print_per_layer_stat=True)



# %%
# Single Full connected Layer
class SimpleFC(nn.Module):
    def __init__(self):
        super(SimpleFC, self).__init__()
        self.conv = nn.Conv2d(3, 1024, 3, 1, 1)
        self.fc = nn.Linear(1024 * 4 * 4, 11)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = torch.flatten(x, 1)  # Flatten all dimensions except batch
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x
    
model1 = SimpleFC()
model2 = SimpleFC()
macs, params1 = get_model_complexity_info(model1, (3, 4, 4), as_strings=True, print_per_layer_stat=True)
print(f"MACs: {macs}, Params: {params1} \n")

flops, params2 = profile(model2, inputs=(torch.randn(1, 3, 4, 4), ))
print(f"FLOPs: {flops / 1e3:.2f} K")
print(f"Params: {params2 / 1e3:.2f} K \n")

summary(model2, input_size=(1, 3, 4, 4))
# %%
layer = nn.Linear(512, 1000, bias=True)
summary(layer, input_size=(1, 512))
# %%
conv = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1, stride=2)
summary(conv, input_size=(1, 3, 224, 224))  # params should be 3*3*3*64 + 64
# %%gconv = nn.Conv2d(32, 64, kernel_size=3, padding=1, groups=2)
gconv = nn.Conv2d(32, 64, kernel_size=3, padding=1, groups=2)
summary(gconv, input_size=(1, 32, 56, 56))  # 3*3*32/2*64 + 64 = 9280

# %%
depthwise_separable = nn.Sequential(
    nn.Conv2d(32, 32, kernel_size=3, padding=1, groups=32),  # depthwise
    nn.Conv2d(32, 64, kernel_size=1)                         # pointwise
)
summary(depthwise_separable, input_size=(1, 32, 56, 56))
# 3*3*32 + 32 + 32*64 + 64 = 2432
# %%
bn = nn.BatchNorm2d(64)
summary(bn, input_size=(1, 64, 32, 32))   # expect 2*64 params (γ, β)
# %%
ln = nn.LayerNorm(128)
summary(ln, input_size=(1, 10, 128))      # last dim = 128 → expect 2*128 params
# %%
class LSTMWrap(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=128, hidden_size=256, num_layers=1, batch_first=True)
    def forward(self, x):
        y, _ = self.lstm(x)
        return y

model = LSTMWrap()
summary(model, input_size=(32, 10, 128))  # (batch, seq, d_in)
# %%
class GRUWrap(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(input_size=128, hidden_size=256, num_layers=1, batch_first=True)
    def forward(self, x):
        y, _ = self.gru(x)
        return y

model = GRUWrap()
summary(model, input_size=(32, 10, 128))
# %%
class MHAWrap(nn.Module):
    def __init__(self):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=512, num_heads=1, batch_first=True)
    def forward(self, x):
        y, _ = self.mha(x, x, x)
        return y

model = MHAWrap()
summary(model, input_size=(32, 16, 512))

# %%
ffn = nn.Sequential(
    nn.Linear(512, 2048),
    nn.GELU(),
    nn.Linear(2048, 512)
)
summary(ffn, input_size=(32, 16, 512))  # (batch, seq, d)
# %%
enc = nn.TransformerEncoderLayer(d_model=512, nhead=8, dim_feedforward=2048, batch_first=True)
summary(enc, input_size=(32, 16, 512))
# %%
