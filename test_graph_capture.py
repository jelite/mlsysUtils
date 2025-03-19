import torch
import torchvision.models as models
from  graph_mlsys.graph_capture import get_graph

# Load the ResNet-50 model
model = models.resnet50()  # Use pretrained weights
model = model.cuda()  # Move model to GPU if available

input = torch.randn(1, 3, 224, 224).cuda()  # Create random input tensor

g, output = get_graph(model, input)  # Capture the CUDA graph
# 그래프 실행
g.replay()  # 그래프 실행

print(output)