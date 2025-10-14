import torch

state_dict = torch.load('data/voc2007/partial_label_0.10.pt')

print(state_dict)

# print(state_dict['some_layer.weight'])
