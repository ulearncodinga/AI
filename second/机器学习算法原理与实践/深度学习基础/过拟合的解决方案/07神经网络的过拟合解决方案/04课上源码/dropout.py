import torch


pre_layer_out = torch.tensor([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]], dtype=torch.float32)

m = torch.nn.Dropout(p=0.2)
input = torch.randn(10, 1)
print(pre_layer_out)

# output = m(pre_layer_out)
# print(output)

def dropout_layer(p):
    global input, pre_layer_out
    if p == 1:
        return torch.zeros_like(pre_layer_out)
    if p == 0:
        return input
    mask = (input > p).float()
    print(mask)
    return mask * pre_layer_out / (1.0 - p)

print(dropout_layer(0.2))


