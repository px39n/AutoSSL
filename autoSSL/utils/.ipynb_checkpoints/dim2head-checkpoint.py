from lightly.models.modules import heads
import torch.nn as nn

def dim2head(dimension):
    head_list = []
    for dimin,dimout, bn_type, relu_type in dimension:
        bn_module = nn.BatchNorm1d(dimout) if bn_type == "BN" else None
        relu_module = nn.ReLU(inplace=True) if relu_type == "RELU" else None
        head_list.append((dimin, dimout, bn_module, relu_module))
    
    return heads.ProjectionHead(head_list)
