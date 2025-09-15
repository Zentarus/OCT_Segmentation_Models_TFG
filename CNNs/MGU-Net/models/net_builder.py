from models.nets.TSNet import TSMGUNet
from models.nets.OSNet import OSMGUNet

def net_builder(name,pretrained_model=None,pretrained=False,n_classes=10):
    if name == 'tsmgunet':
        net = TSMGUNet(n_classes=n_classes)
    elif name == 'osmgunet':
        net = OSMGUNet(n_classes=n_classes)
    else:
        raise NameError("Unknow Model Name!")
    return net
