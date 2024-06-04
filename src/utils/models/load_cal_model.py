import torch

from .cal_model import WSDAN_CAL
from .model_wrappers import NormalizationAndCutoutWrapper, NormalizationAndResizeWrapper, IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

class CALWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    @property
    def return_layers(self):
        """I'm the 'x' property."""
        print("getter of x called")
        return self.model.clip_model.clip_model._return_layers

    @return_layers.setter
    def return_layers(self, value):
        print("setter of x called")
        self.model.clip_model.clip_model._return_layers = value

    @return_layers.deleter
    def return_layers(self):
        print("deleter of x called")
        del self.model.clip_model.clip_model._return_layers

    def forward(self, *args):
        out = self.model(*args)
        return out[0]

    def state_dict(self, *args, **kwargs):
        return self.model.state_dict(*args, **kwargs)
def _load_model(dataset):
    if dataset == 'cub':
        num_classes = 200
        chkpt = 'cal_models/bird/wsdan-resnet101-cal/model_bestacc.pth'
    elif dataset == 'aircraft':
        num_classes = 100
        chkpt = 'cal_models/aircraft/wsdan-resnet101-cal/model_bestacc.pth'
    elif dataset == 'cars':
        num_classes = 196
        chkpt = 'cal_models/car/wsdan-resnet101-cal/model_bestacc.pth'
    else:
        raise NotImplementedError()

    num_attentions = 32
    net = 'resnet101'
    net = WSDAN_CAL(num_classes=num_classes, M=num_attentions, net=net, pretrained=True)

    checkpoint = torch.load(chkpt)

    # Get epoch and some logs
    logs = checkpoint['logs']
    start_epoch = int(logs['epoch'])  # start from the beginning

    # Load weights
    state_dict = checkpoint['state_dict']
    net.load_state_dict(state_dict)

    # load feature center
    if 'feature_center' in checkpoint:
        feature_center = checkpoint['feature_center'].cuda()

    img_size = 448
    net = CALWrapper(net)
    return net, img_size

def load_cal_model(dataset, auto_resize=True):
    net, img_size = _load_model(dataset)
    if not auto_resize:
        img_size = None

    net = NormalizationAndResizeWrapper(net, size=img_size, mean=torch.tensor(IMAGENET_DEFAULT_MEAN),
                                          std=torch.tensor(IMAGENET_DEFAULT_STD))

    net = net.eval()
    for param in net.parameters():
        param.requires_grad_(False)
    return net

def load_cal_model_with_cutout(dataset, cut_power, num_cutouts, checkpointing=False, auto_resize=True,
                               noise_sd=0, noise_schedule=None, noise_descending_steps=None ):
    net, img_size = _load_model(dataset)
    if not auto_resize:
        img_size = None

    net = NormalizationAndCutoutWrapper(net, size=img_size, mean=torch.tensor(IMAGENET_DEFAULT_MEAN),
                                          std=torch.tensor(IMAGENET_DEFAULT_STD),
                                          cut_power=cut_power, num_cutouts=num_cutouts,
                                        checkpointing=checkpointing,
                                        noise_sd=noise_sd, noise_schedule=noise_schedule, noise_descending_steps=noise_descending_steps)

    net = net.eval()
    for param in net.parameters():
        param.requires_grad_(False)
    return net
