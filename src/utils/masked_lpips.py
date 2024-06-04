from lpips import LPIPS, upsample, spatial_average, normalize_tensor
import lpips as lp
import torch
import torchvision.transforms.functional as TF
import torch.nn as nn
import os

class MaskedLPIPS(LPIPS):
    def __init__(self, pretrained=True, net='alex', version='0.1', lpips=True, spatial=False,
                 pnet_rand=False, pnet_tune=False, use_dropout=True, model_path=None, eval_mode=True, verbose=True):
        weigth_dir = os.path.join(os.path.split(lp.__file__)[0], 'weights', f'v{version}')
        if net == 'alex':
            model_path = os.path.join(weigth_dir, 'alex.pth')
        elif net == 'vgg':
            model_path = os.path.join(weigth_dir, 'vgg.pth')
        else:
            raise NotImplementedError()
        super().__init__(pretrained=pretrained, net=net, version=version, lpips=lpips, spatial=spatial,
                         pnet_rand=pnet_rand, pnet_tune=pnet_tune, use_dropout=use_dropout,
                         model_path=model_path, eval_mode=eval_mode, verbose=verbose)

    def forward(self, in0, in1, mask, retPerLayer=False, normalize=False):
        if normalize:  # turn on this flag if input is [0,1] so it can be adjusted to [-1, +1]
            in0 = 2 * in0 - 1
            in1 = 2 * in1 - 1

        # v0.0 - original release had a bug, where input was not scaled
        in0_input, in1_input = (self.scaling_layer(in0), self.scaling_layer(in1)) if self.version == '0.1' else (
        in0, in1)
        outs0, outs1 = self.net.forward(in0_input), self.net.forward(in1_input)
        feats0, feats1, diffs = {}, {}, {}

        for kk in range(self.L):
            feats0[kk], feats1[kk] = normalize_tensor(outs0[kk]), normalize_tensor(outs1[kk])
            mask_scaled = TF.resize(mask, [feats0[kk].shape[2], feats0[kk].shape[3]])
            diffs[kk] = (mask_scaled * (feats0[kk] - feats1[kk])) ** 2

        if (self.lpips):
            if (self.spatial):
                res = [upsample(self.lins[kk](diffs[kk]), out_HW=in0.shape[2:]) for kk in range(self.L)]
            else:
                res = [spatial_average(self.lins[kk](diffs[kk]), keepdim=True) for kk in range(self.L)]
        else:
            if (self.spatial):
                res = [upsample(diffs[kk].sum(dim=1, keepdim=True), out_HW=in0.shape[2:]) for kk in range(self.L)]
            else:
                res = [spatial_average(diffs[kk].sum(dim=1, keepdim=True), keepdim=True) for kk in range(self.L)]

        val = 0
        for l in range(self.L):
            val += res[l]

        if (retPerLayer):
            return (val, res)
        else:
            return val
