import torch
from torch import nn
from torch.nn import functional as F
from torch.utils import checkpoint as checkpoint
from torchvision.transforms import functional as TF
from torchvision.utils import save_image

IMAGENET_DEFAULT_MEAN = [0.485, 0.456, 0.406]
IMAGENET_DEFAULT_STD = [0.229, 0.224, 0.225]
# class NormalizationAndCutoutWrapper(torch.nn.Module):
#     def __init__(self, model, size,  mean=torch.tensor([0.0, 0.0, 0.0]), std=torch.tensor([1.0, 1.0, 1.0]),
#                  noise_sd=0, noise_schedule='constant', cut_power=1.0, num_cutouts=16, batches=1, checkpointing=False):
#         super().__init__()
#
#         self.model = model
#
#         mean = mean[..., None, None]
#         std = std[..., None, None]
#
#         self.size = size
#         self.cutout = MakeCutouts(size, cut_power)
#         self.noise_sd = noise_sd
#         self.noise_schedule = noise_schedule
#         self.num_cutouts = num_cutouts
#         self.batches = batches
#
#         self.train(model.training)
#
#         self.model = model
#         self.register_buffer("mean", mean)
#         self.register_buffer("std", std)
#         self.checkpointing = checkpointing
#     @property
#     def return_layers(self):
#         """I'm the 'x' property."""
#         print("getter of x called")
#         return self.model.model.model._return_layers
#
#     @return_layers.setter
#     def return_layers(self, value):
#         print("setter of x called")
#         self.model.model.model._return_layers = value
#
#     @return_layers.deleter
#     def return_layers(self):
#         print("deleter of x called")
#         del self.model.model.model._return_layers
#
#     def forward(self, x, *args, augment=True):
#         if not augment:
#            x = TF.resize(x, self.size, interpolation=TF.InterpolationMode.BILINEAR)
#            return self.model((x - self.mean) / self.std, *args)
#
#         batches = []
#         for _ in range(self.batches):
#             x_augm = self.cutout(x, self.num_cutouts)
#             if self.noise_sd > 0:
#                 if self.noise_schedule == 'linear':
#                     sd = torch.linspace(0.001, self.noise_sd, self.num_cutouts, device=x.device)[:, None, None, None]
#                 elif self.noise_schedule == 'constant':
#                     sd = self.noise_sd
#                 else:
#                     raise NotImplementedError
#                 x_augm = x_augm + sd * torch.randn_like(x)
#
#             x_augm = (x_augm - self.mean) / self.std
#             if self.checkpointing:
#                 out = checkpoint.checkpoint(self.model, x_augm, *args)
#             else:
#                 out = self.model(x_augm)
#             return out
#
#         out = torch.cat(batches, dim=0)
#         return out
#
#     def state_dict(self, *args, **kwargs):
#         return self.model.state_dict(*args, **kwargs)

class NormalizationAndCutoutWrapper(torch.nn.Module):
    def __init__(self, model, size,  mean=torch.tensor([0.0, 0.0, 0.0]), std=torch.tensor([1.0, 1.0, 1.0]),
                 noise_sd=0, noise_schedule='constant', noise_descending_steps=None, cut_power=1.0, num_cutouts=16,
                 batches=1, checkpointing=False,
                 ):
        super().__init__()

        self.model = model

        mean = mean[..., None, None]
        std = std[..., None, None]

        self.size = size
        self.cutout = MakeCutouts(size, cut_power)
        self.num_cutouts = num_cutouts
        self.train(model.training)

        self.model = model
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)
        self.checkpointing = checkpointing

        self.noise_sd = noise_sd
        self.noise_schedule = noise_schedule
        if self.noise_schedule == 'descending':
            if checkpointing:
                noise_descending_steps = noise_descending_steps * 2 * batches
            self.current_noise_step = 0
            self.descending_noise_sds = torch.linspace(self.noise_sd, 0, noise_descending_steps)
        else:
            self.current_noise_step = None
            self.descending_noise_sds = None



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

    def forward(self, x, *args, augment=False, **kwargs):
        def chkpt_function(x):
            if augment:
                if self.num_cutouts > 0:
                    x = self.cutout(x, self.num_cutouts)
                else:
                    x = TF.resize(x, self.size, interpolation=TF.InterpolationMode.BILINEAR)
                if self.noise_sd > 0:
                    if self.noise_schedule == 'descending':
                        noise_sd = self.descending_noise_sds[self.current_noise_step]
                        self.current_noise_step += 1
                        if self.current_noise_step == len(self.descending_noise_sds):
                            self.current_noise_step = 0
                            print('Resetting noise schedule')
                    else:
                        noise_sd = self.noise_sd
                    x = x + noise_sd * torch.randn_like(x)
                    #save_image(x.cpu(), f'augm_{self.current_noise_step}.png')
            else:
                x = TF.resize(x, self.size, interpolation=TF.InterpolationMode.BILINEAR)
            return self.model((x - self.mean) / self.std, *args, **kwargs)

        if self.checkpointing:
            out = checkpoint.checkpoint(chkpt_function, x, use_reentrant=False)
        else:
            out = chkpt_function(x)
        return out

    def state_dict(self, *args, **kwargs):
        return self.model.state_dict(*args, **kwargs)

class NormalizationAndResizeWrapper(torch.nn.Module):
    def __init__(self, model, size, mean=torch.tensor([0.0, 0.0, 0.0]), std=torch.tensor([1.0, 1.0, 1.0])):
        super().__init__()

        self.model = model

        mean = mean[None,:, None, None]
        std = std[None,:, None, None]

        self.size = size
        self.train(model.training)

        self.model = model
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

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

    def forward(self, x, *args, augment=False, **kwargs):
        if self.size is not None:
            x = TF.resize(x, self.size, interpolation=TF.InterpolationMode.BILINEAR)

        return self.model((x - self.mean) / self.std, *args, **kwargs)

    def state_dict(self, *args, **kwargs):
        return self.model.state_dict(*args, **kwargs)


class NormalizationAndRandomNoiseWrapper(torch.nn.Module):
    def __init__(self, model, size, num_samples, noise_sd, noise_schedule='constant', batches=1, mean=torch.tensor([0.0, 0.0, 0.0]), std=torch.tensor([1.0, 1.0, 1.0]), checkpointing=False):
        super().__init__()

        self.model = model
        self.num_samples = num_samples
        self.noise_sd = noise_sd
        self.noise_schedule = noise_schedule
        self.train(model.training)

        self.size = size
        self.model = model
        self.batches = batches

        mean = mean[..., None, None]
        std = std[..., None, None]
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)
        self.checkpointing = checkpointing

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

    def forward(self, x, *args, augment=False):
        if not augment:
           x = TF.resize(x, self.size, interpolation=TF.InterpolationMode.BILINEAR)
           return self.model((x - self.mean) / self.std, *args)

        batches = []
        for _ in range(self.batches):
            x_augm = self.cutout(x, self.num_cutouts)
            if self.noise_sd > 0:
                if self.noise_schedule == 'linear':
                    sd = torch.linspace(0.001, self.noise_sd, self.num_cutouts, device=x.device)[:, None, None, None]
                elif self.noise_schedule == 'constant':
                    sd = self.noise_sd
                else:
                    raise NotImplementedError
                x_augm = x_augm + sd * torch.randn_like(x)

            x_augm = (x_augm - self.mean) / self.std
            if self.checkpointing:
                out = checkpoint.checkpoint(self.model, x_augm, *args)
            else:
                out = self.model(x_augm)
            batches.append(out)

        out = torch.cat(batches, dim=0)
        return out

    def state_dict(self, *args, **kwargs):
        return self.model.state_dict(*args, **kwargs)


class MakeCutouts(nn.Module):
    def __init__(self, cut_size, cut_power=1.0):
        super().__init__()

        self.cut_size = cut_size
        self.cut_power = cut_power

    def forward(self, pixel_values, num_cutouts):
        sideY, sideX = pixel_values.shape[2:4]
        max_size = min(sideX, sideY)
        min_size = min(sideX, sideY, self.cut_size)
        cutouts = []
        for _ in range(num_cutouts):
            size = int(torch.rand([]) ** self.cut_power * (max_size - min_size) + min_size)
            offsetx = torch.randint(0, sideX - size + 1, ())
            offsety = torch.randint(0, sideY - size + 1, ())
            cutout = pixel_values[:, :, offsety : offsety + size, offsetx : offsetx + size]
            cutouts.append(F.adaptive_avg_pool2d(cutout, self.cut_size))
        return torch.cat(cutouts)
