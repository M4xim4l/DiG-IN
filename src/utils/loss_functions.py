import torch
import torch.nn.functional as F
from lpips import LPIPS


def get_loss_function(loss, classifier, classifier2=None):
    if classifier2 is None:
        if loss in ['log_conf', 'NLL', 'CE']:
            def loss_function(image, targets, augment=True):
                class_out = classifier(image, augment=augment)
                target_ = torch.zeros(class_out.shape[0], dtype=torch.long, device=image.device).fill_(targets)
                loss = F.cross_entropy(class_out, target_, reduction='mean')
                return loss
        elif loss == 'conf':
            def loss_function(image, targets, augment=True):
                class_out = classifier(image, augment=augment)
                probs = torch.softmax(class_out, dim=1)
                log_conf = probs[:, targets]
                loss = -log_conf.mean()
                return loss
        else:
            raise NotImplementedError()
    else:
        if loss == 'conf':
            def loss_function(image, targets, augment=True):
                class_out1 = classifier(image, augment=augment)
                probs1 = torch.softmax(class_out1, dim=1)
                conf1 = probs1[:, targets]

                class_out2 = classifier2(image, augment=augment)
                probs2 = torch.softmax(class_out2, dim=1)
                conf2 = probs2[:, targets]

                loss = -conf1.mean() + conf2.mean()
                return loss
        elif loss == 'log_conf':
            def loss_function(image, targets, augment=True):
                class_out1 = classifier(image, augment=augment)
                log_probs1 = torch.log_softmax(class_out1, dim=1)
                log_conf1 = log_probs1[:, targets]

                class_out2 = classifier2(image, augment=augment)
                log_probs2 = torch.log_softmax(class_out2, dim=1)
                log_conf2 = log_probs2[:, targets]

                loss = (-log_conf1 + log_conf2).mean()
                return loss
        elif loss == 'logits':
            def loss_function(image, targets, augment=True):

                class_out1 = classifier(image, augment=augment)

                target_mask = torch.zeros_like(class_out1)
                target_mask[:, targets] = -1e12

                target_logits1 = class_out1[:, targets]
                max_logits1 = torch.max(class_out1 - target_mask, dim=1)[0]

                class_out2 = classifier2(image, augment=augment)
                target_logits2 = class_out2[:, targets]
                max_logits2 = torch.max(class_out2 - target_mask, dim=1)[0]

                loss = - (target_logits1 - max_logits1).mean() + (target_logits2 - max_logits2).mean()
                return loss
        else:
            raise NotImplementedError()

    return loss_function


def get_feature_loss_function(loss, classifier, layer_activations):
    if loss in ['neuron_activation', 'neg_neuron_activation']:
        neg_factor = -1. if 'neg' in loss else 1.
        def loss_function(image, target_class_target_neuron, augment=True, return_act=False):
            target_class, target_neuron = target_class_target_neuron
            if len(image.shape) == 3:
                image = image[None, :]
            bs = image.shape[0]
            _ = layer_activations(image, augment=augment)
            act = layer_activations.activations[0][0]
            neuron = act[:, target_neuron].mean()
            loss = neg_factor * -neuron

            if return_act:
                return loss, neuron
            return loss
    elif loss in ['neuron_activation_plus_log_confidence', 'neg_neuron_activation_plus_log_confidence']:
        neg_factor = -1. if 'neg' in loss else 1.
        def loss_function(image, target_class_target_neuron, augment=True, return_act=False):
            target_class, target_neuron = target_class_target_neuron
            if len(image.shape) == 3:
                image = image[None, :]
            bs = image.shape[0]
            _ = layer_activations(image, augment=augment)
            act = layer_activations.activations[0][0]
            neuron = act[:, target_neuron].mean()
            loss = neg_factor * 10 * -neuron

            class_out = classifier(image, augment=augment)
            log_probs = torch.log_softmax(class_out, dim=1)
            log_conf = log_probs[:, target_class]
            loss = loss - 0.5 * log_conf.mean()
            if return_act:
                return loss, neuron

            return loss
    else:
        raise NotImplementedError()

    return loss_function

def get_clip_loss(loss, img_encoder, factor=1.):
    requires_negative_prompt = False
    if loss == 'inner_product':
        def loss_function(image, target_text_encoding, augment=True):
            img_emb = img_encoder(image, augment=augment)
            return -(factor * img_emb.float() @ target_text_encoding.transpose(1, 0)).mean()
    elif loss == 'spherical_distance':
        def loss_function(image, target_text_encoding, augment=True):
            img_emb = img_encoder(image, augment=augment)
            return factor * (img_emb - target_text_encoding).norm(dim=-1).div(2).arcsin().pow(2).mul(2).mean()
    elif loss == 'cross_entropy':
        def loss_function(image, positive_negative_text_encodings, augment=True):
            positive_text_encoding = positive_negative_text_encodings[0]
            negative_text_encoding = positive_negative_text_encodings[1]

            img_emb = img_encoder(image, augment=augment)
            positive_logits = factor * img_emb.float() @ positive_text_encoding.transpose(1, 0)
            negative_logits = factor * img_emb.float() @ negative_text_encoding.transpose(1, 0)

            logits = factor * torch.stack([positive_logits.squeeze(dim=1), negative_logits.squeeze(dim=1)], dim=1)
            target = torch.zeros(len(logits), dtype=torch.long, device=image.device)
            return F.cross_entropy(logits, target)

        requires_negative_prompt = True
    else:
        raise NotImplementedError(f'Loss {loss} not supported')

    return loss_function, requires_negative_prompt

##########################

def calculate_confs(classifier, imgs, device, target_class=None, return_predictions=False):
    confs = torch.zeros(imgs.shape[0])
    preds = torch.zeros(imgs.shape[0], dtype=torch.long)
    with torch.no_grad():
        for i in range(imgs.shape[0]):
            img = torch.clamp(imgs[i,:].to(device)[None, :], 0, 1)
            out = classifier(img, augment=False)
            probs = torch.softmax(out, dim=1)
            _, pred = torch.max(probs, dim=1)
            if target_class is None:
                conf, _ = torch.max(probs, dim=1)
            else:
                conf = probs[:, target_class]
            confs[i] = conf
            preds[i] = pred

    if return_predictions:
        return confs, preds
    else:
        return confs


def calculate_conf_diff(classifier, classifier_2, imgs, device, target_class=None):
    conf_diff = torch.zeros(imgs.shape[0])
    with torch.no_grad():
        for i in range(imgs.shape[0]):
            img = imgs[i,:].to(device)[None, :]
            out = classifier(img, augment=False)
            probs = torch.softmax(out, dim=1)
            if target_class is None:
                conf, target_class = torch.max(probs, dim=1)
            else:
                conf = probs[:, target_class]

            out_2 = classifier_2(img, augment=False)
            probs_2 = torch.softmax(out_2, dim=1)
            conf2 = probs_2[:, target_class]

            conf_diff[i]= conf - conf2

    return conf_diff


def calculate_neuron_activations(classifier, layer_activations, imgs, device, target_neuron, loss):
    losses = torch.zeros(imgs.shape[0])
    activations = torch.zeros(imgs.shape[0])
    loss_function = get_feature_loss_function(loss, classifier, layer_activations)
    with torch.no_grad():
        for i in range(imgs.shape[0]):
            image = imgs[i,:].to(device)[None, :]
            neg_loss, act = loss_function(image, [None, target_neuron], augment=False, return_act=True)
            losses[i] = -neg_loss
            activations[i] = act
    return losses, activations

def calculate_lp_distances(imgs, starting_imgs, ps = (1., 2.)):
    assert len(imgs) == len(starting_imgs)
    distances = {p: [] for p in ps}
    for i in range(len(imgs)):
        img = imgs[i].view(-1)
        start_img = starting_imgs[i].view(-1).to(img.device)

        for p in ps:
            d_i = torch.norm(img - start_img, p=p).item()
            distances[p].append(d_i)
    return distances


def calculate_lpips_distances(imgs, starting_imgs):
    assert len(imgs) == len(starting_imgs)
    distances = []

    loss_fn_alex = None
    device = None

    for i in range(len(imgs)):
        img = imgs[i]
        if img.dim() == 3:
            img = img[None, :, :, :]

        if loss_fn_alex is None:
            device = img.device
            loss_fn_alex = LPIPS(net='alex').to(device)
        else:
            img = img.to(device)

        start_img = starting_imgs[i]
        if start_img.dim() == 3:
            start_img = start_img[None, :, :, :].to(device)

        d = loss_fn_alex(img, start_img, normalize=True).mean().item()
        distances.append(d)
    return distances

def make_loss_dict(loss_function, name, weight):
    loss_dict = {
        'loss': loss_function,
        'name': name,
        'weight': weight
    }
    return loss_dict

