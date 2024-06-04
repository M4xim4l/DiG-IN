from .load_robust_model import load_madry_l2_with_cutout, load_madry_l2
from .load_shape_model import load_sin_model_with_cutout, load_sin_model
from .load_timm_model import load_timm_model_with_cutout, load_timm_model

def load_classifier(classifier_name, device, num_cutouts=0, noise_sd=0.):
    if classifier_name == 'resnet50_sin':
        if num_cutouts > 0:
            classifier = load_sin_model_with_cutout(0.3, num_cutouts, checkpointing=True,
                                                    noise_sd=noise_sd)
        else:
            classifier = load_sin_model()
    elif classifier_name == 'madry_l2':
        if num_cutouts > 0:
            classifier = load_madry_l2_with_cutout(cut_power=0.3, num_cutouts=num_cutouts, noise_sd=noise_sd)
        else:
            classifier = load_madry_l2()
    else:
        if num_cutouts > 0:
            classifier = load_timm_model_with_cutout(classifier_name, 0.3, num_cutouts, checkpointing=True,
                                                     noise_sd=noise_sd)
        else:
            classifier = load_timm_model(classifier_name)

    classifier = classifier.to(device)
    return classifier
