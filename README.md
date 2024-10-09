# DiG-IN: Diffusion Guidance for Investigating Networks - Uncovering Classifier Differences, Neuron Visualisations, and Visual Counterfactual Explanations [CVPR2024]


![Alt text](figures/teaser.jpg?raw=true "Title")

This is the official implementation to our CVPR 2024 paper: [DiG-IN: Diffusion Guidance for Investigating Networks - Uncovering Classifier Differences, Neuron Visualisations, and Visual Counterfactual Explanations](https://arxiv.org/abs/2311.17833)

### Setup
To create a conda environment for this project, please run:

```
conda create --name dig_in python=3.10
conda activate dig_in
conda install nvidia/label/cuda-12.1.0::cuda-nvcc
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```



To allow for [SAM-HQ](https://github.com/SysCV/sam-hq) segmentation, please download the model from [here](https://drive.google.com/file/d/1qobFYrI4eyIANfBSmYcGuWRaSIXfMOQ8/view)
and put it into a sam_hq folder in the root directory. 

All scripts are meant to be executed from the root directory. Most scripts support parallel execution on multiple GPUs via
torchrun. Make sure to specify the number of GPUs via: 
> --nproc-per-node N. 

You can specify the CUDA device ids via CUDA_VISIBLE_DEVICES, for example:

> CUDA_VISIBLE_DEVICES=0,2,7
> 

[//]: # (### Classifier Differences )

[//]: # ()
[//]: # (```)

[//]: # (CUDA_VISIBLE_DEVICES=... torchrun --nproc-per-node N --standalone src/imagenet_guided_generation.py classifier1= classifier2=)

[//]: # (```)


### Synthethic Neuron Activations using [CogVLM](https://github.com/THUDM/CogVLM)

First, you have to calculate the activations on the ImageNet train set. To do so, use:

```
python src/imagenet_cog_neuron_activations.py imagenet_folder=YOUR/PATH/TO/IMAGENET
```

Next, we use CogVLM to name the objects in the highest activating train images:

```
CUDA_VISIBLE_DEVICES=... torchrun --nproc-per-node N --standalone src/imagenet_cog_neuron_visualisation_stage1.py
```

Finally, you can generate visualisations using DiG-IN via:

```
CUDA_VISIBLE_DEVICES=... torchrun --nproc-per-node N --standalone src/imagenet_cog_neuron_visualisation_stage2.py imagenet_folder=YOUR/PATH/TO/IMAGENET
```

By default, all results will be saved in:

> ./output_cvpr/imagenet_cogvlm_neurons/

If you want to change the target neurons, you can do so via the argument

> target_neurons=[...]

### Img2Img: Automatic Captioning and Null-Text Inversion

For the following Img2Img tasks (Neuron Counterfactuals and UVCEs), we use [Null-Text Inversion](https://github.com/google/prompt-to-prompt/#null-text-inversion-for-editing-real-images) 
as initialization. Since Null-Text inversion requires a prompt for each image, we use [Open-Flamingo](https://github.com/mlfoundations/open_flamingo) to caption the images.

You can download [the captions used for our experiments](https://drive.google.com/file/d/1dN8OJC0zYvdVfLFcfNCWMj86oBab4-Iw/view?usp=sharing) and extract them to the default result folder: "./output_cvpr" or create them by yourself via:

ImageNet:
```
CUDA_VISIBLE_DEVICES=... torchrun --nproc-per-node N --standalone src/open_flamingo_imagenet.py imagenet_folder=YOUR/PATH/TO/IMAGENET
```

For CUB, Cars and Food-101 use the following code and set the dataset argument to food101, cars or cub:

```
CUDA_VISIBLE_DEVICES=... torchrun --nproc-per-node N --standalone src/open_flamingo_cub_cars_food.py dataset=DATASET dataset_folder=YOUR/DATASET/PATH
```

Once you have obtained the captions, you can invert the images via:

```
CUDA_VISIBLE_DEVICES=... torchrun --nproc-per-node N --standalone src/imagenet_inversion.py  guidance_scale=3.0 imagenet_folder=YOUR/PATH/TO/IMAGENET
```

For Food-101, CUB and Cars you can use:

```
CUDA_VISIBLE_DEVICES=... torchrun --nproc-per-node N --standalone src/inversion_cub_cars_food.py  guidance_scale=3.0  dataset=DATASET dataset_folder=YOUR/DATASET/PATH
```

If you are only interested in inverting fewer images per class, you can use the "images_per_class" argument. 


### Neuron Counterfactuals
To generate neuron counterfactuals starting from real images from the ImageNet validation set, you can run:


### Universal Visual Counterfactual Explanations (UVCE)
ImageNet:

```
CUDA_VISIBLE_DEVICES=... torchrun --nproc-per-node N --standalone src\uvces_imagenet.py regularizers=[latent_background_l2,px_background_l2] regularizers_ws=[25.0,250.0] imagenet_folder=YOUR/PATH/TO/IMAGENET
```

For CUB, Cars and Food:
```
CUDA_VISIBLE_DEVICES=... torchrun --nproc-per-node N --standalone src\uvces_cub_cars_food.py regularizers=[latent_background_l2,px_background_l2] regularizers_ws=[25.0,250.0] dataset=DATASET dataset_folder=YOUR/DATASET/PATH results_sub_folder=uvces random_images=True num_images=1000
```
