## Visual Concept Translator

[![arXiv](https://img.shields.io/badge/arXiv-2307.14352-b31b1b.svg)](https://arxiv.org/abs/2307.14352)

Visual Concept Translator (VCT) aims to achieve image tranlation with one-shot image guidance. Given only one reference image, VCT can automatically learn its dominant concepts and integrate them into input source image. The following examples show its performance. <br>
<br>

![VCT examples](assets/gallery.jpg?raw=true)
For each image group, the upper-left image is the source image, the lower-left image is the reference image, and the right part is the translated image. The VCT can be applied in many general image-to-image and style transfer tasks.


## Setup
To set up the environment, please run
```
conda create -n vct python=3.8
conda activate vct
pip install -r requirements.txt
```
We test our method on both Nvidia A30 and A100 GPU. However, it should work in any GPU with 24G memory.

## Usage
To use the VCT to image-to-image tasks, please run
```
accelerate launch main.py \
    --concept_image_dir="./examples/concept_image" \
    --content_image_dir="./examples/content_image" \
    --output_image_path="./outputs" \
    --initializer_token="girl" \
    --max_train_steps=500 \
    --concept_embedding_num=3 \
    --cross_attention_injection_ratio=0.2 \
    --self_attention_injection_ratio=0.9 \
    --use_l1
```
Please put your one-shot concept image into `concept_image_dir`, and any number of content images into `content_image_dir`. The translated images will be saved in `output_image_path`.

The `initializer_token` is used as the beginning of concept embeddings. The `max_train_steps` defines the training steps. For different concept, the optimal training step is also different, so you can adjust the `max_train_steps` to generate better results (always between 100 to 1000).

Inspired by [prompt-to-prompt](https://github.com/google/prompt-to-prompt), the VCT also applies the self-attention and cross-attention injection. Larger `self_attention_injection_ratio` or `cross_attention_injection_ratio` means more source contents preserved and less target concepts transferred. If you think the current results are not desired, please adjust these two parameters to achieve more content preservation or concept translation.

## Citation
If this code is useful for your work, please cite our paper:

```
@article{cheng2023general,
  title={General Image-to-Image Translation with One-Shot Image Guidance},
  author={Cheng, B. and Liu, Z. and Peng, Y. and Lin, Y.},
  journal={arXiv preprint arXiv:2307.14352},
  year={2023}
}
```





