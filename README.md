# Remote Sensing Image Super-Resolution for Imbalanced Textures: A Texture-Aware Diffusion Framework (CVPR2026)

<a href='https://arxiv.org/abs/2604.13994'><img src='https://img.shields.io/badge/Paper-Arxiv-b31b1b.svg'></a > &nbsp;&nbsp;

This is the official PyTorch codes for the paper:
>**Remote Sensing Image Super-Resolution for Imbalanced Textures: A Texture-Aware Diffusion Framework**<br>  
Enzhuo, Zhang, Sijie, Zhao,  Dilxat Muhtar, Zhenshi Li, [Xueliang Zhang](https://sgos.nju.edu.cn/zxl1/list.htm), [Pengfeng Xiao](https://sgos.nju.edu.cn/xpf/list.htm), <br>
> Nanjing University


:star: If Texadiff is helpful to your images or projects, please help star this repo. Thank you! :point_left:

## :runner: TODO
- [x] Release Checkpoints
- [x] Release inference code
- [x] Release training code 

## :wrench: Dependencies and Installation

1. Clone repo

```bash
git clone https://github.com/ZezFuture/TexAdiff.git
cd Texadiff
```

2. Install packages
```bash
conda create -n texadiff python=3.10 -y
conda activate texadiff
pip install --upgrade pip
pip install -r requirements.txt
```

## :surfer: Train

**Step 1: Download Checkpoints**


- Download the [[stable-diffusion-xl-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)] checkpoints and place it in the `checkpoints/stable-diffusion-xl-base-1.0` directory.
- Download the [[sdxl-vae-fp16-fix](https://huggingface.co/madebyollin/sdxl-vae-fp16-fix)] checkpoints and place it in the `checkpoints/sdxl-vae-fp16-fix` directory.
- Download the [[our model](https://huggingface.co/ZEZRS/TexADiff/tree/main)] checkpoints and place them in the following directories: `checkpoints/TexADiff` .


**Step 2: Prepare training data**

Place the high-quality remote sensing images and their corresponding text prompts in the data directory. The expected directory structure is as follows:

```text
data/train/
├── image/
│   ├── xxx.png
│   ├── xxx.jpg
│   └── ...
└── prompt/
    ├── xxx.txt
    ├── xxx.txt
    └── ...
```

For training, we use **MillionAID** images with resolutions ranging from **512 to 2048 pixels**, together with the **LoveDA-Train** and **DOTA-Train** datasets.

The corresponding text prompts are mainly collected from the **RS5M** dataset. For images without available captions, we use **Qwen2.5-VL** to generate additional descriptions.

You can either preprocess these datasets by yourself or directly download our preprocessed training data from Hugging Face [[LHRS/diffdata](https://huggingface.co/datasets/LHRS/diffdata/tree/main/img512tar)]


**Step 3: Train SwinIR**

You can use the **BasicSR** framework to train **SwinIR**. Note that the degradation process used in our setting differs from the standard **bicubic downsampling** originally supported by SwinIR. We therefore recommend preprocessing the training data using the degradation pipeline and configurations described in our paper, or modifying the BasicSR codebase to generate the degraded LR images online during training. Alternatively, you can directly use our pre-trained **SwinIR checkpoint**.


**Step 4: Train RTDM Model**


You can first modify the `bash/train_rtdm.sh` and then use the following command to start training.

```bash
bash bash/train_rtdm.sh
```

**Step 5: Train SR Model**

You can first modify the `bash/train_sr.sh` and then use the following command to start training.

```bash
bash bash/train_sr.sh
```

## :surfer: Quick Inference

**Step 1: Download Checkpoints**

- Download the [[our model](https://huggingface.co/ZEZRS/TexADiff/tree/main)] checkpoints and place them in the following directories: `checkpoints/TexADiff` .
- Download the [[stable-diffusion-xl-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)] checkpoints and place it in the `checkpoints/stable-diffusion-xl-base-1.0` directory.
- Download the [[sdxl-vae-fp16-fix](https://huggingface.co/madebyollin/sdxl-vae-fp16-fix)] checkpoints and place it in the `checkpoints/sdxl-vae-fp16-fix` directory.

**Step 2: Prepare testing data**

Place low-quality remote sensing images in  the `test_data` directory

**Step 3: Running testing command**

You can modify `bash/run.sh` according to your needs. If you have fully completed the previous steps, you can also directly run the following command.

```bash
bash bash/run.sh
```

**Step 4: Check the results**

The processed results will be saved in the `[output]` directory.


<!-- ## :book: Citation

If you find our repo useful for your research, please consider citing our paper:

```bibtex

``` -->

## Acknowledgments
Our project is based on [diffusers](https://github.com/huggingface/diffusers), [controlnext](https://github.com/JIA-Lab-research/ControlNeXt) and [pasd](https://github.com/yangxy/PASD).


## :postbox: Contact

For technical questions, please contact `ezzhang03[AT]gmail.com`
