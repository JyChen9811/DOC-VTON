# DOC-VTON

## Official Codes for OccluMix: Towards De-Occlusion Virtual Try-on by Semantically-Guided Mixup (TMM 2022)


## Our Environment
anaconda3

pytorch 1.1.0

torchvision 0.3.0

cuda 9.0

cupy 6.0.0

opencv-python 4.5.1

4 v100 GPU for training; 1 v100 GPU for test

python 3.6

## Installation
conda create -n tryon python=3.6

source activate tryon     or     conda activate tryon

conda install pytorch=1.1.0 torchvision=0.3.0 cudatoolkit=9.0 -c pytorch

conda install cupy     or     pip install cupy==6.0.0

pip install opencv-python

## Test Script
python test_w_enhance.py --name demo --resize_or_crop None --batchSize 1 --gpu_ids=1

## Checkpoints Downlowd Address
[Checkpoints for Test](https://drive.google.com/file/d/1yj8khxliGEcEfFLhT_NJ3zUYN1UlV1t1/view?usp=sharing)

## Visual Comparison
We provide visualization results of various state-of-the-art methods (e.g. CP-VTON, CP-VTON+, ClothFlow, ACGPN, PF-AFN, DCTON, RT-VTON and our DOC-VTON) to facilitate your experimental comparisons.
[Different results](https://drive.google.com/file/d/1loiMvddHoRi7-eBz4qy45f3CgfFiGCyT/view?usp=sharing)
