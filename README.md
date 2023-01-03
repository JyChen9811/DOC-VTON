# DOC-VTON

## Official Codes for OccluMix: Towards De-Occlusion Virtual Try-on by Semantically-Guided Mixup (TMM 2022)

## Visual Comparison
We provide visualization results of various state-of-the-art methods (e.g. CP-VTON, CP-VTON+, ClothFlow, ACGPN, PF-AFN, DCTON, RT-VTON and our DOC-VTON) to facilitate your experimental comparisons.

[Different results](https://drive.google.com/file/d/1loiMvddHoRi7-eBz4qy45f3CgfFiGCyT/view?usp=sharing)


| Model             | Published                                    | Code                                                         | FID                                                       |
| ----------------- | -------------------------------------------- | :----------------------------------------------------------- | ------------------------------------------------------------ |
| CP-VTON            | [ECCV2018](https://arxiv.org/pdf/1807.07688.pdf) | [Code](https://github.com/sergeywong/cp-vton)                                                     | 24.43                    |
| CP-VTON+ | [CVPRW2020](https://minar09.github.io/cpvtonplus/cvprw20_cpvtonplus.pdf) | [Code](https://github.com/minar09/cp-vton-plus)    | 21.08 |
| ClothFlow            | [ICCV2019](https://openaccess.thecvf.com/content_ICCV_2019/papers/Han_ClothFlow_A_Flow-Based_Model_for_Clothed_Person_Generation_ICCV_2019_paper.pdf) | -           | 14.43 |
| ACGPN          | [CVPR2020](https://openaccess.thecvf.com/content_CVPR_2020/papers/Yang_Towards_Photo-Realistic_Virtual_Try-On_by_Adaptively_Generating-Preserving_Image_Content_CVPR_2020_paper.pdf) | [Code](https://github.com/switchablenorms/DeepFashion_Try_On) | 15.67                     |
| DCTON       | [CVPR2021](https://openaccess.thecvf.com/content/CVPR2021/papers/Ge_Disentangled_Cycle_Consistency_for_Highly-Realistic_Virtual_Try-On_CVPR_2021_paper.pdf)    | [Code](https://github.com/ChongjianGE/DCTON)            | 14.82 |
| PF-AFN             | [CVPR2021](https://openaccess.thecvf.com/content/CVPR2021/papers/Ge_Parser-Free_Virtual_Try-On_via_Distilling_Appearance_Flows_CVPR_2021_paper.pdf)    | [Code](https://github.com/geyuying/PF-AFN)                        | 10.09                             |
| RT-VTON            | [CVPR2022](https://openaccess.thecvf.com/content/CVPR2022/papers/Yang_Full-Range_Virtual_Try-On_With_Recurrent_Tri-Level_Transform_CVPR_2022_paper.pdf)    | -                        | 11.66                             |
| DOC-VTON            | TMM2022    | [Code](https://github.com/JyChen9811/DOC-VTON)                        | 9.54                           |

------

## Auxiliary test data
We provide [densepose results](https://drive.google.com/file/d/1LiiuKvNLTtmQ3WKSxpLlP8NL10fO04UT/view?usp=sharing) of VITON test imgs.


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


