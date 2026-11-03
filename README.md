# :snake: [IEEE Access 2027 Submission] Repository for Bi-HiSS: Preserving 2D Spatial Dependencies under Mamba Serialization for Earth Observation

This repository contains the source code of Bi-HiSS Mamba 2D: A Spatially Enhanced Mamba Module for Earth Observation.

This model was run on one NVIDIA GH200 for each dataset. This can also be runned on one A100 for each dataset.

Apart from the PyYAML version in the requirements.txt, you can use the latest version of each of the mentioned modules. I have experimented with running on RTX 5080, and it did work. 

## Environment and Additional Configuration

It is sufficient to do

```bash
pip install -r requirements.txt
```

If this does not work, start by running this command on the requirements.txt file from  [Exact](https://github.com/MiSsU-HH/Exact/blob/master/requirements.txt) and slowly add each required module from our requirements.txt.

To train for semantic segmentation, execute the following command, replacing ** with the appropriate directory names:

```bash
python train_and_eval/segmentation_training_transf.py --config_file configs/**/CMTSViT.yaml --seeds * --gpu_ids 0,1
```
Seeds is an optional argument. you do not need to use it. However, if you do please use the seed table values.

The configuration script for each model:
1. MTSViT - Ablation for Feed Forward layer
2. CMTSViT - Main Bi-HiSS
3. DMTSViT - Ablation for other serialisation methods

Please see the DeepSatModels main [README](https://github.com/michaeltrs/DeepSatModels/blob/main/README.md) and [TSViT README](https://github.com/michaeltrs/DeepSatModels/blob/main/README_TSVIT.md) for further details.

**Note**: 
1. Mamba modules are crucial; ensure the wheel is downloaded by first installing PyTorch when the GPU is activated, so it can use CUDA. The rule of thumb is just to start with

```bash
pip install torch torchvision
```

Do not specify the version and then do 

```bash
pip install mamba-ssm
```

Please see the issues from mamba-ssm [GitHub](https://github.com/state-spaces/mamba/issues) to help you in downloading this. 

2. Although this paper talks about semantic segmentation, it can work for object classification. However, it is not configured properly in the initial script. Please see this [issue](https://github.com/michaeltrs/DeepSatModels/issues/18) for further details.

## Dataset Preparation

PASTIS - https://github.com/michaeltrs/DeepSatModels/issues/2

MTLCC (Germany) - https://github.com/michaeltrs/DeepSatModels/tree/main/data#mtlcc-dataset-germany

T31TFM_1618 - https://github.com/michaeltrs/DeepSatModels/tree/main/data#t31tfm_1618-dataset-france

## Random Seeds

| Dataset | Seed 1 | Seed 2 | Seed 3 |
|---|---|---|---|
| PASTIS (Fold 1) | 5 | 10 | 15 |
| T31TFM-1618 | 0 | 1 | 2 |

## Acknowledgement

This project is built upon [Exact](https://github.com/MiSsU-HH/Exact), [TSViT](https://github.com/michaeltrs/DeepSatModels) and [PASTIS](https://github.com/VSainteuf/pastis-benchmark).
