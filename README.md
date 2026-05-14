# :snake: [BMVC 2026 Submission] Repository for Bi-HiSS Mamba 2D: A Spatially Enhanced Mamba Module for Earth Observation

This repository contains the source code of Bi-HiSS Mamba 2D: A Spatially Enhanced Mamba Module for Earth Observation

## Environment and Additional Configuration

It is sufficient to do

```bash
pip install -r requirements.txt
```

If this does not work, start by running this command on the requirements.txt file from  [Exact](https://github.com/MiSsU-HH/Exact/blob/master/requirements.txt) and slowly add each required module using the code.

To train for semantic segmentation, execute the following command, replacing ** with the appropriate directory names:

```bash
python train_and_eval/segmentation_training_transf.py --config_file configs/**/CMTSViT.yaml --gpu_ids 0,1
```
The configuration script for each model:
1. MTSViT - Ablation for Feed Forward layer
2. CMTSViT - Main Bi-HiSS Mamba 2D
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

2. Apart from the PyYAML version in the requirements.txt, you can use the latest version of each of the mentioned modules. I have experimented with running on RTX 5080, and it did work.

3. Although this paper talks about semantic segmentation, it can work for object classification. However, it is not configured properly in the initial script. Please see this [issue](https://github.com/michaeltrs/DeepSatModels/issues/18) for further details.

## Dataset Preparation

PASTIS - https://github.com/michaeltrs/DeepSatModels/issues/2

MTLCC (Germany) - https://github.com/michaeltrs/DeepSatModels/tree/main/data#mtlcc-dataset-germany

T31TFM_1618 - https://github.com/michaeltrs/DeepSatModels/tree/main/data#t31tfm_1618-dataset-france

## Acknowledgement

This project is built upon [Exact](https://github.com/MiSsU-HH/Exact), [TSViT](https://github.com/michaeltrs/DeepSatModels) and [PASTIS](https://github.com/VSainteuf/pastis-benchmark).
