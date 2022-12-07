# CoPur: Certifiably Robust Collaborative Inference via Feature Purification

This repository is the official implementation of "[CoPur: Certifiably Robust Collaborative Inference via Feature Purification](https://openreview.net/forum?id=r5rzV51GZx)". 

# Download and Installation
The required packages can be installed by:

```
pip install -r requirement.txt
```

For datasets:
- Download [NUS_WIDE dataset](https://lms.comp.nus.edu.sg/wp-content/uploads/2019/research/nuswide/NUS-WIDE.html) to folder `nus-wide/NUS_WIDE`
- The Extrasensory dataset is available in `extrasensory/example.csv`

# Usage



## Training and testing on NUS-WIDE dataset:
0. Enter the folder

```
cd nus-wide
```
1. The pre-trained Local feature extractors can be found in the folder ``nus_results``
2. Feature subspace learning

```
python nus_ae_train.py --ae_epochs 400 --ae_lr 0.001 --img_feature_div 0 360 634 --text_feature_div 0 500 1000
```

3. Fusion center training

```
python nus_server_ae_train.py --use_ae --lr 0.001 --epochs 200 --img_feature_div 0 360 634 --text_feature_div 0 500 1000
```

4. Inference under untargeted attack

```
python nus_infer_attack.py --use_ae --L_lr 0.001 --img_feature_div 0 360 634 --text_feature_div 0 500 1000 --corruption_amp 10
```

5. Inference under targeted attack

```
python nus_infer_attack.py --use_ae --L_lr 0.001 --img_feature_div 0 360 634 --text_feature_div 0 500 1000 --num_test_samples 1000 --targeted 1
```

## Training and testing on Extrasensory dataset:

0. Enter the folder

```
cd extrasensory
```
1. Local feature extractor training (trained models are saved in `nus_results' folder)
```
python sensor_normal_train.py --epochs 200 --emb_dim 32 --num_class 1 --lr 0.001 --text_feature_div 0 26 52 83 129 138 155 183 209 213 221
```
2. Feature subspace learning
```
python sensor_ae_train.py --ae_epochs 400 --ae_lr 0.001 --text_feature_div 0 26 52 83 129 138 155 183 209 213 221
```
3. Fusion center training
```
python sensor_server_ae_train.py --use_ae --epochs 200 --lr 0.001 --text_feature_div 0 26 52 83 129 138 155 183 209 213 221
```
4. Inference under untargeted attack
```
python sensor_infer_attack.py --use_ae  --L_lr 0.001 --text_feature_div 0 26 52 83 129 138 155 183 209 213 221 --corruption_amp 10 
```
5. Inference under targeted attack
```
python sensor_infer_attack.py --use_ae  --L_lr 0.001 --text_feature_div 0 26 52 83 129 138 155 183 209 213 221 --targeted 1 
```


## Citation
If you find our work useful in your research, please consider citing:
```
@inproceedings{
liu2022copur,
title={CoPur: Certifiably Robust Collaborative Inference via Feature Purification},
author={Jing Liu and Chulin Xie and Oluwasanmi O Koyejo and Bo Li},
booktitle={Advances in Neural Information Processing Systems},
editor={Alice H. Oh and Alekh Agarwal and Danielle Belgrave and Kyunghyun Cho},
year={2022},
url={https://openreview.net/forum?id=r5rzV51GZx}
}
```
