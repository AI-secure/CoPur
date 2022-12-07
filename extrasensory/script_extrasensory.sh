# Local feature extractor training (trained models are saved in `sensor_results' folder)
python sensor_normal_train.py --epochs 200 --emb_dim 32 --num_class 1 --lr 0.001 --text_feature_div 0 26 52 83 129 138 155 183 209 213 221
# Feature subspace learning
python sensor_ae_train.py --ae_epochs 400 --ae_lr 0.001 --text_feature_div 0 26 52 83 129 138 155 183 209 213 221
# Fusion center training
python sensor_server_ae_train.py --use_ae --epochs 200 --lr 0.001 --text_feature_div 0 26 52 83 129 138 155 183 209 213 221
# inference under untargeted attack
python sensor_infer_attack.py --use_ae  --L_lr 0.001 --text_feature_div 0 26 52 83 129 138 155 183 209 213 221 --corruption_amp 10 
# inference under targeted attack
python sensor_infer_attack.py --use_ae  --L_lr 0.001 --text_feature_div 0 26 52 83 129 138 155 183 209 213 221 --targeted 1 
