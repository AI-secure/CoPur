#Please download NUS_WIDE dataset (https://lms.comp.nus.edu.sg/wp-content/uploads/2019/research/nuswide/NUS-WIDE.html) to folder NUS_WIDE

# The pre-trained Local feature extractors can be found in the folder `nus_results'
# Feature subspace learning
python nus_ae_train.py --ae_epochs 400 --ae_lr 0.001 --img_feature_div 0 360 634 --text_feature_div 0 500 1000
# Fusion center training
python nus_server_ae_train.py --use_ae --lr 0.001 --epochs 200 --img_feature_div 0 360 634 --text_feature_div 0 500 1000
# inference under untargeted attack
python nus_infer_attack.py --use_ae --L_lr 0.001 --img_feature_div 0 360 634 --text_feature_div 0 500 1000 --corruption_amp 10
# inference under targeted attack
python nus_infer_attack.py --use_ae --L_lr 0.001 --img_feature_div 0 360 634 --text_feature_div 0 500 1000 --num_test_samples 1000 --targeted 1
