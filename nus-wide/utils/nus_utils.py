import numpy as np
from utils.nus_wide_data_util import get_labeled_data
import copy
from models.LocalModel import VFLPassiveModel
from models.ServerModel import VFLActiveModelWithOneLayer,QuanVFLActiveModelWithOneLayer

import os

def create_models(args,quan_server=False):
    if quan_server== False:
        LocalModels= []
        GradientsRes=[]
        for i in range(args.num_clients):
            local_model = VFLPassiveModel(emb_dim=args.emb_dim)
            LocalModels.append(local_model)
            GradientsRes.append(None)
        active_model = VFLActiveModelWithOneLayer(emb_dim=args.emb_dim,class_num= args.num_class)
        return LocalModels, GradientsRes,  active_model
    else:
        local_model = VFLPassiveModel(emb_dim=args.emb_dim)
        active_model = QuanVFLActiveModelWithOneLayer(emb_dim=args.emb_dim,class_num= args.num_class)
        return local_model, active_model

def load_models(args,quan_server=False):
    if quan_server==False:
        LocalModels= []
        for i in range(args.num_clients):
            local_model = VFLPassiveModel(emb_dim=args.emb_dim)
            local_model.built = True
            local_model.load_weights(os.path.join(args.local_paths[i],args.ckpt_name)) 
            LocalModels.append(local_model)
    
        active_model = VFLActiveModelWithOneLayer(emb_dim=args.emb_dim,class_num= args.num_class)
        active_model.built = True
        active_model.load_weights(os.path.join(args.sever_path,args.ckpt_name))
        return LocalModels,  active_model
    else:
        local_model = VFLPassiveModel(emb_dim=args.emb_dim)
        local_model.built = True
        local_model.load_weights(os.path.join(args.local_paths[0],args.ckpt_name)) 
        active_model = QuanVFLActiveModelWithOneLayer(emb_dim=args.emb_dim,class_num= args.num_class)
        active_model.built = True
        active_model.load_weights(os.path.join(args.sever_path,args.ckpt_name))
        return local_model,  active_model
    

def create_folder(args):
    if not os.path.exists(args.path_predix):
        os.makedirs(args.path_predix)
    print("save to", args.path_predix)

    SavedPaths=[]
    for i in range(args.num_img_clients):
        _dir= os.path.join(args.path_predix, 'img_{}_{}'.format(args.img_feature_div[i], args.img_feature_div[i+1]))
        if not os.path.exists(_dir):
            os.makedirs(_dir)
        SavedPaths.append(_dir)
    for i in range(args.num_text_clients):
        _dir= os.path.join(args.path_predix, 'text_{}_{}'.format(args.text_feature_div[i], args.text_feature_div[i+1]))
        if not os.path.exists(_dir):
            os.makedirs(_dir)
        SavedPaths.append(_dir)
        
    _dir= os.path.join(args.path_predix,'server')
    if not os.path.exists(_dir):
        os.makedirs(_dir)
    server_savedpath=_dir

    return SavedPaths, server_savedpath
    

def get_local_outputs(num_img_clients, images, img_feature_div, 
        num_text_clients, texts, text_feature_div, LocalModels):
    LocalOutputs =[] 
    for i in range(num_img_clients):
        images_div = images[:,img_feature_div[i]:img_feature_div[i+1]]
        LocalOutputs.append(LocalModels[i](images_div))
    for i in range(num_text_clients):
        text_div = texts[:,text_feature_div[i]:text_feature_div[i+1]]
        LocalOutputs.append(LocalModels[i+num_img_clients](text_div))

    return LocalOutputs



def load_data(args, test=False):
    
    if test==True:
        test_X_image, test_X_text, test_Y = get_labeled_data('', args.top_k, args.num_test_samples, 'Test')
        print("load test data done")
        x_test, y_test = (np.array(test_X_image).astype('float32'), np.array(test_X_text).astype('float32')), np.array(test_Y).astype('float32')
        
        return x_test, y_test
    else:
        train_X_image, train_X_text, train_Y = get_labeled_data('', args.top_k, args.num_train_samples, 'Train')
        print("load train data done")
        test_X_image, test_X_text, test_Y = get_labeled_data('', args.top_k, args.num_test_samples, 'Test')
        print("load test data done")
        x_train, x_test, y_train, y_test = (np.array(train_X_image).astype('float32'), np.array(train_X_text).astype('float32')), \
                                        (np.array(test_X_image).astype('float32'), np.array(test_X_text).astype('float32')), \
                                        np.array(train_Y).astype('float32'), np.array(test_Y).astype('float32')
        
        return x_train, x_test, y_train, y_test

