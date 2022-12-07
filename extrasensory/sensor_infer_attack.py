
import argparse
import os
import time
import numpy as np
from utils.sensor_util import read_user_data, load_sensor_data, load_models, get_local_outputs
import tensorflow as tf
from math import ceil

from models.AE import AE_NUS as AutoEncoder
from utils.robustae import  inference_purify_miss


def parse_path(args):
    SavedPaths=[]
    for i in range(args.num_img_clients):
        _dir= os.path.join(args.path_predix, 'img_{}_{}'.format(args.img_feature_div[i], args.img_feature_div[i+1]))
       
        SavedPaths.append(_dir)
    for i in range(args.num_text_clients):
        _dir= os.path.join(args.path_predix, 'sensor_{}_{}'.format(args.text_feature_div[i], args.text_feature_div[i+1]))
        
        SavedPaths.append(_dir)
    if args.server_model_path == '':
        _dir= os.path.join(args.path_predix,'final_server_AE{}'.format( args.use_ae))
    else:
        _dir = os.path.join(args.path_predix,  args.server_model_path)
   
    server_savedpath=_dir

    return SavedPaths, server_savedpath
    

def parse_command():
    parser = argparse.ArgumentParser()

    parser.add_argument('--emb_dim', type=int, default=32)
    parser.add_argument('--batch_size', type=int, default=1)

    parser.add_argument('--num_class', type=int, default=1)
    parser.add_argument('--seed', type=int, default=0)
    
    parser.add_argument('--num_test_samples', type=int, default=10000)
    parser.add_argument('--num_train_samples', type=int, default=60000)
    parser.add_argument('--path_predix', type=str, default='sensor_results')
    parser.add_argument('--ckpt_name', type=str, default='best_checkpoints')
    parser.add_argument('--mode', type=str, default='normal',
                        choices=[
                            'normal',
                        ])
    parser.add_argument('--img_feature_div', nargs='+', type=int, default=[0, 360, 634])
    parser.add_argument('--text_feature_div', nargs='+', type=int, default=[0,  500, 1000])
    parser.add_argument('--use_ae', action='store_true')
    parser.add_argument('--purify_epochs', type=int, default=10)
    parser.add_argument('--initial_epochs', type=int, default=100)
    parser.add_argument('--L_lr', type=float, default=0.001)
    parser.add_argument('--L0_lr', type=float, default=0.1)
    parser.add_argument('--attack',  action='store_true')
    parser.add_argument('--tau',type=float, default=1000)
    parser.add_argument('--sigma',type=float, default=0.5)
    parser.add_argument('--corruption_amp',type=float, default=10)

    parser.add_argument('--server_model_path',type=str, default='')
    parser.add_argument('--num_noise', type=int, default=100)
    parser.add_argument('--attackers_index', nargs='+', type=int, default=[9])########
    parser.add_argument('--miss_index', nargs='+', type=int, default=[])  ########
    parser.add_argument('--PGDeps',type=float, default=0.5)
    parser.add_argument('--PGDiters', type=int, default=30)
    parser.add_argument('--targeted', type=int, default=0)

    args = parser.parse_args()
    if args.seed is not None:
        import random
        random.seed(args.seed)
        tf.random.set_seed(args.seed)
    
    args.num_img_clients=0 
    args.num_text_clients= len(args.text_feature_div)-1
    args.num_clients = args.num_img_clients+ args.num_text_clients
    args.num_attacker = len(args.attackers_index)##############
    print('attacker index', args.attackers_index)
    args.num_miss = len(args.miss_index)  ##############
    print('miss index', args.miss_index)

    args.observe_list=list(set(list(range(0,args.num_text_clients))) - set(args.miss_index))
    print('observed',args.observe_list)
    args.path_predix_load = args.path_predix
    savefolder= '{}_{}i_{}t_try{}'.format(args.mode, args.num_img_clients, args.num_text_clients, args.seed)
    args.path_predix=os.path.join(args.path_predix, savefolder)

    
    args.top_k= ['buildings', 'grass', 'animal', 'water', 'person'] 

    return args 

args = parse_command()

(X1,Y1,M1,timestamps1,feature_names,feat_sensor_names,label_names) = read_user_data('example.csv')

sensors_to_use = ['Acc','Gyro','Magnet','WAcc','Compass','Loc','Aud','PS','LF','TS']
#
target_label ='SITTING'# 'FIX_walking'; # this is just code name for the cleaned version of the label 'Walking'
x_train, y_train= load_sensor_data(X1[0:2500,:],Y1[0:2500],M1[0:2500,:],feat_sensor_names,label_names,sensors_to_use,target_label,0)
x_test, y_test= load_sensor_data(X1[2500:-1,:],Y1[2500:-1],M1[2500:-1,:],feat_sensor_names,label_names,sensors_to_use,target_label,0)

L_optimizer  =  tf.keras.optimizers.SGD(learning_rate=args.L0_lr)
L_optimizer2  =  tf.keras.optimizers.SGD(learning_rate=args.L_lr)
rae_loss_object=tf.keras.losses.MeanSquaredError()


loss_object = tf.keras.losses.CategoricalCrossentropy()

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.BinaryAccuracy(name='test_accuracy')

test_loss1 = tf.keras.metrics.Mean(name='test_loss1')
test_accuracy1 = tf.keras.metrics.BinaryAccuracy(name='test_accuracy1')

test_loss2 = tf.keras.metrics.Mean(name='test_loss2')
test_accuracy2 = tf.keras.metrics.BinaryAccuracy(name='test_accuracy2')

test_loss3 = tf.keras.metrics.Mean(name='test_loss3')
test_accuracy3 = tf.keras.metrics.BinaryAccuracy(name='test_accuracy3')

if args.use_ae:
    print("use AE")
    autoencoder_model = AutoEncoder(out_dim= args.num_clients * args.emb_dim)
    autoencoder_model.built = True
    autoencoder_model.load_weights(os.path.join(args.path_predix,'ae_ckpt.tf'))



def RandomizedSmooth(args, LocalOutputs, active_model,batch_size=100):
    counts = 0 
    x=np.concatenate(tuple(LocalOutputs), axis=1)
    num=args.num_noise
    for _ in range(ceil(num / batch_size)):
        this_batch_size = min(batch_size, num)
        num -= this_batch_size
        batch = x 
        batch = np.repeat(batch, this_batch_size, axis=0)
        noise = np.random.randn(*batch.shape) * args.sigma
        batch +=noise
        predictions = active_model(tf.split(batch , args.num_clients, 1 ))
        predictions1=np.sign(predictions) 
        counts += sum(predictions1)
        
    return counts 


def RandomizedSmooth_block(args, LocalOutputs, active_model, batch_size=100):
    counts = 0#np.zeros(args.num_class, dtype=int)
    x = np.concatenate(tuple(LocalOutputs), axis=1)
    num = args.num_noise
    for _ in range(ceil(num / batch_size)):
        this_batch_size = min(batch_size, num)
        num -= this_batch_size

        batch = x 
        batch = np.repeat(batch, this_batch_size, axis=0)
        noise = np.random.randn(batch.shape[0], args.emb_dim) * args.sigma

        for j in range(args.num_attacker):
            atk_index = args.attackers_index[j]
            batch[:, atk_index * args.emb_dim:(atk_index+1)* args.emb_dim] += noise  
        for j in range(args.num_miss):
            missing_index = args.miss_index[j]
            batch[:, missing_index * args.emb_dim:(missing_index+1)* args.emb_dim] += noise       


        predictions = active_model(tf.split(batch, args.num_clients, 1))
      
        predictions1=np.sign(predictions) 
        counts += sum(predictions1)
    

    return counts 

def count_arr(arr: np.ndarray, length: int) -> np.ndarray:
    counts = np.zeros(length, dtype=int)
    for idx in arr:
        counts[idx] += 1
    return counts


def create_adversarial_pattern(_LocalOutputs,args, labels, server_model,PGD=True):

    loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    if PGD == True:
        LocalOutputs = _LocalOutputs

        hadv = [_LocalOutputs[index] for index in args.attackers_index]
        h0=tf.convert_to_tensor(np.concatenate(tuple(hadv), axis=1),dtype=tf.float32)

        for i in range(args.PGDiters):

            with tf.GradientTape() as tape:
                tape.watch(h0)  # the adversarial's feature

                ADV_Outputs=tf.split(h0, args.num_attacker, 1)
                for j in range(args.num_attacker):
                    atk_index = args.attackers_index[j]
                    LocalOutputs[atk_index] = ADV_Outputs[j]

                output = server_model(LocalOutputs)
                loss = loss_object(1-labels, output)

            gradient = tape.gradient(loss, h0)

            h0 -= args.PGDeps * gradient

        return LocalOutputs



def test(_LocalModels,_active_model ):
    
    test_ds = tf.data.Dataset.from_tensor_slices(
        (x_test, y_test)).batch(args.batch_size)
    for texts, labels in test_ds:
        
        _LocalOutputs= get_local_outputs(args.num_text_clients, texts, args.text_feature_div, _LocalModels)

        
        for j in range(args.num_miss):
            missing_index = args.miss_index[j]
            _LocalOutputs[missing_index] = 0 * _LocalOutputs[missing_index]
        if args.targeted==1:
            _LocalOutputs=create_adversarial_pattern(_LocalOutputs, args, labels, _active_model, PGD=True)
        else:
            for j in range(args.num_attacker):
                atk_index = args.attackers_index[j]
                _LocalOutputs[atk_index] = -args.corruption_amp * _LocalOutputs[atk_index]
        print('corrupted -',args.corruption_amp)

        temp=np.concatenate(tuple(_LocalOutputs), axis=1)

        rae_output,layer_output=autoencoder_model(temp)

        if args.use_ae:
            LocalOutputs = inference_purify_miss(args, _LocalOutputs,autoencoder_model,L_optimizer,L_optimizer2,rae_loss_object, vis=False)
        else:
            LocalOutputs = _LocalOutputs
     
       
        test_output = _active_model(LocalOutputs)


        test_output1 = _active_model(_LocalOutputs)#no AE, just use corrupted h
        test_output2 = _active_model(tf.split(rae_output, args.num_clients, 1 ))# use DE(h)

        test_loss(loss_object(labels, test_output))
        test_accuracy(labels, test_output)


        test_loss1(loss_object(labels, test_output1))
        test_accuracy1(labels, test_output1)

        test_loss2(loss_object(labels, test_output2))
        test_accuracy2(labels, test_output2)
        label_RS=RandomizedSmooth(args, LocalOutputs, _active_model,100)


        test_accuracy3(labels,label_RS) 


local_paths, sever_path= parse_path(args)
args.local_paths= local_paths
args.sever_path= sever_path
LocalModels, active_model= load_models(args)
start= time.time()
test(LocalModels, active_model)

end= time.time()
template = '{} Test Loss: {}, Test Accuracy: {}, Test Accuracy3: {}, Time: {}'
print(template.format(args.path_predix, 
                    test_loss.result(),
                    test_accuracy.result()*100,
                    test_accuracy3.result()*100,
                    end-start
                    ))

test_loss.reset_states()
test_accuracy.reset_states()
