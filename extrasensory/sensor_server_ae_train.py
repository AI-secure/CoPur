import argparse
import os
import time
import numpy as np
import copy 
from utils.sensor_util import read_user_data, load_sensor_data, get_local_outputs
import tensorflow as tf
from models.AE import AE_NUS as AutoEncoder
 


def parse_path(args):
    SavedPaths=[]
    
    for i in range(args.num_img_clients):
        _dir= os.path.join(args.path_predix, 'img_{}_{}'.format(args.img_feature_div[i], args.img_feature_div[i+1]))
        SavedPaths.append(_dir)
    for i in range(args.num_text_clients):
        _dir= os.path.join(args.path_predix, 'sensor_{}_{}'.format(args.text_feature_div[i], args.text_feature_div[i+1] ))
        SavedPaths.append(_dir)
        
    _dir= os.path.join(args.path_predix,'final_server_AE{}'.format( args.use_ae))
    if not os.path.exists(_dir):
        os.makedirs(_dir)
    server_savedpath=_dir
    print(SavedPaths)
    print("server_savedpath", server_savedpath)
    return SavedPaths, server_savedpath

def create_models(args):
    from models.LocalModel import VFLPassiveModel
    from models.ServerModel import VFLActiveModelWithOneLayer
    
    LocalModels= []
    for i in range(args.num_clients):
        local_model = VFLPassiveModel(emb_dim=args.emb_dim)
        local_model.built = True
   
        local_model.load_weights(os.path.join(args.local_paths[i],'best_checkpoints')) 
        local_model.trainable = False
        LocalModels.append(local_model)

    active_model = VFLActiveModelWithOneLayer(emb_dim=args.emb_dim,class_num= args.num_class)
    
    return LocalModels,  active_model
    
def parse_command():
    parser = argparse.ArgumentParser()

    parser.add_argument('--emb_dim', type=int, default=32)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--num_class', type=int, default=1)
    parser.add_argument('--seed', type=int, default=0)
    
    parser.add_argument('--num_test_samples', type=int, default=10000)
    parser.add_argument('--num_train_samples', type=int, default=60000)
    
    parser.add_argument('--path_predix', type=str, default='sensor_results')
    parser.add_argument('--mode', type=str, default='normal',
                        choices=[
                            'normal',
              
                        ])
    parser.add_argument('--img_feature_div', nargs='+', type=int, default=[0 , 360, 634])
    parser.add_argument('--text_feature_div', nargs='+', type=int, default=[0, 500, 1000])
    
  
    parser.add_argument('--vis',  action='store_true')
    parser.add_argument('--use_ae', action='store_true')

    parser.add_argument('--lr', type=float, default=0.001)
  
    
    args = parser.parse_args()
    if args.seed is not None:
        import random
        random.seed(args.seed)
        tf.random.set_seed(args.seed)
    args.num_img_clients=0# len(args.img_feature_div)-1
    args.num_text_clients= len(args.text_feature_div)-1
    args.num_clients = args.num_img_clients+ args.num_text_clients
  
    args.path_predix_load = args.path_predix

    savefolder= '{}_{}i_{}t_try{}'.format(args.mode, args.num_img_clients, args.num_text_clients, args.seed)

    args.path_predix=os.path.join(args.path_predix, savefolder)
    local_paths, sever_path=  parse_path(args)
    args.local_paths= local_paths
    args.sever_path= sever_path
    args.top_k= ['buildings', 'grass', 'animal', 'water', 'person'] 
    print("save to", args.path_predix)


    return args 


args = parse_command()
if args.vis:
    from tensorboardX import SummaryWriter
    import datetime
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join(args.path_predix,current_time)
    args.train_writer = tf.summary.create_file_writer(log_dir)
else: 
    args.train_writer=None

(X1,Y1,M1,timestamps1,feature_names,feat_sensor_names,label_names) = read_user_data('example.csv')
sensors_to_use = ['Acc','Gyro','Magnet','WAcc','Compass','Loc','Aud','PS','LF','TS']
#
target_label ='SITTING'# 'FIX_walking'; # this is just code name for the cleaned version of the label 'Walking'
x_train, y_train= load_sensor_data(X1[0:2500,:],Y1[0:2500],M1[0:2500,:],feat_sensor_names,label_names,sensors_to_use,target_label,0)
x_test, y_test= load_sensor_data(X1[2500:-1,:],Y1[2500:-1],M1[2500:-1,:],feat_sensor_names,label_names,sensors_to_use,target_label,0)
LocalModels,  active_model= create_models(args)

if args.use_ae:
    autoencoder_model = AutoEncoder(out_dim= args.num_clients * args.emb_dim)
    autoencoder_model.built = True
    autoencoder_model.load_weights(os.path.join(args.path_predix,'ae_ckpt.tf'))


optimizer_server = tf.keras.optimizers.SGD(learning_rate=args.lr)


loss_object = tf.keras.losses.CategoricalCrossentropy()
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.BinaryAccuracy(name='train_accuracy')
test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.BinaryAccuracy(name='test_accuracy')
acc_train = []
acc_test = []
loss_train = []
loss_test = []
best_acc=0

def train(epoch):
    
    train_ds = tf.data.Dataset.from_tensor_slices(
        (x_train, y_train)).shuffle(y_train.shape[0]).batch(args.batch_size)
    server_step=0 
    for texts, labels in train_ds:
        server_step+=1
        _LocalOutputs= get_local_outputs(args.num_text_clients, texts, args.text_feature_div, LocalModels)
        
        if args.use_ae:
            rae_output,layer_output=autoencoder_model(tf.concat(_LocalOutputs,1))
            LocalOutputs = tf.split(rae_output, args.num_clients, 1 )
        else:
            LocalOutputs = _LocalOutputs
        with tf.GradientTape() as active_tape:
            active_output = active_model(LocalOutputs)
            loss = loss_object(labels, active_output)
        
        [active_model_gradients] = active_tape.gradient(loss, [active_model.trainable_variables])
        optimizer_server.apply_gradients(zip(active_model_gradients, active_model.trainable_variables))

        train_loss(loss)
        train_accuracy(labels, active_output)
        if args.vis:
            with args.train_writer.as_default():
                tf.summary.scalar('Server_train_loss', loss, step=server_step+epoch*len(train_ds))
    
def test(epoch):
    
    _LocalOutputs = get_local_outputs(args.num_text_clients, x_test, args.text_feature_div, LocalModels)
    if args.use_ae:
            rae_output,layer_output=autoencoder_model(tf.concat(_LocalOutputs,1))
            LocalOutputs = tf.split(rae_output, args.num_clients, 1 )
    else:
        LocalOutputs = _LocalOutputs
    test_output = active_model(LocalOutputs)
    test_accuracy(y_test, test_output)
    if args.vis:
        with args.train_writer.as_default():
            tf.summary.scalar('Server_test_loss', test_loss.result(), step=epoch )
            tf.summary.scalar('Server_test_acc', test_accuracy.result(), step=epoch )


for epoch in range(args.epochs):
    start= time.time() 
    train(epoch)
    test(epoch)
  
    end= time.time()
    template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}, Time: {}'
    print(template.format(epoch+1,
                        train_loss.result(),
                        train_accuracy.result()*100,
                        test_loss.result(),
                        test_accuracy.result()*100,
                        end-start
                        ))
    
    acc_train.append(train_accuracy.result())
    acc_test.append(test_accuracy.result())
    loss_train.append(train_loss.result())
    loss_test.append(test_loss.result())
    if best_acc <= test_accuracy.result():
        best_acc= test_accuracy.result()
        active_model.save_weights(os.path.join(args.sever_path, 'best_checkpoints'))
    train_loss.reset_states()
    train_accuracy.reset_states()
    test_loss.reset_states()
    test_accuracy.reset_states()
    active_model.save_weights(os.path.join(args.sever_path, 'checkpoints'))
    with open(os.path.join(args.path_predix, 'acc_test.txt'), "w") as outfile:
        outfile.write("\n".join("{:.4f}".format(item*100) for item in acc_test))

    with open(os.path.join(args.path_predix, 'acc_train.txt'), "w") as outfile:
        outfile.write("\n".join("{:.4f}".format(item*100) for item in acc_train))

