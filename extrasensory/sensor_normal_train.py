import argparse
import os
import time
import numpy as np
from utils.sensor_util import read_user_data, load_sensor_data, create_folder, create_models, get_local_outputs
import tensorflow as tf

def parse_command():
    parser = argparse.ArgumentParser()

    parser.add_argument('--emb_dim', type=int, default=60)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--save_epochs', nargs='+', type=int, default=[30,40,50])
    parser.add_argument('--num_class', type=int, default=1)
    parser.add_argument('--seed', type=int, default=0)

    parser.add_argument('--num_test_samples', type=int, default=10000)
    parser.add_argument('--num_train_samples', type=int, default=60000)

    parser.add_argument('--path_predix', type=str, default='sensor_results')
    parser.add_argument('--mode', type=str, default='normal',
                        )
    parser.add_argument('--img_feature_div', nargs='+', type=int, default=[0, 360, 634])
    parser.add_argument('--text_feature_div', nargs='+', type=int, default=[0, 500, 1000])
    parser.add_argument('--lr', type=float, default=0.005)


    args = parser.parse_args()
    if args.seed is not None:
        import random
        random.seed(args.seed)
        tf.random.set_seed(args.seed)
    args.num_img_clients= 0
    args.num_text_clients= len(args.text_feature_div)-1
    args.num_clients = args.num_img_clients+ args.num_text_clients

    savefolder= '{}_{}i_{}t_try{}'.format(args.mode, args.num_img_clients,args.num_text_clients ,  args.seed)
    args.path_predix=os.path.join(args.path_predix, savefolder)
    local_paths, sever_path=  create_folder(args)
    args.local_paths= local_paths
    args.sever_path= sever_path
    args.top_k= ['buildings', 'grass', 'animal', 'water', 'person']
    print("will save in epochs:", args.save_epochs)


    return args


args = parse_command()


(X1,Y1,M1,timestamps1,feature_names,feat_sensor_names,label_names) = read_user_data('example.csv')

sensors_to_use = ['Acc','Gyro','Magnet','WAcc','Compass','Loc','Aud','PS','LF','TS']
#
target_label ='SITTING'# 'FIX_walking'; # this is just code name for the cleaned version of the label 'Walking'
x_train, y_train= load_sensor_data(X1[0:2500,:],Y1[0:2500],M1[0:2500,:],feat_sensor_names,label_names,sensors_to_use,target_label,0)

print(x_train.shape)
print(sum(y_train))

x_test, y_test= load_sensor_data(X1[2500:-1,:],Y1[2500:-1],M1[2500:-1,:],feat_sensor_names,label_names,sensors_to_use,target_label,0)
print(x_test.shape)

LocalModels, GradientsRes,  active_model= create_models(args)
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=args.lr,
    decay_steps=10000,
    decay_rate=0.99)

optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule)
loss_object = tf.keras.losses.BinaryCrossentropy() 

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.BinaryAccuracy(name='train_accuracy')
test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.BinaryAccuracy(name='test_accuracy')
test_label_accuracy = tf.keras.metrics.BinaryAccuracy(name='test_label_accuracy')
acc_train = []
acc_test = []
loss_train = []
loss_test = []
best_acc=0


def train(epoch):
    train_ds = tf.data.Dataset.from_tensor_slices(
        (x_train, y_train)).shuffle(y_train.shape[0]).batch(args.batch_size)
    current_step = 0
    for texts, labels in train_ds:
        current_step += 1
        LocalOutputs = []
        LocalEmbeds = []

        with tf.GradientTape() as passive_tape:
    
            for i in range(args.num_text_clients):
                texts_div = texts[:, args.text_feature_div[i]:args.text_feature_div[i + 1]]
                local_output = local_embed = LocalModels[i + args.num_img_clients](texts_div)
                LocalOutputs.append(local_output)
                LocalEmbeds.append(local_embed)

            with tf.GradientTape() as active_tape:
                for i in range(args.num_clients):
                    active_tape.watch(LocalOutputs[i])

                output = active_model(LocalOutputs)
                loss = loss_object(labels, output)

            trainable_variables = []
            for i in range(args.num_clients):
                trainable_variables.append(LocalOutputs[i])
            trainable_variables.append(active_model.trainable_variables)

            gradients = active_tape.gradient(loss, trainable_variables)
            EmbedGradients = gradients[:-1]
            server_model_gradients = gradients[-1]
            optimizer.apply_gradients(
                zip(server_model_gradients, active_model.trainable_variables))  # update server model

            EmbLoss = []
            local_trainable_varaibles = []
            for i in range(args.num_clients):
                EmbLoss.append(tf.multiply(LocalEmbeds[i], EmbedGradients[i].numpy()))
                local_trainable_varaibles.append(LocalModels[i].trainable_variables)
        LocalGradients = passive_tape.gradient(EmbLoss, local_trainable_varaibles)
        for i in range(args.num_clients):
            optimizer.apply_gradients(zip(LocalGradients[i], local_trainable_varaibles[i]))

        train_loss(loss)
        train_accuracy(labels, output)
       


def test(epoch):
    LocalOutputs = get_local_outputs(args.num_text_clients, x_test, args.text_feature_div, LocalModels)
    test_output = active_model(LocalOutputs)
    print(test_output.shape)
    print( np.mean(test_output == y_test))
    test_accuracy(y_test, test_output)


for epoch in range(args.epochs):
    start = time.time()
    train(epoch)
    test(epoch)

    end = time.time()
    template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}, Time: {}'
    print(template.format(epoch + 1,
                          train_loss.result(),
                          train_accuracy.result() * 100,
                          test_loss.result(),
                          test_accuracy.result() * 100,
                          end - start
                          ))

    acc_train.append(train_accuracy.result())
    acc_test.append(test_accuracy.result())
    loss_train.append(train_loss.result())
    loss_test.append(test_loss.result())
    if best_acc <= test_accuracy.result():
        best_acc = test_accuracy.result()
        # Save the weights
        for i in range(args.num_clients):
            LocalModels[i].save_weights(os.path.join(args.local_paths[i], 'best_checkpoints'))
        active_model.save_weights(os.path.join(args.sever_path, 'best_checkpoints'))
    train_loss.reset_states()
    train_accuracy.reset_states()
    test_loss.reset_states()
    test_accuracy.reset_states()
    if epoch + 1 in args.save_epochs:
        epoch_filename = "epoch{}_checkpoints".format(epoch + 1)
        for i in range(args.num_clients):
            LocalModels[i].save_weights(os.path.join(args.local_paths[i], epoch_filename))
        active_model.save_weights(os.path.join(args.sever_path, epoch_filename))

    with open(os.path.join(args.path_predix, 'acc_test.txt'), "w") as outfile:
        outfile.write("\n".join("{:.4f}".format(item * 100) for item in acc_test))

    with open(os.path.join(args.path_predix, 'acc_train.txt'), "w") as outfile:
        outfile.write("\n".join("{:.4f}".format(item * 100) for item in acc_train))