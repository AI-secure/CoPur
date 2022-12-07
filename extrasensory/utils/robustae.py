import numpy as np
import tensorflow as tf
def l21_rownorm(X):
    """
    This function calculates the l21 norm of a matrix X, i.e., \sum ||X[i,:]||_2
    Input:
    -----
    X: {numpy array}
    Output:
    ------
    l21_norm: {float}
    """
    return (np.sqrt(np.multiply(X, X).sum(1))).sum()

def l21_colnorm(X):
    """
    This function calculates the l21 norm of a matrix X, i.e., \sum ||X[:,j]||_2
    Input:
    -----
    X: {numpy array}
    Output:
    ------
    l21_norm: {float}
    """
    return (np.sqrt(np.multiply(X, X).sum(0))).sum()


def inference_purify_miss(args, LocalOutputs,ae_model,L_optimizer,L_optimizer2, rae_loss_object, vis=False):
    h=np.concatenate(tuple(LocalOutputs), axis=1)
    rae_output,layer_output=ae_model(h)
    L=tf.Variable(rae_output,trainable=True)
    purify_epochs= args.purify_epochs
    for epoch in range(args.initial_epochs):
        with tf.GradientTape() as passive_tape:
            rae_output,layer_output=ae_model(L)
            rae_output_split=  tf.split(rae_output, args.num_clients, 1 )
        
            loss=0
            for i in range(args.num_clients):
             
                loss += tf.sqrt(rae_loss_object(rae_output_split[i], LocalOutputs[i]))
            L_gradients = passive_tape.gradient(loss,[L])
            if tf.norm(L_gradients)<0.1:
                break
            L_optimizer.apply_gradients(zip(L_gradients, [L]))
    L2 = tf.Variable(rae_output, trainable=True)
    for epoch in range(purify_epochs):
        with tf.GradientTape() as passive_tape:
            rae_output,layer_output=ae_model(L2)
            rae_output_split=  tf.split(rae_output, args.num_clients, 1 )
            L_split2 =  tf.split(L2, args.num_clients, 1 )
            loss2=0
            for i in range(args.num_clients):
                loss2 +=  args.tau*  tf.sqrt(rae_loss_object(rae_output_split[i], L_split2[i]))
               
            for i in range(len(args.observe_list)):
                loss2 +=  tf.sqrt(rae_loss_object( LocalOutputs[args.observe_list[i]],L_split2[args.observe_list[i]]))
            L_gradients = passive_tape.gradient(loss2,[L2])

            if tf.norm(L_gradients)<0.1:
                break

            L_optimizer2.apply_gradients(zip(L_gradients, [L2]))
            if vis:
                print('Purify epoch {}, purify L2 Loss2: {}'.format(epoch+1,loss2.numpy()))

    return  tf.split(rae_output, args.num_clients, 1 )

