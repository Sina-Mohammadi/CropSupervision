from tensorflow.keras import backend as K
import tensorflow as tf
import tensorflow_addons as tfa



def SupCon_Loss(labels, feature_vectors,temperature=0.05):

    feature_vectors=K.reshape(feature_vectors,(-1,K.int_shape(feature_vectors)[-1]))
    labels=K.reshape(labels[...,0],(-1,1))

    feature_vectors_normalized = tf.math.l2_normalize(feature_vectors, axis=1)
    logits = tf.divide(
        tf.matmul(
            feature_vectors_normalized, tf.transpose(feature_vectors_normalized)
        ),
        temperature,
    )
    return tfa.losses.npairs_loss(tf.squeeze(labels), logits)






def IOU_Loss(y_true, y_pred,num_classes=3):
    
    eps   = K.constant(1e-22, dtype='float32')
    losses = []

    for j in range(num_classes):

        y_true1 = y_true[...,j:j+1]
        y_true1  = K.batch_flatten(y_true1)

        y_pred1 = y_pred[...,j:j+1] 
        y_pred1  = K.batch_flatten(y_pred1)

        I = K.sum(y_pred1 * y_true1 , axis=-1)
        U        = K.sum(y_pred1+y_true1 -(y_pred1 * y_true1) , axis=-1) 
        IOU   = I / (U + eps)
        Mean_IOU   = K.mean(IOU)
        IOU_Loss1 =1-Mean_IOU
        losses.append(IOU_Loss1)


    return K.sum(losses)




def F1_Loss(y_true, y_pred,num_classes=3):
    
    eps   = K.constant(1e-22, dtype='float32')

    losses=[]
    
    for j in range(num_classes):

        y_true1 = y_true[...,j:j+1]
        y_true1  = K.batch_flatten(y_true1)

        num_GT1       = K.sum(y_true1 , axis=-1)
        y_pred1 = y_pred[...,j:j+1] 
        y_pred1  = K.batch_flatten(y_pred1)


        num_M_and_GT1 = K.sum(y_pred1 * y_true1 , axis=-1)
        num_M1        = K.sum(y_pred1 , axis=-1) 
        prec_for_all_images1   = num_M_and_GT1 / (num_M1 + eps)
        recall_for_all_images1 = num_M_and_GT1 / (num_GT1+ eps)         
        Mean_Prec_all1   = K.mean(prec_for_all_images1)
        Mean_Recall_all1 = K.mean(recall_for_all_images1)
        F_Loss1 =1-(( 2*Mean_Prec_all1*Mean_Recall_all1 ) / (Mean_Prec_all1+Mean_Recall_all1+eps ))

        losses.append(F_Loss1)

      
    return K.sum(losses)
    
    

    
