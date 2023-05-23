from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.layers import BatchNormalization,Activation, Input, Conv3D , Concatenate , UpSampling3D, Lambda, Reshape
from loss_functions import IOU_Loss,F1_Loss,SupCon_Loss
from tensorflow.keras.optimizers import SGD
K.set_image_data_format("channels_last")


def FCN_3D(MidS,OutS,lr_rate):
    
    inputlayer=Input(shape=(128,128,23,6))

    conv0=Conv3D(32,kernel_size=(3,3,5),strides=(1,1,1), padding='same')(inputlayer)
    conv0 = BatchNormalization()(conv0)
    conv0 = Activation('relu')(conv0)

    conv1=Conv3D(64,kernel_size=(3,3,5),strides=(2,2,1), padding='same')(conv0)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)

    conv2=Conv3D(128,kernel_size=(3,3,5),strides=(2,2,1), padding='same')(conv1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation('relu')(conv2)

    conv3=Conv3D(256,kernel_size=(3,3,5),strides=(2,2,1), padding='same')(conv2)
    conv3 = BatchNormalization()(conv3)
    conv3 = Activation('relu')(conv3)

    conv4=Conv3D(512,kernel_size=(3,3,5),strides=(2,2,1), padding='same')(conv3)
    conv4 = BatchNormalization()(conv4)
    conv4 = Activation('relu')(conv4)

    conv4U=UpSampling3D(size=(2, 2, 1))(conv4)
    conv4Uconv3=Concatenate()([conv4U,conv3])

    conv5=Conv3D(256,kernel_size=(3,3,5),strides=(1,1,1), padding='same')(conv4Uconv3)
    conv5 = BatchNormalization()(conv5)
    conv5 = Activation('relu')(conv5)

    conv5U=UpSampling3D(size=(2, 2, 1))(conv5)
    conv5Uconv2=Concatenate()([conv5U,conv2])

    conv6=Conv3D(128,kernel_size=(3,3,5),strides=(1,1,1), padding='same')(conv5Uconv2)
    conv6 = BatchNormalization()(conv6)
    conv6 = Activation('relu')(conv6)

    conv6U=UpSampling3D(size=(2, 2, 1))(conv6)
    conv6Uconv1=Concatenate()([conv6U,conv1])

    conv7=Conv3D(64,kernel_size=(3,3,5),strides=(1,1,1), padding='same')(conv6Uconv1)
    conv7 = BatchNormalization()(conv7)
    conv7 = Activation('relu')(conv7)

    conv7U=UpSampling3D(size=(2, 2, 1))(conv7)
    conv7Uconv0=Concatenate()([conv7U,conv0])

    conv8=Conv3D(32,kernel_size=(3,3,5),strides=(1,1,1), padding='same')(conv7Uconv0)
    conv8 = BatchNormalization()(conv8)
    conv8 = Activation('relu')(conv8)

    conv9=Conv3D(32,kernel_size=(3,3,5),strides=(1,1,1), padding='same')(conv8)
    conv9 = BatchNormalization()(conv9)
    conv9 = Activation('relu')(conv9)

    conv10=Conv3D(16,kernel_size=(3,3,5),strides=(1,1,1), padding='same')(conv9)
    conv10 = BatchNormalization()(conv10)
    conv10 = Activation('relu')(conv10)

    conv11 = Conv3D(3, kernel_size=(1,1,23), strides=(1,1,1))(conv10)
    squeezed = Lambda(lambda x: K.squeeze(x, 3))(conv11)

    mainoutput = Activation('softmax',name='mainoutput')(squeezed)

    if OutS=='IOU':
        output_loss=IOU_Loss
    elif OutS=='F1':
        output_loss=F1_Loss
    else:
        output_loss='categorical_crossentropy'
        
        
    if MidS=='off':
        model=Model(inputlayer,mainoutput)
        
        model.compile(optimizer=SGD(lr=lr_rate, momentum=0.9), loss=output_loss,metrics={"mainoutput":"accuracy"})

            
    elif MidS=='SupCon':        
        mid4_out=Reshape((-1,23*512),name='mid4_out')(conv4)
        mid3_out=Reshape((-1,23*256),name='mid3_out')(conv5)
        mid2_out=Reshape((-1,23*128),name='mid2_out')(conv6)
        mid1_out=Reshape((-1,23*64),name='mid1_out')(conv7)
        mid0_out=Reshape((-1,23*32),name='mid0_out')(conv8)
        
        model=Model(inputlayer,[mainoutput,mid0_out,mid1_out,mid2_out,mid3_out,mid4_out])
        
        model.compile(optimizer=SGD(lr=lr_rate, momentum=0.9), loss={"mainoutput":output_loss,"mid0_out":SupCon_Loss,"mid1_out":SupCon_Loss,"mid2_out":SupCon_Loss,"mid3_out":SupCon_Loss,"mid4_out":SupCon_Loss},loss_weights=[12,1,1,1,1,1],metrics={"mainoutput":"accuracy"})
       
        
    elif MidS=='Cross-entropy': 
        mid4_out=Conv3D(3,kernel_size=(1,1,23),strides=(1,1,1))(conv4)
        mid4_out = Lambda(lambda x: K.squeeze(x, 3))(mid4_out)
        mid4_out = Activation('softmax',name="mid4_out")(mid4_out)
        
        mid3_out=Conv3D(3,kernel_size=(1,1,23),strides=(1,1,1))(conv5)
        mid3_out = Lambda(lambda x: K.squeeze(x, 3))(mid3_out)
        mid3_out = Activation('softmax',name="mid3_out")(mid3_out)
        
        mid2_out=Conv3D(3,kernel_size=(1,1,23),strides=(1,1,1))(conv6)
        mid2_out = Lambda(lambda x: K.squeeze(x, 3))(mid2_out)
        mid2_out = Activation('softmax',name="mid2_out")(mid2_out)
                
        mid1_out=Conv3D(3,kernel_size=(1,1,23),strides=(1,1,1))(conv7)
        mid1_out = Lambda(lambda x: K.squeeze(x, 3))(mid1_out)
        mid1_out = Activation('softmax',name="mid1_out")(mid1_out)
        
        mid0_out=Conv3D(3,kernel_size=(1,1,23),strides=(1,1,1))(conv8)
        mid0_out = Lambda(lambda x: K.squeeze(x, 3))(mid0_out)
        mid0_out = Activation('softmax',name="mid0_out")(mid0_out)
        
        model=Model(inputlayer,[mainoutput,mid0_out,mid1_out,mid2_out,mid3_out,mid4_out])
        
        model.compile(optimizer=SGD(lr=lr_rate, momentum=0.9), loss={"mainoutput":output_loss,"mid0_out":'categorical_crossentropy',"mid1_out":'categorical_crossentropy',"mid2_out":'categorical_crossentropy',"mid3_out":'categorical_crossentropy',"mid4_out":'categorical_crossentropy'},loss_weights=[20,1,1,1,1,1],metrics={"mainoutput":"accuracy"})

    
    
    return model