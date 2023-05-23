from tensorflow.keras.utils import to_categorical
import random
import numpy as np
import cv2


def data_generator(images,labels,MidS,batch_size=1):
    mm=0
    zz=np.arange(len(labels))
    random.shuffle(zz)
    
    while True:
        mm+=batch_size
        if mm>len(labels):
            mini_batch_indices =zz[mm-batch_size:len(labels)]
            r=mm-len(labels)
            mm=0
            zz=np.arange(len(labels))
            random.shuffle(zz)
            mm+=r
            mini_batch_indices=np.concatenate([mini_batch_indices,zz[0:mm]])
        else:    
            mini_batch_indices =zz[mm-batch_size:mm]
        imgs=[]
        lbls=[]


        for t in mini_batch_indices:
            imgs.append(images[t])
            lbls.append(to_categorical(labels[t],3))

        a=np.array(imgs)
        b=np.array(lbls)

        if MidS=='off':
            yield a,b
        elif MidS=='SupCon':
            mid1_lbls=[]
            mid2_lbls=[]
            mid3_lbls=[]
            mid4_lbls=[]
            for j in b:
                cc=cv2.resize(j,(0,0), fx=0.5, fy=0.5)
                cc=np.where(cc>=0.5,1,0)
                mid1_lbls.append(cc)
                d=cv2.resize(j,(0,0), fx=0.25, fy=0.25)
                d=np.where(d>=0.5,1,0)
                mid2_lbls.append(d)
                e=cv2.resize(j,(0,0), fx=0.125, fy=0.125)
                e=np.where(e>=0.5,1,0)
                mid3_lbls.append(e)
                f=cv2.resize(j,(0,0), fx=0.0625, fy=0.0625)
                f=np.where(f>=0.5,1,0)
                mid4_lbls.append(f)
            yield (a,[b,np.reshape(np.argmax(b,axis=-1),(batch_size,-1,1)),np.reshape(np.argmax(np.array(mid1_lbls),axis=-1),(batch_size,-1,1)),np.reshape(np.argmax(np.array(mid2_lbls),axis=-1),(batch_size,-1,1)),np.reshape(np.argmax(np.array(mid3_lbls),axis=-1),(batch_size,-1,1)),np.reshape(np.argmax(np.array(mid4_lbls),axis=-1),(batch_size,-1,1))])
    
        elif MidS=='Cross-entropy':
            mid1_lbls=[]
            mid2_lbls=[]
            mid3_lbls=[]
            mid4_lbls=[]
            for j in b:
                cc=cv2.resize(j,(0,0), fx=0.5, fy=0.5)
                cc=np.where(cc>=0.5,1,0)
                mid1_lbls.append(cc)
                d=cv2.resize(j,(0,0), fx=0.25, fy=0.25)
                d=np.where(d>=0.5,1,0)
                mid2_lbls.append(d)
                e=cv2.resize(j,(0,0), fx=0.125, fy=0.125)
                e=np.where(e>=0.5,1,0)
                mid3_lbls.append(e)
                f=cv2.resize(j,(0,0), fx=0.0625, fy=0.0625)
                f=np.where(f>=0.5,1,0)
                mid4_lbls.append(f)
            yield (a,[b,b,np.array(mid1_lbls),np.array(mid2_lbls),np.array(mid3_lbls),np.array(mid4_lbls)])
            