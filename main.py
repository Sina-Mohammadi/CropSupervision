import numpy as np
from tensorflow.keras import callbacks
from model import FCN_3D
from data_generator import data_generator
import argparse
import tables
from numpy.random import seed
from tensorflow import random

seed(1)
random.set_seed(1)



def args():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default='./data/')
    parser.add_argument("--save_dir", type=str, default='./save/')
    parser.add_argument("--out_supervision", type=str, default='IOU',choices=['IOU', 'F1', 'Cross-entropy']) 
    parser.add_argument("--mid_supervision", type=str, default='SupCon',choices=['SupCon','Cross-entropy','off']) 
    parser.add_argument("--validation_fold", type=int, default=5)    
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--site", type=str,choices=['A','B','C','D'])

    return parser.parse_args()




if __name__ == "__main__":


    cfg = args()

    if cfg.out_supervision not in ['IOU', 'F1', 'Cross-entropy']:             
        raise ValueError(" The name of the Output Supevision method must be one of the following options: 'IOU', 'F1', 'Cross-entropy' ")
        
    if cfg.mid_supervision not in ['SupCon','Cross-entropy','off']:        
        raise ValueError(" The name of the Middle Supevision method must be one of the following options: 'SupCon', 'CE' , or 'off' ")     
    
    
    
    
    hdf5_file = tables.open_file(cfg.data_dir+'/Site'+cfg.site+'_train.hdf5', mode='r+')
    
    folds = [0,117,234,352,470,588]
    i = np.arange(588)
    j = np.arange(folds[cfg.validation_fold-1], folds[cfg.validation_fold])
    k = np.delete(i,j)
    train_images = hdf5_file.root.data[k, :, :, :, :]
    train_labels = hdf5_file.root.truth[k, :, :]
    
    val_images = hdf5_file.root.data[j, :, :, :, :]
    val_labels = hdf5_file.root.truth[j, :, :]
    



    train_data =data_generator(train_images, train_labels,MidS= cfg.mid_supervision)
    val_data =data_generator(val_images, val_labels,MidS= cfg.mid_supervision)


           
        
        
    model = FCN_3D(MidS=cfg.mid_supervision,OutS=cfg.out_supervision,lr_rate=cfg.learning_rate)
    
    
    if cfg.mid_supervision=='off':
        monitor_name='accuracy'
    else:
        monitor_name='mainoutput_accuracy'
        
        
    #Defining a custom callback to start the early stopping process after the train accuracy reaches 96%    
    class custom_callback(callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            current = logs.get(monitor_name)
            if np.greater(current, 0.96):
                self.model.stop_training = True   

    check_point = callbacks.ModelCheckpoint('./save/'+cfg.site+'/OutS('+cfg.out_supervision+')-MidS('+cfg.mid_supervision+')/ValFold('+str(cfg.validation_fold)+').hdf5',monitor='val_'+monitor_name, verbose=1)                
   
    model.fit_generator(train_data, steps_per_epoch=int(np.ceil(len(train_labels))), epochs=200, verbose=1, callbacks=custom_callback(),shuffle=True)
    model.fit_generator(train_data, steps_per_epoch=int(np.ceil(len(train_labels))), epochs=200, verbose=1, callbacks=check_point, validation_data=val_data, validation_steps=int(np.ceil(len(val_labels))),shuffle=True)  
    
    
