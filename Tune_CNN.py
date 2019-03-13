from keras.preprocessing.image import ImageDataGenerator
import model_util as util
from keras.applications import VGG19
import os

import pandas as pd

base_dir = '/home/jupyter/Data'
train_dir = os.path.join(base_dir, 'train')

train_datagen = ImageDataGenerator(
      rescale=1./255,
      rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest',
    validation_split=0.2
    )



train_generator = train_datagen.flow_from_directory(
        # This is the target directory
        train_dir,
        # All images will be resized to 150x150
        target_size=(150, 150),
        batch_size=20,
        class_mode='categorical',
        subset="training",
        shuffle = True
        )

validation_generator = train_datagen.flow_from_directory(
        # This is the target directory
        train_dir,
        # All images will be resized to 150x150
        target_size=(150, 150),
        batch_size=20,
        class_mode='categorical',
        subset="validation",
        shuffle = True
        )
        
from keras import optimizers
p = {'lr': (1e-3, 1e-5),# ],
     #'batch_size': (10,20, 30),
     'epochs': [150],

     'optimizer': [optimizers.Adam,optimizers.RMSprop],
     #'losses': ['logcosh', 'binary_crossentropy'],
     'activation':['relu','elu'],
     #'last_activation': ['sigmoid']
    }
    
    
models = []
for lr in p['lr']:
    for epoch in p['epochs']:
        #for ls in p['losses']:
            #for bs in p['batch_size']:
                for opt in p['optimizer']:
                    for act in p['activation']:
                        #print(str(lr) + ' ' + str(epoch) + ' ' + ls + ' ' + str(bs) + ' ' + str(opt) + ' '+ act)
                        runmodel = [lr,epoch,opt,act]
                        models.append(runmodel)
                        




vgg19_base = VGG19(weights='imagenet',
                  include_top=False,
                  input_shape=(150, 150, 3))

vgg19_base.trainable = True

set_trainable = False
for layer in vgg19_base.layers:
    if layer.name == 'block5_conv1' or layer.name == 'block4_conv1':
        set_trainable = True
    if set_trainable:
        layer.trainable = True
    else:
        layer.trainable = False
print(len(vgg19_base.trainable_weights))


final_models = []
count = 0
for i in models:
    count += 1
    output = util.run_tuning_model_2hidden_hyper(vgg19_base,i[0],i[1],i[2],i[3],train_generator,validation_generator,final_models,count)
    


df = pd.DataFrame(output,columns = ['Model','Inital Learning Rate','Max Epochs','Optimizers','Activation','Train Accuracy','Train Loss','Val Accuracy','Val Loss'])
df.to_csv('/home/jupyter/TuningOutput/TuningOutput.csv',index=False)