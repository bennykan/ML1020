import time
import pandas as pd
from keras import models
from keras import layers
from keras import optimizers
from keras import callbacks
from keras import backend as K
from keras.utils import multi_gpu_model
from keras import optimizers as opt

def create_submission(model,test_data_as_generator,output_directory,output_filename):
    test_data_as_generator.reset()
    start = time.time()
    probabilities = model.predict_generator(test_data_as_generator, len(test_data_as_generator),verbose=1)
    end = time.time()
    print(end - start)
    
    #output_dir = '/home/jupyter/Submission/'

    df_prob = pd.DataFrame(probabilities)
    df_label = pd.DataFrame(test_data_as_generator.filenames[0:79727])
    df = pd.concat([df_label,df_prob],axis=1)
    df.columns = ['img','c0','c1','c2','c3','c4','c5','c6','c7','c8','c9']
    df.head()
    df.to_csv(output_directory + output_filename,index=False)

    
    #Model with Conv Base Included
def creade_model_2hidden(pretraind_model):
    
    model = models.Sequential()
    model.add(pretraind_model)
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu', input_dim=5 * 5 * 2048))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(10, activation = 'softmax'))

    class MyCallback(callbacks.Callback):
        def on_epoch_end(self,epoch,logs=None):
            print(K.eval(self.model.optimizer.lr))
    PrintLR=MyCallback()

    #savebest=callbacks.ModelCheckpoint(filepath='/home/jupyter/Saved_Models/checkpoint-{val_acc:.2f}.h5',monitor='val_acc',save_best_only=True)
    ReduceLR=callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.75, verbose=0, mode='auto', cooldown=20, min_lr=1e-7)
    #Earlystop = callbacks.EarlyStopping(monitor='val_loss', patience=20)

    callbacks_list=[ReduceLR,PrintLR]

 

    return model,callbacks_list

    #Model with Conv Base Included
def run_tuning_model_2hidden_hyper(pretraind_model,lrate,num_epoch,optimizers,activate_func,train_generator,validation_generator,final_models,count):
    
    print('Started')
    model = models.Sequential()
    model.add(pretraind_model)
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation=activate_func, input_dim=5 * 5 * 2048))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(128, activation=activate_func))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(10, activation = 'softmax'))

    class MyCallback(callbacks.Callback):
        def on_epoch_end(self,epoch,logs=None):
            print(K.eval(self.model.optimizer.lr))
    PrintLR=MyCallback()

    #savebest=callbacks.ModelCheckpoint(filepath='/home/jupyter/Saved_Models/checkpoint-{val_acc:.2f}.h5',monitor='val_acc',save_best_only=True)
    ReduceLR=callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.75, verbose=0, mode='auto', cooldown=20, min_lr=1e-7)
    #Earlystop = callbacks.EarlyStopping(monitor='val_loss', patience=20)

    callbacks_list=[ReduceLR,PrintLR]

    parallel_model = multi_gpu_model(model, gpus=2)
    parallel_model.compile(loss='categorical_crossentropy',
              optimizer=optimizers(lr=lrate),
              metrics=['acc'])
    
    history = parallel_model.fit_generator(
      train_generator,
      steps_per_epoch=len(train_generator),
      epochs=num_epoch,
      validation_data=validation_generator,
      validation_steps=len(validation_generator),
      callbacks=callbacks_list,
      #use_multiprocessing = True,
      workers = 4,
      shuffle = True
      #,verbose = 1
     )
    print('Finished')
    savemodel = "Model_ {} ".format(count)
    model_json = model.to_json()
    with open("/home/jupyter/Saved_Models/" + savemodel + ".json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("/home/jupyter/Saved_Models/" + savemodel + ".h5")
    print("Saved model to disk")
    parameters = [savemodel,lrate,num_epoch,optimizers,activate_func,history.history['acc'][num_epoch-1],history.history['loss'][num_epoch-1],history.history['val_acc'][num_epoch-1],history.history['val_loss'][num_epoch-1]]
    final_models.append(parameters)
    print(final_models)
    
    return final_models

