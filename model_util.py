import time
import pandas as pd
def create_submission(model,test_data_as_generator,output_directory,output_filename):

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
