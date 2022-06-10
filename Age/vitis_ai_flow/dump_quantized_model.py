
import os
import argparse #DB
import pandas as pd
import tensorflow as tf
import tensorflow.keras.models as models
from tensorflow import keras
#from config import AgeGenRec_config as cfg #DB
from tensorflow_model_optimization.quantization.keras import vitis_quantize
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-n",  "--network", default="Age", help="input CNN")
args = vars(ap.parse_args())

network_name = args["network"]

#KERAS_MODEL_DIR = cfg.KERAS_MODEL_DIR #DB

#WEIGHTS_DIR = os.path.join(KERAS_MODEL_DIR, network_name)


# Quantize the model
if (network_name == "Age"):
	quantized_model = models.load_model('build/quantized_results/AgeGen/Age/quantized_model.h5', compile = False)
	file = open("./dump/AgeGen/Age/pathImageAge.txt","w")
elif (network_name == "Gen"):
	quantized_model = models.load_model('build/quantized_results/AgeGen/Gen/quantized_model.h5', compile = False)
	file = open("./dump/AgeGen/Gen/pathImageGen.txt","w")


SEED = 42

# Set the image
if (network_name == "Age"):
	df = pd.read_csv('build/dataset/morph/morph_2008_nonCommercial.csv')
elif (network_name == "Gen"):
	df = pd.read_csv('build/dataset/morph/gender_balanced.csv')

datagen = ImageDataGenerator(
    validation_split=0.999,
)


df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
split_row_idx = int(len(df) * 0.2)
df_test = df.iloc[:split_row_idx, :]

if (network_name == "Age"):
	dump_dataset = datagen.flow_from_dataframe(
	    dataframe=df_test,
	    directory='build/dataset/morph/',
	    x_col='photo',
	    y_col='age',
	    class_mode='raw',
	    target_size=(240,200),
	    subset='training',
	    seed=SEED,
            save_to_dir="./dump/AgeGen/Age/Image", 
	    save_prefix='',
            save_format='png')
        
elif (network_name == "Gen"):
	dump_dataset = datagen.flow_from_dataframe(
    	dataframe=df_test,
    	directory='build/dataset/morph/',
    	x_col='photo',
   	y_col='gender',
    	class_mode='binary',
    	classes=['M', 'F'],
    	target_size=(240,200),
    	subset='training',
    	seed=SEED,
	save_to_dir="./dump/AgeGen/Gen/Image", 
	save_prefix='',
        save_format='png')
        


for i in dump_dataset.filepaths:
	file.write(i+"\n")
file.close()


# Dumping the Simulation Results
# To dump sim results, we use function vitis_quantize.VitisQuantizer.dump_model 
# explained at https://www.xilinx.com/htmldocs/xilinx2019_2/vitis_doc/tensorflow_2x.html
# vitis_quantize.VitisQuantizer.dump_model(
# dataset=None,
# output_dir=’./dump_results’,
# weights_only=False)
# dataset parameter: tf.data.Dataset or np.numpy object, is the dataset used to dump, not needed if weights_only is set to True
# output_dir: a string object
# weights_only: set to True to only dump the weights, set to False will also dump the activation results
# As 0-th argument it has to be passed the quantized_model

if (network_name == "Age"):
	vitis_quantize.VitisQuantizer.dump_model(quantized_model, dump_dataset, output_dir='./dump/AgeGen/Age', weights_only=False)
elif (network_name == "Gen"):
	vitis_quantize.VitisQuantizer.dump_model(quantized_model, dump_dataset, output_dir='./dump/AgeGen/Gen', weights_only=False)


