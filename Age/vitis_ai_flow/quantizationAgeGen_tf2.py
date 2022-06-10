
import os
import argparse #DB
import pandas as pd
import tensorflow as tf
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



# Load Model
if (network_name == "Age"):
   #model = keras.models.load_model(os.path.join(WEIGHTS_DIR,"age-model_1630696196.147801.h5"))
   model = keras.models.load_model("build/weights/AgeGen/Age/age-model_1643142215.094539_globmax_512_128_32_e50.h5")
elif (network_name == "Gen"):
   #model = keras.models.load_model(os.path.join(WEIGHTS_DIR,"gender-model_1630751749.098821.h5"))
   model = keras.models.load_model("build/weights/AgeGen/Gen/gender_model.h5")

SEED = 42
print("\n")

if (network_name == "Age"):
	df = pd.read_csv('build/dataset/morph/morph_2008_nonCommercial.csv')
elif (network_name == "Gen"):
	df = pd.read_csv('build/dataset/morph/gender_balanced.csv')

datagen = ImageDataGenerator(
    validation_split=0.999,
)

df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
split_row_idx = int(len(df) * 0.2)
df_train = df.iloc[split_row_idx:, :]

if (network_name == "Age"):
	calibration = datagen.flow_from_dataframe(
	    dataframe=df_train,
	    directory='build/dataset/morph/',
	    x_col='photo',
	    y_col='age',
            batch_size = 1,
	    class_mode='raw',
	    target_size=(240,200),
	    subset='training',
	    seed=SEED)

elif (network_name == "Gen"):
	calibration = datagen.flow_from_dataframe(
    	dataframe=df_train,
    	directory='build/dataset/morph/',
    	x_col='photo',
   	y_col='gender',
    	class_mode='binary',
    	classes=['M', 'F'],
    	target_size=(240,200),
    	subset='training',
    	seed=SEED)

#calibration.next()
print("\n")

# Post-Training Quantize

quantizer = vitis_quantize.VitisQuantizer(model)

quantized_model = quantizer.quantize_model(
       calib_dataset = calibration, 
       calib_batch_size = 1,
       train_with_bn = False,
       freeze_bn_delay = -1,
       replace_sigmoid = False,
       replace_relu6 = False, 
       include_cle = True, 
       cle_steps = 10,
       forced_cle = True )

if (network_name == "Age"):
   quantized_model.save('build/quantized_results/AgeGen/Age/quantized_model.h5')
elif (network_name == "Gen"):
   quantized_model.save('build/quantized_results/AgeGen/Gen/quantized_model.h5')


