import os
import tensorflow as tf
import argparse #DB
import pandas as pd
import tensorflow.keras.models as models
from tensorflow import keras
#from config import AgeGenRec_config as cfg #DB
from tensorflow_model_optimization.quantization.keras import vitis_quantize
from tensorflow.keras.preprocessing.image import ImageDataGenerator

ap = argparse.ArgumentParser()
ap.add_argument("-n",  "--network", default="Age", help="input CNN")
args = vars(ap.parse_args())

network_name = args["network"]

if (network_name == "Age"):
	quantized_model = models.load_model('build/quantized_results/AgeGen/Age/quantized_model.h5', compile = False)
elif (network_name == "Gen"):
	quantized_model = models.load_model('build/quantized_results/AgeGen/Gen/quantized_model.h5', compile = False)



SEED = 42
print("\n")

if (network_name == "Age"):
	df = pd.read_csv('build/dataset/morph/morph_2008_nonCommercial.csv')
elif (network_name == "Gen"):
	df = pd.read_csv('build/dataset/morph/gender_balanced.csv')

datagen = ImageDataGenerator(
    validation_split=0.1,
)

df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
split_row_idx = int(len(df) * 0.2)
df_test = df.iloc[:split_row_idx, :]

if (network_name == "Age"):
	test_images = datagen.flow_from_dataframe(
	    dataframe=df_test,
	    directory='build/dataset/morph/',
	    x_col='photo',
	    y_col='age',
	    class_mode='raw',
	    target_size=(240,200),
	    seed=SEED)
elif (network_name == "Gen"):
	test_images = datagen.flow_from_dataframe(
    	dataframe=df_test,
    	directory='build/dataset/morph/',
    	x_col='photo',
   	y_col='gender',
    	class_mode='binary',
    	classes=['M', 'F'],
    	target_size=(240,200),
    	seed=SEED)

#GENERARE CARTELLA IMMAGINI PER LA VALUTAZIONE

# Compile the model
if (network_name == "Age"):
	quantized_model.compile(
	    optimizer = 'adam',
	    loss = "mean_absolute_percentage_error",
	    metrics=['mse'])
elif (network_name == "Gen"):
        quantized_model.compile(
	    optimizer = 'adam',
	    loss = "binary_crossentropy",
	    metrics=['accuracy'])

# Evaluate Quantized Model
quantized_model.evaluate(test_images)
