"""
1. Loads a netCDF dataset
2. Prepare the tensorflow datasets
3. Instantiate and train the GAN
4. Save quality metrics and statistics
"""

import os
import sys
import time
import argparse
import netCDF4

import numpy as np
import xarray as xr

import joblib

import tensorflow as tf
from tensorflow import keras

from models import gen_UpConv, discriminator
import gan
import train
import preprocess

sys.path.append('..')
from utils.fancyprint import msgprint



###########################
# Parse argument
# create out directories
###########################
proj_dir = os.environ.get('WORK')

parser = argparse.ArgumentParser(description="Train the GAN.")

parser.add_argument("--out",
                    help=f'Out dir (Will be created in {proj_dir})')
parser.add_argument("--td",
                    help="Training dataset (Full path).")
parser.add_argument("--vd",
                    help="Validation dataset (Full path).")
parser.add_argument("-I", action="store_true",
                    help="Interactive session (print colored messages)")

args = parser.parse_args()

out_dir = os.path.join(proj_dir,args.out)
train_ds = args.td
valid_ds = args.vd
interact = args.I

msgprint('\nRoot output directories in:',
         color='act',interactive=interact)
msgprint(f'{out_dir}\n',
         color='path',wait=2,interactive=interact)
os.makedirs(out_dir,exist_ok=True)



#####################################
# Open the training/validation sets
# Extract total precipitation
# Preprocess total precipitation
#####################################
msgprint(f'Opening the training and the validation sets',
         color='act',interactive=interact)

f = netCDF4.Dataset(train_ds,mode='r')

f_train = xr.open_dataset(train_ds)

msgprint('Training dataset:',
         color='act',interactive=interact)
msgprint(f'{train_ds}\n',
         color='path',interactive=interact)
print(f_train)
time.sleep(2)

f_valid = xr.open_dataset(valid_ds)

msgprint('Validation dataset:',
         color='act',interactive=interact)
msgprint(f'{valid_ds}\n',
         color='path',interactive=interact)
print(f_valid)
time.sleep(2)


msgprint('\nExtract total precipitation, reshape tp to be model friendly',
         color='act',interactive=interact)

p_T = f_train['tp'].values
p_V = f_valid['tp'].values

p_T = p_T.reshape([p_T.shape[0],64,64,1])
p_V = p_V.reshape([p_V.shape[0],64,64,1])
print(f'Total precipitation shape after reshaping (train set): {p_T.shape}')
print(f'Total precipitation shape after reshaping (validation set): {p_V.shape}\n')



######################################
# Upscale, transform, min-max scaling
######################################
msgprint('Upscale, transform, min-max scaling',
         color='act',interactive=interact)

pn_T, scaler_p_T = preprocess.transform_normalize(p_T)
pn_V = preprocess.transform_normalize_valid(p_V,scaler_p_T)
scaler_p_V = scaler_p_T

pu_T = preprocess.upscale(p_T,8,8)    #Upscaling BEFORE transformation/normalization
pu_V = preprocess.upscale(p_V,8,8)    

pu_nn_T = preprocess.nn_remap(pu_T,64,64)    #Useless?
pu_nn_V = preprocess.nn_remap(pu_V,64,64)

pun_T, scaler_pu_T = preprocess.transform_normalize(pu_T)
pun_V = preprocess.transform_normalize_valid(pu_V,scaler_pu_T)
scaler_pu_V = scaler_pu_T

pun_nn_T = preprocess.nn_remap(pun_T,64,64)
pun_nn_V = preprocess.nn_remap(pun_V,64,64)

# Save the scalers for later use
scaler_dir = os.path.join(out_dir,"scalers")
os.makedirs(scaler_dir,exist_ok=True)

joblib.dump(scaler_p_T, os.path.join(scaler_dir,'scaler_train_set.joblib'))
joblib.dump(scaler_pu_T, os.path.join(scaler_dir,'scaler_train_set_coarse.joblib'))



##############################
# List to create TF datasets
##############################

var_list = [p_T,p_V,pn_T,pn_V,pu_T,pu_V,pun_T,pun_V,pun_nn_T,pun_nn_V]
var_names = ['p_T','p_V','pn_T','pn_V','pu_T','pu_V','pun_T','pun_V','pun_nn_T','pun_nn_V']

train_vars = pn_T,pun_T,pun_nn_T
val_vars = pn_V,pun_V,pun_nn_V


print('='*75)
Header = ['Variable','Shape','Min','Max']
print(f'\n{Header[0]:15}{Header[1]:30}{Header[2]:15}{Header[3]:15}')
print('-'*75)
for var in zip(var_list,var_names):
    var_min = np.around(np.min(var[0]),decimals=4)
    var_max = np.around(np.max(var[0]),decimals=4)
    print(f'{var[1]:15}{str(var[0].shape):30}{str(var_min):15}{str(var_max):15}')
print('='*75,'\n')



##################################
# Create Tensorflow datasets
##################################
msgprint('Create the Tensorflow datasets\n',
         color='act',interactive=interact)

BUFFER_SIZE_T = p_T.shape[0]
BUFFER_SIZE_V = p_V.shape[0]
BATCH_SIZE = 64
sd = 1337

train_sets = tf.data.Dataset.from_tensor_slices(train_vars)
train_sets = train_sets.shuffle(BUFFER_SIZE_T,seed=sd).batch(BATCH_SIZE)

val_sets = tf.data.Dataset.from_tensor_slices(val_vars)
val_sets = val_sets.shuffle(BUFFER_SIZE_V,seed=sd).batch(BATCH_SIZE)

ref_ds_train = tf.constant(p_T)
ref_ds_val = tf.constant(p_V)



######################################################
# Set generator and discriminator hyperparameters
# Instantiate generator and discriminator
#################################################
msgprint('Instantiate the generator and the discriminator(s)\n',
         color='act',interactive=interact)

GEN_KS = 4    # Test with 2, 3, 4, 5
GEN_DROPOUT = True
GEN_DROPOUT_RATE = 0.2     # Test with 0.1, 0.2, 0.3, 0.4, 0.5

DIS_KS = 4    # Test with 2, 3, 4, 5
DIS_DROPOUT = False
DIS_DROPOUT_RATE = 0.0    # Test with 0.1, 0.2, 0.3, 0.4, 0.5



generator = gen_UpConv(ks=GEN_KS,
                       dropout_flag=GEN_DROPOUT,
                       dropout_rate=GEN_DROPOUT_RATE)
generator.summary(line_length=75)

discriminator = discriminator(ks=DIS_KS,
                              dropout_flag=DIS_DROPOUT,
                              dropout_rate=DIS_DROPOUT_RATE,
                              FT_in=False)
discriminator.summary(line_length=75)



#############################
# Set GAN hyperparameters
# Instantiate GAN
#############################
msgprint('Instantiate and compile the GAN\n',
         color='act',interactive=interact)

EPOCHS = 400

LR = 2e-4
NOISE_DIM = 113    #for generator
gp_weight = 10.0    #For gradient penalty


gen_optimizer = keras.optimizers.Adam(learning_rate=LR)
discr_optimizer = keras.optimizers.Adam(learning_rate=LR)

# raingan = gan.GAN(generator=generator,
#                   discriminator=discriminator,  
#                   noise_dim=NOISE_DIM)


raingan = gan.WGANGP(generator=generator,
                     discriminator=discriminator,   
                     noise_dim=NOISE_DIM,
                     discriminator_extra_steps=5,
                     gp_weight=gp_weight)



raingan.compile(d_optimizer=discr_optimizer,
                g_optimizer=gen_optimizer,
                metr=tf.keras.metrics.RootMeanSquaredError())
                

msgprint(f'Using generator loss {raingan.g_loss_fn}',
         color='act',interactive=interact)
msgprint(f'Using discriminator loss {raingan.d_loss_fn}\n',
         color='act',interactive=interact)


##############
# Callbacks
##############
msgprint('Instantiate training callbacks\n',
         color='act',interactive=interact)

timer = train.Timer()

scorer = train.LossScorer(out_dir,
                          interactive=interact)

samples = train.SampleImages(train_sets,
                             generator,
                             discriminator,
                             scaler_p_T,
                             out_dir,
                             interactive=interact)

quality = train.Statistics(train_sets,
                           val_sets,
                           ref_ds_train,
                           ref_ds_val,
                           raingan,
                           scaler_p_T,
                           scaler_p_V,
                           out_dir,
                           f,
                           interactive=interact)

ckpts = train.TrainCheckpoints(generator = generator,
                               discriminator = discriminator,
                               gen_optimizer = gen_optimizer,
                               discr_optimizer = discr_optimizer,
                               out_dir = out_dir,
                               interactive=interact)



##########
# Check
##########
print(generator)
print(raingan.generator)

print(discriminator)
print(raingan.discriminator)


#################################################
# Train the GAN
# Save the trained generator and discriminator
#################################################
msgprint('\nStart GAN training\n',
         color='opt',interactive=interact)

res = raingan.fit(train_sets,
                  epochs=EPOCHS,
                  callbacks=[timer,scorer,samples,quality,ckpts],
                  verbose=2)
                  #validation_data=val_sets)           

#print(res.history)

msgprint('Save the trained generator and discriminator',
         color='act',interactive=interact)

generator.save(os.path.join(out_dir,"generator"))
discriminator.save(os.path.join(out_dir,"discriminator"))


print("End of main.py")

