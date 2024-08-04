"""
Callbacks used during GAN training
"""

import os
import sys
import time
import pickle

import netCDF4

import numpy as np
from matplotlib import pyplot as plt

from sklearn.metrics import mean_squared_error

import tensorflow as tf
from tensorflow import keras
import tensorflow_probability as tfp

from sample_images import print_sample_images

sys.path.append('..')

from utils.powerspec import powerspec
from utils.pdf import plot_pdf
from utils.create_netcdf import create_netcdf
from utils.fancyprint import msgprint
from utils.plot_losses import plot_losses

import preprocess



class Timer(keras.callbacks.Callback):

    def on_train_begin(self, logs=None):
        self.train_start_time = time.time()
        self.tot_times = []    #For tot elapsed times
        self.epoch_times = []    #Fot epoch elapsed times


    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time.time()


    def on_epoch_end(self, epoch, logs=None):
        
        epoch_elaps_time = time.time() - self.epoch_start_time
        tot_elaps_time = time.time() - self.train_start_time

        self.tot_times.append(tot_elaps_time)
        self.epoch_times.append(epoch_elaps_time)

        print('\n'*2+'='*50)
        print(f'EPOCH {epoch+1}')
        print('-'*50)
        print(f'Elapsed time {epoch_elaps_time:.2f} s | Total elapsed time {tot_elaps_time:.2f} s')



class LossScorer(keras.callbacks.Callback):

    def __init__(self,out_dir,interactive=False):

        self.out_dir = out_dir
        
        self.losses = dict()
        self.losses['losses'] = []    #For the losses 
        self.losses['dvals'] = []       #For the discriminator probability

        self.interact = interactive
                
        
    def on_epoch_begin(self, epoch, logs=None):
        
        self.epoch_losses = []
        self.epoch_d_vals = []
            
    
    def on_train_batch_end(self, batch, logs=None):
        
        """
        keys = list(logs.keys())
        print("...Training: end of batch {}; got log keys: {}".format(batch, keys))

        print('\ng_loss: ',logs['g_loss'])
        print('d_loss: ',logs['d_loss'])
        print('d_loss_real: ',logs['d_loss_real'])
        print('d_loss_fake: ',logs['d_loss_fake'])
        print('d_probs_real: ',logs['d_probs_real'])
        print('d_probs_fake: ',logs['d_probs_fake'])
        """

        self.epoch_losses.append((logs['g_loss'],logs['d_loss'],logs['d_loss_real'],logs['d_loss_fake'])) 
        self.epoch_d_vals.append((logs['d_probs_real'],logs['d_probs_fake']))
        

    def on_epoch_end(self, epoch, logs=None):

        """
        keys = list(logs.keys())
        print("\nEnd epoch {} of training; got log keys: {}".format(epoch, keys))

        print('\ng_loss: ',logs['g_loss'])
        print('d_loss: ',logs['d_loss'])
        print('d_loss_real: ',logs['d_loss_real'])
        print('d_loss_fake: ',logs['d_loss_fake'])
        print('d_probs_real: ',logs['d_probs_real'])
        print('d_probs_fake: ',logs['d_probs_fake'])
        """
        
        self.avg_epoch_losses = np.mean(self.epoch_losses,axis=0)

        self.losses['losses'].append(self.epoch_losses)
        self.losses['dvals'].append(self.epoch_d_vals)

        print('-'*50)
        Header = ['','Average loss']
        print(f'\n{Header[0]:30}{Header[1]:10}')
        print('-'*50)
        for name, loss in zip(['Generator','Discriminator (Total)','Discriminator (Real)','Discriminator (Fake)'],self.avg_epoch_losses):
            print(f'{name:30}{loss:6.4f}')
        print('='*50,'\n')

        

    def on_train_end(self, logs=None):
        
        msgprint('Plot and save training losses (.pkl and .png) in',
                 color='act',interactive=self.interact)
        msgprint(f'{self.out_dir}\n',
                 color='path',interactive=self.interact)

        with open(f'{self.out_dir}/learning_metrics.pkl','wb') as f:
            pickle.dump(self.losses,f)

        plot_losses(self.out_dir)



class SampleImages(keras.callbacks.Callback):
    # Plot and save examples of generated images

    def __init__(self, datasets, generator, discriminator, scaler, out_dir, interactive=False):

        self.datasets = datasets

        self.generator = generator
        self.discriminator = discriminator
        self.scaler = scaler

        self.out_dir = out_dir

        self.interact = interactive


    def on_train_begin(self, logs=None):

        msgprint("Create folder for example plots",
                 color='act',interactive=self.interact)
        self.explt_dir = os.path.join(self.out_dir,'example_plots')    #For the function print_sample_images
        msgprint(f'{self.explt_dir}\n',
                 color='path',interactive=self.interact)
        os.makedirs(self.explt_dir,exist_ok=True)


    def on_epoch_end(self, epoch, logs=None):

        if epoch == 0 or (epoch+1) % 10 == 0:
            
            epoch += 1
            
            msgprint('Creating example plots',
                     color='act',interactive=self.interact)
            
            batch_1, *_ = self.datasets    #Unpack the dataset  

            print_sample_images(batch_1,
                                self.scaler,
                                self.generator,
                                self.discriminator,
                                epoch,
                                self.explt_dir)

            print('Example images saved in')
            msgprint(f'{self.explt_dir}\n',
                     color='path',interactive=self.interact)



def stats(dataset):
    clim = tf.math.reduce_mean(dataset,axis=0)
    std = tf.math.reduce_std(dataset,axis=0)
    p95 = tfp.stats.percentile(dataset,q=95,axis=0)
    p99 = tfp.stats.percentile(dataset,q=99,axis=0)

    return clim, std, p95, p99



class Statistics(keras.callbacks.Callback):
    """
    Compute statistics on the generated dataset
    at the end of each epoch
    """

    def __init__(self,
                 train_ds,
                 valid_ds,
                 ref_train_ds,
                 ref_valid_ds,
                 model,
                 scaler_T,
                 scaler_V,
                 out_dir,
                 template_file,
                 interactive=False):
                 
        self.train_ds = train_ds
        self.valid_ds = valid_ds
        self.ref_train_ds = ref_train_ds
        self.ref_valid_ds = ref_valid_ds

        self.model = model    #Qui dovrei passare l'oggetto GAN
        self.scaler_T = scaler_T
        self.scaler_V = scaler_V

        self.out_dir = out_dir

        self.template_file = template_file

        self.interact = interactive

        # Set output directories
        self.stat_dir = os.path.join(self.out_dir,'statistics')    #For clim, StD, q95, q99  
        self.specplot_dir = os.path.join(self.out_dir,'powerspec')    #For power spectrum
        self.pdf_dir = os.path.join(self.out_dir,'pdf')     #For pdf
        self.save_dir = os.path.join(self.out_dir,'save')     #For pdf

        
        # Set temporary variables to store quality metrics
        self.RMSEs = dict()
       
        self.stat_names = ('clim','std','p95','p99',
                           'clim_valid','std_valid','p95_valid','p99_valid')
        
        for s in self.stat_names:
            self.RMSEs[s] = []
        
        self.RMSEs['LSD'] = []
        self.RMSEs['LSD_valid'] = []

        self.RMSEs['RMSE_train'] = []
        self.RMSEs['RMSE_valid'] = []


        # Variables to store reference statistics
        self.ref_stats = dict()

        # Variable to store reference power spectra
        self.ref_Af_T = None    # For train dataset
        self.ref_k_T = None

        self.ref_Af_V = None    # For valid dataset
        self.ref_k_V = None

        #self.ref_avg_af_coarse = None
        #self.ref_avg_k_coarse = None

        
    def on_train_begin(self, logs=None):

        # Create dirs to store results
        os.makedirs(self.stat_dir,exist_ok=True)          
        os.makedirs(self.specplot_dir,exist_ok=True)
        os.makedirs(self.pdf_dir,exist_ok=True)
        os.makedirs(self.save_dir,exist_ok=True)

        # Reference statistics
        msgprint('Compute statistics for the reference (train and validation) datasets',
                 color='act',interactive=self.interact)

        ref_clim_T, ref_std_T, ref_p95_T, ref_p99_T = stats(self.ref_train_ds)   
        ref_clim_V, ref_std_V, ref_p95_V, ref_p99_V = stats(self.ref_valid_ds)
              
        # Print some info 
        Header = ['', 'Shape', 'Min (mm/day)','Max (mm/day)']
        Stats = {'Clim (Train)' : ref_clim_T,
                 'St Dev (Train)' : ref_std_T,
                 'p95 (Train)' : ref_p95_T,
                 'p99 (Train)' : ref_p99_T,
                 'Clim (Valid)' : ref_clim_V,
                 'St Dev (Valid)' : ref_std_V,
                 'p95 (Valid)' : ref_p95_V,
                 'p99 (Valid)' : ref_p99_V}
        print('='*70)
        print('\nReference datasets statistics')
        print('-'*70,f'\n{Header[0]:20}{Header[1]:20}{Header[2]:20}{Header[3]:20}')
        print('-'*70)
        for stat in Stats:
            print(f'{stat:20}'\
                f'{str(tf.shape(Stats[stat]).numpy()):20}'\
                f'{str(np.around(tf.reduce_min(Stats[stat]),decimals=6)):20}'\
                f'{str(np.around(tf.reduce_max(Stats[stat]),decimals=6)):20}')
        print('='*70,'\n')
        if self.interact:
            time.sleep(3)


        # Create output files for statistics
        msgprint('Create output files for statistics in',
                 color='act',interactive=self.interact)
        msgprint(f'{self.stat_dir}\n',
                 color='path',interactive=self.interact)
                     
        self.fstat_name = 'statistics.nc'
        
        fstat = create_netcdf.save_stats_netcdf(*self.stat_names,
                                                filein=self.template_file,
                                                outdir=self.stat_dir,
                                                fname=self.fstat_name)

        # Save reference statistics
        msgprint('Save reference statistics in',
                 color='act',interactive=self.interact)
        msgprint(f'{self.stat_dir}\n',
                 color='path',interactive=self.interact)

        ref_stat_names = ('ref_clim','ref_std','ref_p95','ref_p99',
                         'ref_clim_valid','ref_std_valid','ref_p95_valid','ref_p99_valid')

        self.ref_stats = {'clim':ref_clim_T,'std':ref_std_T,'p95':ref_p95_T,'p99':ref_p99_T,
                          'clim_valid':ref_clim_V,'std_valid':ref_std_V,'p95_valid':ref_p95_V,'p99_valid':ref_p99_V}
        
        for s in self.ref_stats.keys():
            name = f'{s}'
            fstat["reference"][name][:,:] = self.ref_stats[s][:,:]

        fstat.close()
                
        
        # Reference power spectra
        # The computed spectrum is the spectrum AVERAGED over time
        msgprint('Compute the power spectrum for the reference (train and validation) datasets\n',
                 color='act',interactive=self.interact)
        
        self.ref_Af_T, self.ref_k_T = powerspec(self.ref_train_ds)
        self.ref_Af_V, self.ref_k_V = powerspec(self.ref_valid_ds)

        # Create file to save power spectra
        msgprint('Save reference power spectra in',
                 color='act',interactive=self.interact)
        msgprint(f'{self.specplot_dir}\n',
                 color='path',interactive=self.interact)

        self.fspec_name = 'powerspec.nc'

        fspec = create_netcdf.save_spec_netcdf(self.ref_Af_T,self.ref_k_T,
                                               outdir=self.specplot_dir,
                                               fname=self.fspec_name)

        
        fspec["refspec"]["train"]["k"][:] = self.ref_k_T[:]
        fspec["refspec"]["train"]["Af"][:] = self.ref_Af_T[:]
        
        fspec["refspec"]["valid"]["k"][:] = self.ref_k_V[:]
        fspec["refspec"]["valid"]["Af"][:] = self.ref_Af_V[:]

        fspec.close()

      
        #Coarsen ref dataset and compute spectrum
        #ref_ds_train_coarse = preprocess.upscale(self.ref_ds_train.numpy(),8,8)
        #ref_ds_train_coarse_nn = preprocess.nn_remap(ref_ds_train_coarse,64,64)

        #print(ref_ds_train_coarse_nn.shape)

        #self.ref_avg_af_coarse, self.ref_avg_k_coarse = powerspec(ref_ds_train_coarse_nn)

        #Save train and valid sets
        save_name_T = f'Train_set_images'
        save_name_V = f'Validation_set_images'

        save_T = self.ref_train_ds.numpy()
        save_V = self.ref_valid_ds.numpy()
            
        np.save(os.path.join(self.save_dir,save_name_T),save_T)
        np.save(os.path.join(self.save_dir,save_name_V),save_V)

        
    def on_epoch_end(self, epoch, logs=None):

        #epoch += 1    #The value stored in epoch is (current epoch - 1)!

        #Generate images from training and validation datasets       
        msgprint("Running the generator in inference mode",
                 color='act',interactive=self.interact)

        gen_imgsN = self.model.predict(self.train_ds)
        print(f'\nGenerated {gen_imgsN.shape[0]} samples from training set')

        # Print and save RMSE
        RMSE_train = self.model.metr.result().numpy()
        print('RMSE training set: ',RMSE_train)
        self.RMSEs['RMSE_train'].append(RMSE_train)

        # Reset metrics
        self.model.metr.reset_states()
        #print(self.model.metr.result())

        gen_val_imgsN = self.model.predict(self.valid_ds)
        print(f'\nGenerated {gen_val_imgsN.shape[0]} samples from validation set')

        RMSE_valid = self.model.metr.result().numpy()
        print(f'RMSE validation set: {RMSE_valid}\n')
        self.RMSEs['RMSE_valid'].append(RMSE_valid)

        # Reset metrics
        self.model.metr.reset_states()

        ###################################
        # Test for the GAN .predict method
        ###################################

        # gen_imgsN_call_all = np.empty((0,64,64,1))
        # gen_val_imgsN_call_all = np.empty((0,64,64,1)) 

        # count_train = 0
        # for batch in self.train_ds:

        #     img,img_c,img_cnn,_,_ = batch
                         
        #     BATCH_SIZE = tf.shape(img_c)[0]
        #     #noise = tf.random.normal([BATCH_SIZE,113])
        #     noise = tf.zeros([BATCH_SIZE,113])
                    
        #     noise_plus_coarse = tf.concat((noise, tf.reshape(img_c, [BATCH_SIZE, 64])), 1)
        #     gen_imgsN_call = self.model.generator(noise_plus_coarse,training=False)
                        
        #     count_train += gen_imgsN_call.shape[0]
            
        #     gen_imgsN_call_all = np.concatenate((gen_imgsN_call_all,gen_imgsN_call.numpy()),axis=0)

        # print(f'Generated {count_train} samples from training set (CALL)')
            
        # count_val = 0
        # for batch_val in self.valid_ds:

        #     img_val,img_c_val,img_cnn_val,_,_ = batch_val 
             
        #     BATCH_SIZE_VAL = tf.shape(img_c_val)[0]
        #     #noise_val = tf.random.normal([BATCH_SIZE_VAL,113])
        #     noise_val = tf.zeros([BATCH_SIZE_VAL,113])
        
        #     noise_plus_coarse_val = tf.concat((noise_val, tf.reshape(img_c_val, [BATCH_SIZE_VAL, 64])), 1)
        #     gen_val_imgsN_call = self.model.generator(noise_plus_coarse_val,training=False)

        #     count_val += gen_val_imgsN_call.shape[0]

        #     gen_val_imgsN_call_all = np.concatenate((gen_val_imgsN_call_all,gen_val_imgsN_call.numpy()),axis=0)

        # print(f'Generated {count_val} samples from validation set (CALL)\n')
                 
        #gen_imgs_call = np.square(self.scaler.inverse_transform(gen_imgsN_call_all.reshape(gen_imgsN_call_all.shape[0],-1)).reshape(gen_imgsN_call_all.shape))
        #gen_val_imgs_call = np.square(self.scaler.inverse_transform(gen_val_imgsN_call_all.reshape(gen_val_imgsN_call_all.shape[0],-1)).reshape(gen_val_imgsN_call_all.shape))

        ############
        # End test
        ############

        # Denormalize & re-transform the generated images
        gen_imgs = preprocess.inv_transform_normalize(gen_imgsN,self.scaler_T)
        gen_val_imgs = preprocess.inv_transform_normalize(gen_val_imgsN,self.scaler_V)

        #gen_imgs_call = preprocess.inv_transform_normalize(gen_imgsN_call_all,self.scaler)
        #gen_val_imgs_call = preprocess.inv_transform_normalize(gen_val_imgsN_call_all,self.scaler)


        #print('Min: ',np.min(gen_val_imgs),"Min CALL: ",np.min(gen_val_imgs_call))
        #print('Max: ',np.max(gen_val_imgs),"Max CALL: ",np.max(gen_val_imgs_call))

        #print('Min: ',np.min(gen_imgs),"Min CALL: ",np.min(gen_imgs_call))
        #print('Max: ',np.max(gen_imgs),"Max CALL: ",np.max(gen_imgs_call))

        
        # Statistics for the current epoch
        msgprint("Computing and saving statistics metrics",
                 color='act',interactive=self.interact)

        clim, std, p95, p99 = stats(gen_imgs)
        clim_V, std_V, p95_V, p99_V = stats(gen_val_imgs)    #Validation

        #Open file to store statistics (the same created in the method on_train_begin)
        fstat = netCDF4.Dataset(os.path.join(self.stat_dir,self.fstat_name),mode='a')

        fstat['generated']['epoch'][epoch] = epoch + 1
        
        #print(fstat['generated']['epoch'][:])

        epoch_stats = {'clim':clim, 'std':std, 'p95':p95, 'p99':p99,
                       'clim_valid':clim_V,'std_valid':std_V,'p95_valid':p95_V,'p99_valid':p99_V}

        RMSEs = dict()

        for s in epoch_stats.keys():

            # Save statistics to out file
            fstat['generated'][s][epoch,:,:] = epoch_stats[s][:,:,:]

            # Compute RMSE w.r.t. ref datasets
            MSE = keras.metrics.mean_squared_error(tf.reshape(self.ref_stats[s],[-1]),
                                                   tf.reshape(epoch_stats[s],[-1]))

            RMSEs[s] = tf.math.sqrt(MSE)

            # # Man computation of MSE
            # epoch_stats[s] = tf.cast(epoch_stats[s],dtype=tf.float32)
            # MSE_man = tf.math.subtract(tf.reshape(self.ref_stats[s],[-1]),
            #                            tf.reshape(epoch_stats[s],[-1]))
            
            # MSE_man = tf.square(MSE_man)
            # MSE_man = tf.math.reduce_mean(MSE_man,axis=0)
     
            # # sklearn computation of RMSE
            # RMSE_sklearn = mean_squared_error(self.ref_stats[s].numpy().flatten(),
            #                                   epoch_stats[s].numpy().flatten(),
            #                                   squared=False)

            # Append to self.RMSEs to save later
            self.RMSEs[s].append(RMSEs[s])

        # Close stat out file
        fstat.close()

        # Print statisitcs recap
        print('\n'*2+'='*50)
        print(f'EPOCH {epoch+1} statistics RMSEs')
        print('-'*50)
        Header = ['','Training set','Validation set']
        print(f'\n{Header[0]:10}{Header[1]:15}{Header[2]:15}')
        print('-'*50)
        for sname in ['clim','std','p95','p99']:
            val_name = f'{sname}_valid'
            print(f'{sname:10}{str(np.around(RMSEs[sname],decimals=6)):15}{str(np.around(RMSEs[val_name],decimals=6)):15}')
        print('='*50,'\n')

        # Power spectrum and log spectral distance
        # The computed spectrum is the spectrum AVERAGED over time
        msgprint('Computing power spectrum and log spectral distance',
                 color='act',interactive=self.interact)
                
        Af, k = powerspec(gen_imgs)
        Af_valid, k_valid = powerspec(gen_val_imgs)

        # Convert powerspec in db
        log10 = tf.experimental.numpy.log10

        Af_ref_T_db = 10 * log10(self.ref_Af_T)
        Af_ref_V_db = 10 * log10(self.ref_Af_V)

        Af_db = 10 * log10(Af)
        Af_valid_db = 10 * log10(Af_valid)

        # Save power spectra
        fspec = netCDF4.Dataset(os.path.join(self.specplot_dir,self.fspec_name),mode='a')

        fspec["genspec"]["train"]["epoch"][epoch] = epoch + 1
        fspec["genspec"]["train"]["k"][:] = k[:]
        fspec["genspec"]["train"]["Af"][epoch,:] = Af[:]

        fspec["genspec"]["valid"]["epoch"][epoch] = epoch + 1
        fspec["genspec"]["valid"]["k"][:] = k_valid[:]
        fspec["genspec"]["valid"]["Af"][epoch,:] = Af_valid[:]

        # check if k changed
        if epoch != 0:
            assert np.array_equal(fspec["genspec"]["train"]["k"][:],k)

        # Compute Log Spectral Distance
        Af_MSE = keras.metrics.mean_squared_error(Af_ref_T_db,Af_db)
        Af_valid_MSE = keras.metrics.mean_squared_error(Af_ref_V_db,Af_valid_db)
        
        LSD = tf.math.sqrt(Af_MSE)
        LSD_valid = tf.math.sqrt(Af_valid_MSE)

        # Save LSD 
        self.RMSEs['LSD'].append(LSD)
        self.RMSEs['LSD_valid'].append(LSD_valid)

        fspec["genspec"]["train"]["LSD"][epoch] = LSD
        fspec["genspec"]["valid"]["LSD"][epoch] = LSD_valid

        # Close spec file
        fspec.close()

        # Save RMSEs
        msgprint('Saving statistics and powerspec RMSEs\n',
                 color='act',interactive=self.interact)

        with open(f'{self.stat_dir}/RMSEs.pkl','wb') as f:
            pickle.dump(self.RMSEs,f)

        
        # # Plot and save the power spectra
        # pltspec_new = powerspec.plotspec_comparison((Af,k,self.ref_Af_T,self.ref_k_T,'Generated (train)','ERA5 (train)'),
        #                                             (Af_valid,k_valid,self.ref_Af_V,self.ref_k_V,'Generated (valid)','ERA5 (valid)'),
        #                                             epoch=epoch+1,run=self.out_dir)

        # pltspec_new.savefig(f'{self.specplot_dir}/powerspec_epoch{epoch+1:003}.png',
        #                     orientation='landscape',format='png')
        
        # plt.close('all')

        # print('Power spectra saved in')
        # msgprint(f'{self.specplot_dir}\n',
        #          color='path',interactive=self.interact)


        # Save the generated images (To plot the pdf) (every 50 epochs)
        if epoch ==0 or (epoch+1) % 50 == 0:

            msgprint(f'Save the generated images\n',
                     color='act',interactive=self.interact)

            save_name_T = f'Generated_images_train_epoch{epoch+1:003}'
            save_name_V = f'Generated_images_valid_epoch{epoch+1:003}'

            save_T = gen_imgs.numpy()
            save_V = gen_val_imgs.numpy()
            
            np.save(os.path.join(self.save_dir,save_name_T),save_T)
            np.save(os.path.join(self.save_dir,save_name_V),save_V)


            # msgprint(f'Plot the pdf (epoch {epoch+1})\n',c=self.act_col)
        
            # pdf_plot = plot_pdf(self.ref_train_ds,
            #                     gen_imgs,
            #                     epoch+1,
            #                     self.out_dir)

            # pdf_plot.savefig(f'{self.pdf_dir}/pdf_epoch{epoch+1:003}.png',
            #                     orientation='landscape',
            #                     format='png')

            # print('pdf plot saved in')
            # msgprint(f'{self.pdf_dir}\n', c=self.path_col)

               
        
               
class TrainCheckpoints(keras.callbacks.Callback):
    # Plot and save examples of generated images
    def __init__(self,generator,discriminator,gen_optimizer,discr_optimizer,out_dir,interactive=False):

        self.generator = generator
        self. discriminator = discriminator
        self.gen_optimizer = gen_optimizer
        self.discr_optimizer = discr_optimizer

        self.out_dir = out_dir
        self.ckpt_dir = None

        self.interact = interactive
       

    def on_train_begin(self, logs=None):

        self.ckpt_dir = os.path.join(self.out_dir,'checkpoints')    #For checkpoints
        os.makedirs(self.ckpt_dir,exist_ok=True)

        msgprint('Initialize checkpoint object. Checkpoints will be saved in',
                 color='act',interactive=self.interact)
        msgprint(f'{self.ckpt_dir}\n',
                 color='path',interactive=self.interact)

        self.ckpt = tf.train.Checkpoint(generator_optimizer=self.gen_optimizer,
                                        discriminator_optimizer=self.discr_optimizer,
                                        generator=self.generator,
                                        discriminator=self.discriminator)
        
        
    def on_epoch_end(self, epoch, logs=None):

        # Checkpoint every 50 epochs
        if (epoch+1) % 50 == 0:
            
            ckpt_prefix = f'ckpt_epoch{epoch+1:03d}'
            msgprint(f'Saving checkpoint {ckpt_prefix}\n',
                     color='act',interactive=self.interact)
            self.ckpt.save(file_prefix=os.path.join(self.ckpt_dir,ckpt_prefix))
        
        
        
