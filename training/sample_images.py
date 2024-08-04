"""
Function to generate and save 
example images from the generator
at each training epoch.
"""

import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from preprocess import upscale


#======================================
nex = 8 #Number of example to generate
noise_dim = 113
np.random.seed(42)
seed = tf.random.normal([nex, noise_dim])
#======================================

def print_sample_images(img_batches,
                        scaler,generator,discriminator,epoch,save_dir):
    """Function to generate, print and save example images with the trained generator.
    
    Parameters:
        images : 
                 The dataset of original images.
        images_c :
                   The dataset of coarsened images.
        images_cnn :
                     The dataset of coarsend and nn remapped images.
        scaler : 
                     The scaler used to normalize the images.
        generator : 
                    The generator.
        discriminator :
                        The discriminator.
        epoch :
                The current training epoch.
        save_dir : 
                   Directory in which to save the images.
                     
    Returns:
    
    
    """
    
    imgn_batch, imgcn_batch, imgcn_nn_batch = img_batches
    
    imgn = imgn_batch[0:nex]          #n=normalized
    imgcn = imgcn_batch[0:nex]        #cn=coarse, normalized
    imgcn_nn = imgcn_nn_batch[0:nex]  #cn_nn=coarse, normalized, nn remapped

    
    # Denormalize, re-transform and upscale original images
    
    imgn = imgn.numpy()
    img = np.square(scaler.inverse_transform(imgn.reshape(imgn.shape[0],-1)).reshape(imgn.shape))
    #print(np.min(img))
    #print(np.max(img))
    #print(img.shape)
    
    imgc = upscale(img,8,8)
    #print(np.min(imgc))
    #print(np.max(imgc))
    #print(imgc.shape)
    
    imgnu = upscale(imgn,8,8)    #nu = normalized, upscaled. Similar to cn, operations are inverted.
    
    # Generate images, denormalize, re-transform and upscale
    noise_plus_coarse = tf.concat((seed,tf.reshape(imgcn,[nex,64])),axis=1) 
    g_imgn = generator(noise_plus_coarse,training=False)    #training=False -> run in inference mode
    
    g_imgn = g_imgn.numpy()
    g_img = np.square(scaler.inverse_transform(g_imgn.reshape(g_imgn.shape[0],-1)).reshape(g_imgn.shape))
    #print(np.min(g_img))
    #print(np.max(g_img))
    #print(g_img.shape)
    
    g_imgc = upscale(g_img,8,8)
    
    g_imgnc = upscale(g_imgn,8,8)
    

    discr_IN = tf.concat((g_imgn, imgcn_nn), axis=3)
    
    decision = discriminator(discr_IN, training=False)   #Da qui esce un tensore
    
    #############
    # Make plots
    #############
    fig = plt.figure(figsize=(11.7,8.3))
    fig.suptitle(f'Examples of generated images at epoch {epoch:03d}',fontsize=16)

    cols = nex
    rows = 7   
                
    for i in range(cols):
        
        ax1 = fig.add_subplot(rows,cols,i+1)
        if i == 0:
            ax1.set_title('Original',size='large',loc='left')
        ax1.imshow(img[i,:,:,0],cmap = 'BuPu')
        plt.axis('off')
        
        ax2 = fig.add_subplot(rows,cols,cols+i+1)
        if i == 0:
            ax2.set_title('Original coarsened',size='large',loc='left')
        ax2.imshow(imgc[i,:,:,0],cmap = 'BuPu')
        plt.axis('off')
        
        ax3 = fig.add_subplot(rows,cols,2*cols+i+1)
        if i == 0:
            ax3.set_title('Generated',size='large',loc='left')
        ax3.imshow(g_img[i,:,:,0],cmap = 'BuPu')
        plt.axis('off')
        ax3.text(32,85,f'D. score: {decision.numpy()[i,0]:.3f}',fontsize=9,horizontalalignment='center',color='red')

        ax4 = fig.add_subplot(rows,cols,3*cols+i+1)
        if i == 0:
            ax4.set_title('Generated coarsened',size='large',loc='left')
        ax4.imshow(g_imgc[i,:,:,0],cmap = 'BuPu')
        plt.axis('off')
        
        ax5 = fig.add_subplot(rows,cols,4*cols+i+1)
        if i == 0:
            ax5.set_title('Generated coarsened (norm.)',size='large',loc='left')
        ax5.imshow(g_imgnc[i,:,:,0],cmap = 'BuPu')
        plt.axis('off')
        
        ax6 = fig.add_subplot(rows,cols,5*cols+i+1)
        if i == 0:
            ax6.set_title('ERA5 normalized and coarsened',size='large',loc='left')
        ax6.imshow(imgnu[i,:,:,0],cmap = 'BuPu')
        plt.axis('off')
        
        ax7 = fig.add_subplot(rows,cols,6*cols+i+1)
        if i == 0:
            ax7.set_title('ERA5 coarsened and normalized',size='large',loc='left')
        ax7.imshow(imgcn[i,:,:,0],cmap = 'BuPu')
        plt.axis('off')
        
    plt.tight_layout()

    plt.savefig(os.path.join(save_dir,f'example_images_{epoch:03d}.png'),orientation='landscape',format='png',dpi=fig.dpi,bbox_inches='tight')
    #plt.show()
    plt.close()
    
    return
    
