"""
This program plots the learning curves generated during
the training of the GAN.
"""

import os
import pickle
import itertools
import matplotlib.pyplot as plt


def plot_losses(path):
    """
    Plot and save the GAN losses.
    
    Parameters:
         path :
                The directory to work on
             
    Returns: None
    """
    
    f = open(f'{path}/learning_metrics.pkl','rb') 

    metrics = pickle.load(f)

    losses = metrics['losses']
    dvals = metrics['dvals']

    glosses = [item[0] for item in itertools.chain(*losses)]
    dlosses = [item[1]/2.0 for item in itertools.chain(*losses)]
    dlosses_real = [item[0] for item in itertools.chain(*dvals)]
    dlosses_fake = [item[1] for item in itertools.chain(*dvals)]

    fig, (ax1,ax2) = plt.subplots(1,2,figsize=(11.7,8.3))

    fig.suptitle(f'{path}',size='xx-large')

    ax1.plot(glosses,label='Generator (logits)')
    ax1.plot(dlosses,label='Discriminator (logits)')
    ax1.set_title('Total losses',size='x-large')
    ax1.set_ylim(bottom=0.0,top=40)
    ax1.legend(fontsize=10,loc='upper right')

    ax2.plot(dlosses_real,label='Real prob.')
    ax2.plot(dlosses_fake,label='Fake prob.')
    ax2.set_ylim(bottom=0.0,top=1.0)
    ax2.set_title('Discriminator losses',size='x-large')
    ax2.legend(fontsize=15)

    plt.tight_layout()
    plt.savefig(os.path.join(path,'model_losses.png'),orientation='landscape',format='png',dpi=fig.dpi,bbox_inches='tight')

    return





