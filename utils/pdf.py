"""
Plot the probability density function
"""

import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns



def plot_pdf(*prec_dict,sample_size=2000,epoch=None,run=None):

    rows = len(prec_dict)

    fig, ax = plt.subplots(rows,1,
                           figsize=(8,7*rows))

    for id,i in enumerate(prec_dict):

        # Get the right axes index
        if rows == 1:
            a = ax
        else:
            a = ax[id]

        prec = {k:np.array(i[k]) for k in i.keys()}

        # Sub-sample if dataset is too big
        sample_prec = dict()
        
        for k in prec.keys():

            idxs = np.random.choice(prec[k].shape[0],
                                    sample_size,
                                    replace=False)
            
            sample_prec[k] = prec[k][idxs,...]

        # Flatten 
        pr = {k:sample_prec[k].flatten() for k in sample_prec.keys()}

        # Plot pdf
        histplot = sns.histplot(data=pr,
                                stat='density',
                                element='step',
                                fill=False,
                                kde=False,
                                log_scale=(False,True),
                                #height=6,
                                #aspect=1.,
                                label=pr.keys(),
                                ax=a
                                )
        
        # Labels, legend
        sns.despine(top=False, right=False, left=False, bottom=False)
        a.set_xlabel('mm\day',fontsize='x-large')
        a.set_ylabel('Probability Density',fontsize='x-large')

        sns.move_legend(histplot,loc='upper right',bbox_to_anchor=(.925, .85))
        
        # Title
        if id == 0 and epoch != None:
            a.set_title(f'Epoch {epoch:003}',fontsize='large',loc='right')
        if id == 0 and run:
            a.set_title(f'Run: {run}',fontsize='large',loc='left')
        
    fig.suptitle(f'Probability Density Function',fontsize='xx-large')
    plt.tight_layout()

    return fig


def qq_plot(*distrib,n_quant=100):

    quant = np.linspace(0,1,n_quant)

    fig, ax = plt.subplots(figsize=(8,8))

    x_min = 0
    x_max = 1

    #Compute quantiles
    for d in distrib:
    
        emp_dis = d['distr']
        ref_dis = d['ref']

        q_emp = np.quantile(emp_dis,quant)
        q_ref = np.quantile(ref_dis,quant)

        #Extract min and max for bisector
        xm = np.min([q_emp,q_ref])
        xM = np.max([q_emp,q_ref])

        if xm < x_min:
            x_min = xm

        if xM > x_max:
            x_max = xM

        #Make plot
        ax.scatter(q_ref,q_emp,
                   label=d['name'],
                   c=d['color'],
                   marker=d['marker']
                   )

    x = np.linspace(x_min,x_max)
    ax.plot(x,x,c="k",ls="--")

    ax.set_title('Q-Q plot: RainScaleGAN/RainFARM vs. ERA5',fontsize='x-large')

    ax.set_xlabel('ERA5 precipitation (mm/day)')
    ax.set_ylabel('Downscaled precipitation (mm/day)')
    
    ax.legend()

    return fig


