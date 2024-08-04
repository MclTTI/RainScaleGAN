"""
Utilities to print stuff 
"""

import time
from sty import fg, bg, rs



#Colors for message print
colors = {'act':'li_magenta',
          'opt':'li_green',
          'path':'li_blue'}


 
def select_color(color_name):

    if color_name in colors:
        return colors[color_name]
    else:
        print("Invalid color specification!")



def msgprint(string,color='path',wait=0,interactive=False):
    """
    Print the string s as coloured text on STOUT
    and sleep for wait seconds.
    
    Parameters:
        string : string
                 The string to be printed
        color : color attribute
                See https://sty.mewo.dev/docs/coloring.html
        wait : int
               No. of seconds to sleep after printing
        interactive : bool
                      Whether the session is interactive
            
    Returns: None
             
    """
    
    if interactive:
        c = select_color(color)
        cstring = getattr(fg,c) + string + rs.fg
        print(cstring)
    else:
        print(string)
    
    time.sleep(wait)
    
    return
    


def printvariables(dataset,dataset_name=""):
    """Print the summary of variables in a netCDF dataset.
    
    Parameters : 
        dataset : NetCDF dataset
                  The dataset should be opened in read mode.
        dataset_name : str
                       The name (path) of the dataset
    
    Returns : None
    
    """
    vars = dataset.variables
    print('='*90)
    print(f'\nDataset Variables\n({dataset_name})')
    Header = ['Name', 'Shape', 'Dimensions','Units']
    print('-'*90,f'\n{Header[0]:10}{Header[1]:20}{Header[2]:25}{Header[3]:25}')
    print('-'*90)
    for key in vars:
        if hasattr(vars[key],'units'):
            print(f'{vars[key].name:10}{str(vars[key].shape):20}{str(vars[key].dimensions):25}{str(vars[key].units):25}')
        else:
            print(f'{vars[key].name:10}{str(vars[key].shape):20}{str(vars[key].dimensions):25}')
    print('='*90,'\n')
    time.sleep(3)
    return


