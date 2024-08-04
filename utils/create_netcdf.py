"""
Create a netCDF file
to accumulate statistics during model's training
"""

import os
import netCDF4

def create_netcdf(*stats,filein=None,outdir=None,fname='.nc'):
    """Create a netCDF file with dimensions (epoch, longitude, latitude).
       Longitude and latitude are copied from the dataset used to train the model.
     
    Parameters:
        *stats : str 
                 Tuple of the statistics to be stored in the file.
        filein :
                 The netCDF dataset used as a template for lon and lat
                 (same used to train the model).
        outdir : 
                 The directory where to save the created NetCDF file.
                      
    Returns:
        
    """

    print(f'Creating file {fname}')
    fout = netCDF4.Dataset(os.path.join(outdir,fname),mode='w')
    
    fout.createDimension('epoch',None)
    fout.createDimension('lon',64)
    fout.createDimension('lat',64)
    
    fout.createVariable('epoch','u4',('epoch',))
    fout.createVariable('lon','f4',('lon',))
    fout.createVariable('lat','f4',('lat',))

    for stat in stats:
        print(f'Create variable {stat}')
        fout.createVariable(f'{stat}','f8',('epoch','lon','lat',))

    fout['lon'][:] = filein['lon'][:64]
    fout['lat'][:] = filein['lat'][:64]
    #fout['lon'][:] = filein['lon'][9:73] #Taipei
    #fout['lat'][:] = filein['lat'][9:73] #Taipei

    return fout


def create_netcdf_ref(*stats,filein=None,outdir=None,fname='.nc'):
    
    print(f'Creating file {fname}')
    fout = netCDF4.Dataset(os.path.join(outdir,fname),mode='w')
    
    fout.createDimension('lon',64)
    fout.createDimension('lat',64)
    
    fout.createVariable('lon','f4',('lon',))
    fout.createVariable('lat','f4',('lat',))

    for stat in stats:
        print(f'Create variable {stat}')
        fout.createVariable(f'{stat}','f8',('lon','lat',))

    fout['lon'][:] = filein['lon'][:64]
    fout['lat'][:] = filein['lat'][:64]
    
    return fout


def save_stats_netcdf(*stats,filein=None,outdir=None,fname='.nc'):
    
    print(f'Creating file {fname}\n')

    fout = netCDF4.Dataset(os.path.join(outdir,fname),mode='w')
    
    # Group for reference statistics
    ref = fout.createGroup("reference")

    ref.createDimension('lon',64)
    ref.createDimension('lat',64)

    ref.createVariable('lon','f4',('lon',))
    ref.createVariable('lat','f4',('lat',))

    for stat in stats:
        print(f'Create variable {stat}')
        ref.createVariable(f'{stat}','f4',('lon','lat',))


    # Group for generated statistics
    gen = fout.createGroup("generated")
    
    gen.createDimension('epoch',None)
    gen.createDimension('lon',64)
    gen.createDimension('lat',64)
    
    gen.createVariable('epoch','u2',('epoch',))
    gen.createVariable('lon','f4',('lon',))
    gen.createVariable('lat','f4',('lat',))

    for stat in stats:
        print(f'Create variable {stat}')
        gen.createVariable(f'{stat}','f4',('epoch','lon','lat',))

    
    # Write lon & lat
    for g in ['reference','generated']:
        fout[g]['lon'][:] = filein['lon'][:64]
        fout[g]['lat'][:] = filein['lat'][:64]
    
    return fout



def save_spec_netcdf(Af,k,outdir=None,fname='.nc'):
    
    print(f'Creating file {fname}')
    fout = netCDF4.Dataset(os.path.join(outdir,fname),mode='w')

    # Create four groups for reference spectra (train and valid) and for generated spectra
    
    ref = fout.createGroup("refspec")

    ref.createGroup("train")
    ref.createGroup("valid")
    
    gen = fout.createGroup("genspec")
    
    gen.createGroup("train")
    gen.createGroup("valid")

    # Create dimensions and variable 
    for ds in ["train","valid"]:

        # Reference spectra
        ref[ds].createDimension("k",k.shape[0])
    
        ref[ds].createVariable("k","f8",("k",))
        ref[ds].createVariable("Af","f8",("k",))

        # Generated spectra
        gen[ds].createDimension("epoch",None)
        gen[ds].createDimension("k",k.shape[0])
        
        gen[ds].createVariable("epoch","u2",("epoch",))
        
        #gen[ds].createVariable("k","f8",("epoch","k",))
        gen[ds].createVariable("k","f8",("k",))
        gen[ds].createVariable("Af","f8",("epoch","k",))

        gen[ds].createVariable("LSD","f8",("epoch",))
    
    return fout
