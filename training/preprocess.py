"""
Functions to upscale and make nearest neighbour remap
"""

import numpy as np
import tensorflow as tf

from sklearn.preprocessing import MinMaxScaler



def upscale(array,rows,cols):
    """Upscale (coarsen) the input array to the specified spatial resolution.
    
    Parameters:
        array : numpy ndarray
                Array to be upscaled.
        rows : int
               Numbers of rows required for the output array.
        cols : int
               Numbers of columns required for the output array.
    Returns:
        The upscaled array.
    """
    cf_x = int(array.shape[1]/rows)     # coarsen factor
    cf_y = int(array.shape[2]/cols)
    return array.reshape(array.shape[0],rows,cf_x,cols,cf_y,1).sum(axis=4).sum(axis=2) / (cf_x*cf_y)



def upscale_TF(tensor,rows,cols):
    """Upscale (coarsen) the input tensor to the specified spatial resolution.
       Tensorflow version of the function upscale
    
    Parameters:
        array : TF tensor
                Tensor to be upscaled.
        rows : int
               Numbers of rows required for the output tensor.
        cols : int
               Numbers of columns required for the output tensor.
    Returns:
        The upscaled tensor.
    """

    cf_x = int(tensor.shape[1]/rows)     # coarsen factor
    cf_y = int(tensor.shape[2]/cols)

    tensor_tmp = tf.reshape(tensor,[tensor.shape[0],rows,cf_x,cols,cf_y,1])
    
    #tensor_tmp2 = tf.math.reduce_sum(tensor_tmp,axis=4)
    #tensor_tmp2 = tf.math.reduce_sum(tensor_tmp2,axis=2)

    tensor_tmp2 = tf.math.reduce_sum(tensor_tmp,axis=[2,4])

    # assert tf.math.equal(tensor_tmp2,tensor_tmp_alt).numpy().any()
            
    RES = tensor_tmp2 / (cf_x*cf_y)
    
    return RES



def upscale_TF_train(tensor,rows,cols):
    """Upscale (coarsen) the input tensor to the specified spatial resolution.
       Tensorflow version of the function upscale
    
    Parameters:
        array : TF tensor
                Tensor to be upscaled.
        rows : int
               Numbers of rows required for the output tensor.
        cols : int
               Numbers of columns required for the output tensor.
    Returns:
        The upscaled tensor.
    """

    BATCH_SIZE = tf.shape(tensor)[0]
    print(BATCH_SIZE)
    
    #cf_x = int(tensor.shape[1]/rows)     # coarsen factor
    cf_x = tf.cast(tf.math.round(tf.shape(tensor)[1]/rows),dtype=tf.int32)
    #cf_y = int(tensor.shape[2]/cols)
    cf_y = tf.cast(tf.math.round(tf.shape(tensor)[2]/cols),dtype=tf.int32)
    print(cf_x)

    tensor_tmp = tf.reshape(tensor,[BATCH_SIZE,rows,cf_x,cols,cf_y,1])
    
    #tensor_tmp2 = tf.math.reduce_sum(tensor_tmp,axis=4)
    #tensor_tmp2 = tf.math.reduce_sum(tensor_tmp2,axis=2)

    tensor_tmp2 = tf.math.reduce_sum(tensor_tmp,axis=[2,4])

    # assert tf.math.equal(tensor_tmp2,tensor_tmp_alt).numpy().any()
            
    RES = tensor_tmp2 / (tf.cast(cf_x,dtype=tf.float32)*tf.cast(cf_y,dtype=tf.float32))
    
    return RES



def nn_remap(array,rows,cols):
    """Perform the nearest neighbour remap of the input array.
    
    Parameters:
        tensor : numpy ndarray
                 Array to be remapped.
        rows : int
               Numbers of rows required for the output array.
        cols : int
               Numbers of columns required for the uotput array.
    Returns:
        The remapped array.
    """
    sx = int(rows/array.shape[1])
    sy = int(cols/array.shape[2])
    return array.repeat(sx,axis=1).repeat(sy,axis=2)



def nn_remap_TF_train(tensor,rows,cols):
    """Perform the nearest neighbour remap of the input array.
    
    Parameters:
        tensor : numpy ndarray
                 Array to be remapped.
        rows : int
               Numbers of rows required for the output array.
        cols : int
               Numbers of columns required for the uotput array.
    Returns:
        The remapped array.
    """

    shape = tf.shape(tensor)
    
    sx = tf.cast(tf.math.round(rows/shape[1]),dtype=tf.int32)
    sy = tf.cast(tf.math.round(cols/shape[2]),dtype=tf.int32)
    
    #sx = int(rows/array.shape[1])
    #sy = int(cols/array.shape[2])
    
    #return array.repeat(sx,axis=1).repeat(sy,axis=2)

    RES_tmp = tf.repeat(tensor,sx,axis=1)
    RES = tf.repeat(RES_tmp,sy,axis=2)

    return RES



###################
# For real images
###################


def transform_normalize(var):
    """Apply a transformation and the min-max normalization.
    
    Parameters:
        var : numpy ndarray
              Array to be transformed and normalized.
    Returns:
        var : numpy ndarray
                The array transformed and normalized.
        scaler : scikit-learn estimator object
    """
    var = np.sqrt(var)
    scaler = MinMaxScaler(feature_range=(-1,1))
    var = scaler.fit_transform(var.reshape(var.shape[0],-1)).reshape(var.shape)  #MinMax scaler vuole tensore 2D
    return var, scaler 



def transform_normalize_valid(var,fitted_scaler):
    """Apply a transformation and the min-max normalization.
    
    Parameters:
        var : numpy ndarray
              Array to be transformed and normalized.
    Returns:
        var : numpy ndarray
                The array transformed and normalized.
        scaler : scikit-learn estimator object
    """
    var = np.sqrt(var)
    var = fitted_scaler.transform(var.reshape(var.shape[0],-1)).reshape(var.shape)  #MinMax scaler vuole tensore 2D
    return var  


def inv_transform_normalize(tensor, scaler):

    shp = tensor.shape
    tensor_denorm = scaler.inverse_transform(tensor.reshape(shp[0],-1)).reshape(shp)
    tensor_fnl = tf.math.square(tensor_denorm)

    return tensor_fnl



def inv_transform_normalize_TF(tensor,frange,data_min,data_max):

    shp = tf.shape(tensor)

    rmin = tf.cast(frange[0],tf.float32)
    rmax = tf.cast(frange[1],tf.float32)

    tensor_tmp = tf.reshape(tensor,[shp[0],-1])
    inv_scale = tf.divide((data_max - data_min),(rmax - rmin))
    tensor_denorm_tmp = (tensor_tmp - rmin) * inv_scale + data_min
    tensor_denorm = tf.reshape(tensor_denorm_tmp,shp)

    tensor_fnl = tf.math.square(tensor_denorm)

    return tensor_fnl



