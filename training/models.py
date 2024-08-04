"""
Generator and discriminator models

Models are defined using the Keras sequential API
https://www.tensorflow.org/guide/keras sequential_model

The generator uses tf.keras.layers.Conv2DTranspose (upsampling) layers 
to produce an image from a seed (random noise).
Start with a Dense layer that takes this seed as input, 
then upsample several times until you reach the desired image size of 64x64x1.
Notice the tf.keras.layers.LeakyReLU activation for each layer, 
except the output layer which uses tanh.
"""

from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, LeakyReLU, Reshape, Dropout, Conv2DTranspose, UpSampling2D, Conv2D, Flatten
from tensorflow.keras.initializers import RandomNormal



def gen_deconv(ks=3,dropout_flag=False,dropout_rate=0.5):

    print(f"\nModel = gen_deconv")
    print(f"Kernel size = ({ks},{ks})")
    print(f"Dropout flag = {dropout_flag}")
    print(f"Dropout rate = {dropout_rate}\n")

    noise_dim = 113
    initializer = RandomNormal(mean=0., stddev=0.02)

    model = Sequential(name="gen_deconv")

    # input block
    model.add(Dense(units=4*4*512,use_bias=False,
                    kernel_initializer=initializer,
                    input_shape=(noise_dim + 64,),
                    name='Dense_1'))
    model.add(BatchNormalization(name='BN_1'))
    model.add(LeakyReLU(0.2,name='Leaky_ReLU_1'))
    model.add(Reshape((4,4,512)))
    if dropout_flag:
        model.add(Dropout(dropout_rate,name='Dropout_1'))

    assert model.output_shape == (None, 4, 4, 512)    #None is the batch size

    img_size = 4

    # deconv blocks   
    for i,n_filters in zip(range(1,5),[256,128,64,32]):
        
        img_size *= 2

        model.add(Conv2DTranspose(filters=n_filters,kernel_size=(ks,ks),
                                  strides=(2,2),padding='same',
                                  use_bias=False,kernel_initializer=initializer,
                                  name=f'Deconv_{i}'))
        model.add(BatchNormalization(name=f'BN_{i+1}'))
        model.add(LeakyReLU(0.2,name=f'LeakyReLU_{i+1}'))
        if dropout_flag:
            model.add(Dropout(dropout_rate,name=f'Dropout_{i+1}'))

        assert model.output_shape == (None, img_size, img_size, n_filters)

    # Out block
    model.add(Conv2D(filters=1,kernel_size=(ks,ks),
                     strides=(1,1),padding='same',
                     activation='tanh',
                     use_bias=False,kernel_initializer=initializer,
                     name='Conv_5'))

    assert model.output_shape == (None, 64, 64, 1)
    
    return model
  
 

def gen_UpConv(ks=3,dropout_flag=False,dropout_rate=0.5):

    print("\nModel = gen_UpConv")
    print(f"Kernel size = ({ks},{ks})")
    print(f"Dropout flag = {dropout_flag}")
    print(f"Dropout rate = {dropout_rate}\n")

    noise_dim = 113
    initializer = RandomNormal(mean=0., stddev=0.02)

    """
    if dropout_flag:
        m_name = f'Generator conv filters ({ks},{ks}) with dropout'
    else:
        m_name = f'Generator conv filters ({ks},{ks}) (NO dropout)'

    model = Sequential(name=m_name)
    """
    model = Sequential(name="gen_UpConv")
    
    # input block
    model.add(Dense(units=4*4*512,use_bias=False,
                    kernel_initializer=initializer,
                    input_shape=(noise_dim + 64,),
                    name='Dense_1'))
    model.add(BatchNormalization(name='BN_1'))
    model.add(LeakyReLU(0.2,name='Leaky_ReLU_1'))
    model.add(Reshape((4,4,512)))
    if dropout_flag:
        model.add(Dropout(dropout_rate,name='Dropout_1'))

    assert model.output_shape == (None, 4, 4, 512)    #None is the batch size

    img_size = 4

    # deconv blocks   
    for i,n_filters in zip(range(1,5),[256,128,64,32]):
        
        img_size *= 2
        
        model.add(UpSampling2D(size=(2,2),interpolation='nearest',
                               name=f'UpSampling_{i}'))
        model.add(Conv2D(filters=n_filters, kernel_size=(ks,ks),
                         strides=(1,1),padding='same',
                         use_bias=False,kernel_initializer=initializer,
                         name=f'Conv_{i}'))
        model.add(BatchNormalization(name=f'BN_{i+1}'))
        model.add(LeakyReLU(0.2,name=f'LeakyReLU_{i+1}'))
        if dropout_flag:
            model.add(Dropout(dropout_rate,name=f'Dropout_{i+1}'))

        assert model.output_shape == (None, img_size, img_size, n_filters)
       
    # Out block
    model.add(Conv2D(filters=1,kernel_size=(ks,ks),
                     strides=(1,1),padding='same',
                     activation='tanh',
                     use_bias=False,kernel_initializer=initializer,
                     name='Conv_5'))

    assert model.output_shape == (None, 64, 64, 1)
    
    return model



def discriminator(ks=3,dropout_flag=False,dropout_rate=0.5,FT_in=False):

    print("\nModel = discriminator")
    print(f"Kernel size = ({ks},{ks})")
    print(f"Dropout flag = {dropout_flag}")
    print(f"Dropout rate = {dropout_rate}\n")

    if FT_in:
        print('Using the Fourier transform as additional input channel\n')
        INPUT_SHAPE = [64,64,3]
    else:
        INPUT_SHAPE = [64,64,2]

    initializer = RandomNormal(mean=0., stddev=0.02)
    
    model = Sequential(name="discriminator")
    
    # Input block
    model.add(Conv2D(filters=64,kernel_size=(ks,ks),
                     strides=(2,2),padding='same',
                     use_bias=False,kernel_initializer=initializer,
                     input_shape=INPUT_SHAPE,
                     name='Conv_1'))    # For now adding just one dim to input
    model.add(LeakyReLU(0.2,name='LeakyReLU_1'))
    if dropout_flag:
        model.add(Dropout(dropout_rate,name='Dropout_1'))
    
    assert model.output_shape == (None, 32, 32, 64)

    img_size = 32

    # Conv blocks
    for i,n_filters in zip(range(2,5),[128,256,512]):

        img_size //=2

        model.add(Conv2D(filters=n_filters,kernel_size=(ks,ks),
                         strides=(2,2),padding='same',
                         use_bias=False,kernel_initializer=initializer,
                         name=f'Conv_{i}'))
        model.add(LeakyReLU(0.2,name=f'LeakyReLU_{i}'))
        if dropout_flag:
            model.add(Dropout(dropout_rate,name=f'Dropout_{i}'))
    
        assert model.output_shape == (None, img_size, img_size, n_filters)

    # Out block
    model.add(Conv2D(filters=1,kernel_size=(ks,ks),
                     strides=(1,1),padding='same'))

    model.add(Flatten())
    model.add(Dense(1))
    #model.add(Dense(1, activation='sigmoid'))

    assert model.output_shape == (None, 1)

    return model



