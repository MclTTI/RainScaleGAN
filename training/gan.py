import numpy as np

import tensorflow as tf
from tensorflow import keras

import preprocess

#############################
# Seed for the generator
SEED=42
#############################


class GAN(keras.Model):

    def __init__(self,discriminator,generator,noise_dim):

        super(GAN, self).__init__()
        
        self.discriminator = discriminator
        self.generator = generator
        self.noise_dim = noise_dim

        print(f"Using class GAN")



    def compile(self,d_optimizer,g_optimizer,metr):
        
        super(GAN, self).compile()

        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer

        self.d_loss_fn = discriminator_loss
        self.g_loss_fn = generator_loss

        self.metr = metr

        
    def train_step(self,train_batch):
        """Train the models with a batch of images.
        
        Parameters:
            train_batch :
                          A batch of training examples.
                          Each batch is composed by 
                          (img HR, img coarsened, img coarsened nn remapped)
            
        Returns:
            g_loss : 
                     Generator loss
            d_loss : 
                     Discriminator loss (total)
            d_loss_real :
                          Discriminator loss for real images
            d_loss_fake : 
                          Discriminator loss for fake images
            d_probs_real : 
                           Average probability for real images
            d_probs_fake : 
                           Average probability for fake images        
        """

        #Unpack data
        img,img_c,img_cnn = train_batch
        
        #print(img.shape, img_c.shape, img_cnn.shape)
        
        # Generate noise
        BATCH_SIZE = tf.shape(img_c)[0]
        noise = tf.random.normal([BATCH_SIZE, self.noise_dim],seed=SEED)
        
        with tf.GradientTape() as gen_tape, tf.GradientTape() as discr_tape:
            
            # Generate image
            noise_plus_coarse = tf.concat((noise, tf.reshape(img_c, [BATCH_SIZE, 64])), 1)
            gen_imgs = self.generator(noise_plus_coarse, training=True)
        
            # Train the discriminator
            true_IN = tf.concat((img, img_cnn), 3)    # true images + coarse
            true_OUT = self.discriminator(true_IN, training=True) 
            
            fake_IN = tf.concat((gen_imgs, img_cnn), 3)    # generated images + coarse
            fake_OUT = self.discriminator(fake_IN, training=True)

            # Compute the losses
            d_loss, d_loss_real, d_loss_fake = self.d_loss_fn(true_OUT,fake_OUT)
            g_loss = self.g_loss_fn(fake_OUT)

        # Compute gradients and update networks weights
        gen_grad = gen_tape.gradient(g_loss, self.generator.trainable_variables)
        discr_grad = discr_tape.gradient(d_loss, self.discriminator.trainable_variables)

        self.g_optimizer.apply_gradients(zip(gen_grad, self.generator.trainable_variables))
        self.d_optimizer.apply_gradients(zip(discr_grad, self.discriminator.trainable_variables))

        # true_OUT and fake_OUT are the discriminator scores for the images in a batch.
        # Below I convert the score in a probability and compute the average

        d_probs_real = tf.reduce_mean(tf.sigmoid(true_OUT))
        d_probs_fake = tf.reduce_mean(tf.sigmoid(fake_OUT))

        return {"g_loss": g_loss,
                "d_loss": d_loss,
                "d_loss_real": d_loss_real,
                "d_loss_fake": d_loss_fake,
                "d_probs_real": d_probs_real,
                "d_probs_fake": d_probs_fake}

        
    def predict_step(self, data_batch):
        """Run the GAN in inference mode (generator only).
        
        Parameters:
            data_batch :
                          A batch of examples.
                          Each batch is composed by 
                          (img HR, img coarsened, img coarsened nn remapped)
            
        Returns:
            gen_imgs : 
                       Batch of HR images generated from the 
                       coarsened images given as input       
        """

        img, img_c, img_cnn = data_batch

        
        # Generate noise
        BATCH_SIZE = tf.shape(img_c)[0]
        noise = tf.random.normal([BATCH_SIZE, self.noise_dim],seed=SEED)

        print('predict_step ',self.generator)
        # Generate HR images from coarse images
        noise_plus_coarse = tf.concat((noise, tf.reshape(img_c, [BATCH_SIZE, 64])), 1)
        generated_imgs = self.generator(noise_plus_coarse, training=False)

        # Update metrics
        #print(self.metr)
        self.metr.update_state(img,generated_imgs)
        #print(self.metr.result())

        return generated_imgs



class WGANGP(GAN):

    def __init__(self,
                 discriminator,
                 generator,
                 noise_dim,
                 discriminator_extra_steps=3,
                 gp_weight=10):

        super().__init__(discriminator, generator, noise_dim)

        self.d_steps = discriminator_extra_steps
        self.gp_weight = gp_weight

        print(f"Using class WGANGP")
        

    def compile(self,d_optimizer,g_optimizer,metr):
                
        super(WGANGP, self).compile(d_optimizer,g_optimizer,metr)
        
        self.d_loss_fn = wasserstein_dloss
        self.g_loss_fn = wasserstein_gloss

        
    def gradient_penalty(self,real_images,fake_images):
        """Calculates the gradient penalty.
        This loss is calculated on an interpolated image
        and added to the discriminator loss.
        """

        batch_size = tf.shape(real_images)[0]
      
        real_hr = real_images[:,:,:,0]
        real_coarse = real_images[:,:,:,1]
        fake_hr = fake_images[:,:,:,0]
        fake_coarse = fake_images[:,:,:,1]

        
        # Get the interpolated image     
        alpha = tf.random.uniform([batch_size,1,1], 0.0, 1.0, seed=SEED)
        #print(alpha.shape)     
        
        #print(real_hr.shape)
        diff = fake_hr - real_hr
        #print(diff.shape)

        interpolated = real_hr + alpha * diff    # ! in Gulrajani (2017) la sampling distribution è al contrario!
        #print(interpolated.shape)

        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)

            # 1. Get the discriminator output for this interpolated image.
            interpolated = tf.expand_dims(interpolated, -1)
            real_coarse = tf.expand_dims(real_coarse, -1)

            interp_plus_coarse = tf.concat((interpolated, real_coarse),3)
            pred = self.discriminator(interp_plus_coarse,training=True)


        # 2. Calculate the gradients w.r.t to this interpolated image.
        grads = gp_tape.gradient(pred, [interpolated])[0]
        # 3. Calculate the norm of the gradients.
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        
        return gp


    def train_step(self,train_batch):
        """Train the models with a batch of images.
        
        Parameters:
            train_batch :
                          A batch of training examples.
                          Each batch is composed by 
                          (img HR, img coarsened, img coarsened nn remapped)
            
        Returns:
            g_loss : 
                     Generator loss
            d_loss : 
                     Discriminator loss (total)
            d_loss_real :
                          Discriminator loss for real images
            d_loss_fake : 
                          Discriminator loss for fake images
            d_probs_real : 
                           Average probability for real images
            d_probs_fake : 
                           Average probability for fake images     
        """

        #Unpack data
        img,img_c,img_cnn = train_batch
        
        #print(img.shape, img_c.shape, img_cnn.shape)
        
        BATCH_SIZE = tf.shape(img_c)[0]
        #print(BATCH_SIZE)
        
        # For each batch, we are going to perform the
        # following steps as laid out in the original paper:
        # 1. Train the generator and get the generator loss
        # 2. Train the discriminator and get the discriminator loss
        # 3. Calculate the gradient penalty
        # 4. Multiply this gradient penalty with a constant weight factor
        # 5. Add the gradient penalty to the discriminator loss
        # 6. Return the generator and discriminator losses as a loss dictionary

        # Train the discriminator first. The original paper recommends training
        # the discriminator for `x` more steps (typically 5) as compared to
        # one step of the generator. Here we will train it for 3 extra steps
        # as compared to 5 to reduce the training time.
            
        for i in range(self.d_steps):

            # Create the noise vector
            noise = tf.random.normal([BATCH_SIZE, self.noise_dim],seed=SEED)
            
            with tf.GradientTape() as discr_tape:
            
                # Generate images
                noise_plus_coarse = tf.concat((noise, tf.reshape(img_c, [BATCH_SIZE, 64])), 1)
                gen_imgs = self.generator(noise_plus_coarse, training=True)

                true_IN = tf.concat((img, img_cnn), 3)
                fake_IN = tf.concat((gen_imgs, img_cnn), 3)
                
                # Get the logits for real and fake images
                real_OUT = self.discriminator(true_IN, training=True)  
                fake_OUT = self.discriminator(fake_IN, training=True)     
                #print(real_output.shape)

                # Compute the discriminator loss
                d_cost, d_loss_real, d_loss_fake = self.d_loss_fn(real_OUT,fake_OUT)
                
                # Compute the gradient penalty
                gp = self.gradient_penalty(real_images=true_IN, fake_images=fake_IN)
                
                # Add the gradient penalty to the original discriminator loss
                d_loss = d_cost + gp * self.gp_weight

            # Compute gradients and update discriminator weights   
            d_gradient = discr_tape.gradient(d_loss, self.discriminator.trainable_variables)
            self.d_optimizer.apply_gradients(zip(d_gradient, self.discriminator.trainable_variables))
            
        # Train the generator    
        G_noise = tf.random.normal([BATCH_SIZE, self.noise_dim],seed=SEED)
            
        with tf.GradientTape() as gen_tape:
        
            # Generate fake images using the generator
            G_noise_plus_coarse = tf.concat((G_noise, tf.reshape(img_c, [BATCH_SIZE, 64])), 1)
            G_gen_imgs = self.generator(G_noise_plus_coarse, training=True)

            G_fake_IN = tf.concat((G_gen_imgs, img_cnn), 3)

            # Get the discriminator logits for generated images       
            G_fake_OUT = self.discriminator(G_fake_IN, training=True)
            
            # Calculate the generator loss
            g_loss = self.g_loss_fn(G_fake_OUT)
                
        # Compute gradients and update generator weights
        gen_gradient = gen_tape.gradient(g_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(zip(gen_gradient, self.generator.trainable_variables))

        # true_OUT and fake_OUT are the discriminator scores for the images in a batch.
        # Below I convert the score in a probability and compute the average
        d_probs_real = tf.reduce_mean(tf.sigmoid(real_OUT))
        d_probs_fake = tf.reduce_mean(tf.sigmoid(fake_OUT))

        return {"g_loss": g_loss,
                "d_loss": d_loss,
                "d_loss_real": d_loss_real,
                "d_loss_fake": d_loss_fake,
                "d_probs_real": d_probs_real,
                "d_probs_fake": d_probs_fake}
    

    
###################
#     Losses 
###################


def discriminator_loss(real_output, fake_output):
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    real_loss = cross_entropy(tf.ones_like(real_output),real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output),fake_output)
    total_loss = real_loss + fake_loss

    #tf.print(f'Discriminator losses:\nReal: {real_loss:.4f}, Fake: {fake_loss:.4f}, Total: {total_loss:.4f}')
    return total_loss, real_loss, fake_loss


def wasserstein_dloss(real_output,fake_output):
    real_loss = tf.reduce_mean(real_output)
    fake_loss = tf.reduce_mean(fake_output)
    total_loss = fake_loss - real_loss

    # Check for inf values
    #if tf.reduce_any(tf.math.is_inf(real_loss)) or tf.reduce_any(tf.math.is_inf(fake_loss)):
    #    print('real loss or fake loss are inf')
    #    print(real_output)
    #    print(fake_output)
                
    #    exit()


    return total_loss, real_loss, fake_loss


def generator_loss(fake_output):
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    g_loss = cross_entropy(tf.ones_like(fake_output),fake_output)

    #tf.print(f'Generator loss: {g_loss:.4f}')
    return g_loss


def wasserstein_gloss(fake_output):
    g_loss = -tf.reduce_mean(fake_output)
    return g_loss

