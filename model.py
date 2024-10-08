
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers, Model
import tensorflow as tf
import numpy as np
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, PReLU,BatchNormalization, Flatten
from tensorflow.keras.layers import UpSampling2D, LeakyReLU, Dense, Input, add
from tqdm import tqdm
import cv2 
from PIL import Image 
from sklearn.model_selection import train_test_split


tf.keras.backend.clear_session()


#Define blocks to build the generator
def res_block(ip):
    
    res_model = Conv2D(64, (3,3), padding = "same")(ip)
    res_model = BatchNormalization(momentum = 0.5)(res_model)
    res_model = PReLU(shared_axes = [1,2])(res_model)
    
    res_model = Conv2D(64, (3,3), padding = "same")(res_model)
    res_model = BatchNormalization(momentum = 0.5)(res_model)
    
    return add([ip,res_model])

def upscale_block(ip):
    
    up_model = Conv2D(256, (3,3), padding="same")(ip)
    up_model = UpSampling2D( size = 2 )(up_model)
    up_model = PReLU(shared_axes=[1,2])(up_model)
    
    return up_model

#Generator model
def create_gen(gen_ip, num_res_block):
    layers = Conv2D(64, (9,9), padding="same")(gen_ip)
    layers = PReLU(shared_axes=[1,2])(layers)

    temp = layers

    for i in range(num_res_block):
        layers = res_block(layers)

    layers = Conv2D(64, (3,3), padding="same")(layers)
    layers = BatchNormalization(momentum=0.5)(layers)
    layers = add([layers,temp])

    layers = upscale_block(layers)
    # layers = upscale_block(layers)

    op = Conv2D(3, (9,9), padding="same")(layers)

    return Model(inputs=gen_ip, outputs=op)

#Descriminator block that will be used to construct the discriminator
def discriminator_block(ip, filters, strides=1, bn=True):
    
    disc_model = Conv2D(filters, (3,3), strides = strides, padding="same")(ip)
    
    if bn:
        disc_model = BatchNormalization( momentum=0.8 )(disc_model)
    
    disc_model = LeakyReLU( alpha=0.2 )(disc_model)
    
    return disc_model


#Descriminartor, as described in the original paper
def create_disc(disc_ip):

    df = 64
    
    d1 = discriminator_block(disc_ip, df, bn=False)
    d2 = discriminator_block(d1, df, strides=2)
    d3 = discriminator_block(d2, df*2)
    d4 = discriminator_block(d3, df*2, strides=2)
    d5 = discriminator_block(d4, df*4)
    d6 = discriminator_block(d5, df*4, strides=2)
    d7 = discriminator_block(d6, df*8)
    d8 = discriminator_block(d7, df*8, strides=2)
    
    d8_5 = Flatten()(d8)
    d9 = Dense(df*16)(d8_5)
    d10 = LeakyReLU(alpha=0.2)(d9)
    validity = Dense(1, activation='sigmoid')(d10)

    return Model(disc_ip, validity)


#VGG19 
#We need VGG19 for the feature map obtained by the j-th convolution (after activation) 
#before the i-th maxpooling layer within the VGG19 network.(as described in the paper)
#Let us pick the 3rd block, last conv layer. 
#Build a pre-trained VGG19 model that outputs image features extracted at the
# third block of the model
# VGG architecture: https://github.com/keras-team/keras/blob/master/keras/applications/vgg19.py
from tensorflow.keras.applications import VGG19

def build_vgg(hr_shape):
    
    vgg = VGG19(weights="imagenet",include_top=False, input_shape=hr_shape)
    
    return Model(inputs=vgg.inputs, outputs=vgg.layers[10].output)

#Combined model
def create_comb(gen_model, disc_model, vgg, lr_ip, hr_ip):
    gen_img = gen_model(lr_ip)
    
    gen_features = vgg(gen_img)
    
    disc_model.trainable = False
    validity = disc_model(gen_img)
    
    return Model(inputs=[lr_ip, hr_ip], outputs=[validity, gen_features])









def finalmodel():
    lr_shape = (128, 128, 3)

    hr_shape = (256, 256, 3)

    lr_ip = Input(shape=lr_shape)
    hr_ip = Input(shape=hr_shape)

    generator = create_gen(lr_ip, num_res_block = 16)
    # generator.summary()

    discriminator = create_disc(hr_ip)
    discriminator.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy'])


    vgg = build_vgg(hr_shape)
    # print(vgg.summary())
    vgg.trainable = False

    gan_model = create_comb(generator, discriminator, vgg, lr_ip, hr_ip)
    gan_model.compile(loss=["binary_crossentropy", "mse"], loss_weights=[1e-3, 1], optimizer="adam")
    print(gan_model.summary())
    return gan_model



if __name__== "__main__":
    lr_shape = (128, 128, 3)

    hr_shape = (256, 256, 3)

    # hisresimgfolder = "/media/abhishek/r/satproject/hilowres/highresImg"
    # lowresimgfolder = "/media/abhishek/r/satproject/hilowres/lowresImg"

    # totalLowimgs  = os.listdir(lowresimgfolder)
    # totalHighimgs = os.listdir(hisresimgfolder)

    # assert len(totalHighimgs) == len(totalLowimgs)

    # npArrLow = np.zeros((len(totalLowimgs),128,128,3))
    # npArrHigh = np.zeros((len(totalHighimgs),256,256,3))


    # for i in range(len(totalHighimgs)):

    #     limg = cv2.imread(os.path.join(lowresimgfolder,totalLowimgs[i]))
    #     limg = cv2.cvtColor(limg, cv2.COLOR_BGR2RGB)

    #     himg = cv2.imread(os.path.join(hisresimgfolder,totalHighimgs[i]))
    #     himg = cv2.cvtColor(himg, cv2.COLOR_BGR2RGB)

    #     npArrLow[i] = limg
    #     npArrHigh[i] = himg

    
    # np.save('/media/abhishek/r/satproject/hilowres/npArrLow.npy', npArrLow) 
    # np.save('/media/abhishek/r/satproject/hilowres/npArrHigh.npy', npArrHigh) 


    npArrHigh = np.load("/media/abhishek/r/satproject/hilowres/npArrHighNew.npy")
    npArrLow = np.load('/media/abhishek/r/satproject/hilowres/npArrLowNew.npy')


    lr_train, lr_test, hr_train, hr_test = train_test_split(npArrLow, npArrHigh, 
                                                      test_size=0.33, random_state=42)
    

    #gan_model = finalmodel()

  

    lr_shape = (128, 128, 3)

    hr_shape = (256, 256, 3)

    lr_ip = Input(shape=lr_shape)
    hr_ip = Input(shape=hr_shape)

    generator = create_gen(lr_ip, num_res_block = 16)
    # generator.summary()

    discriminator = create_disc(hr_ip)
    discriminator.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy'])


    vgg = build_vgg(hr_shape)
    # print(vgg.summary())
    vgg.trainable = False

    gan_model = create_comb(generator, discriminator, vgg, lr_ip, hr_ip)
    gan_model.compile(loss=["binary_crossentropy", "mse"], loss_weights=[1e-3, 1], optimizer="adam")


    batch_size = 1  
    train_lr_batches = []
    train_hr_batches = []
    for it in range(int(hr_train.shape[0] / batch_size)):
        start_idx = it * batch_size
        end_idx = start_idx + batch_size
        train_hr_batches.append(hr_train[start_idx:end_idx])
        train_lr_batches.append(lr_train[start_idx:end_idx])
        
        
    epochs = 50

    for e in range(epochs):
    
        fake_label = np.zeros((batch_size, 1)) # Assign a label of 0 to all fake (generated images)
        real_label = np.ones((batch_size,1)) # Assign a label of 1 to all real images.
        
        #Create empty lists to populate gen and disc losses. 
        g_losses = []
        d_losses = []
        
        #Enumerate training over batches. 
        for b in tqdm(range(len(train_hr_batches))):
            lr_imgs = train_lr_batches[b] #Fetch a batch of LR images for training
            hr_imgs = train_hr_batches[b] #Fetch a batch of HR images for training
            
            fake_imgs = generator.predict_on_batch(lr_imgs) #Fake images
            
            #First, train the discriminator on fake and real HR images. 
            discriminator.trainable = True
            d_loss_gen = discriminator.train_on_batch(fake_imgs, fake_label)
            d_loss_real = discriminator.train_on_batch(hr_imgs, real_label)
            
            #Now, train the generator by fixing discriminator as non-trainable
            discriminator.trainable = False
            
            #Average the discriminator loss, just for reporting purposes. 
            d_loss = 0.5 * np.add(d_loss_gen, d_loss_real) 
            
            #Extract VGG features, to be used towards calculating loss
            image_features = vgg.predict(hr_imgs)
        
            #Train the generator via GAN. 
            #Remember that we have 2 losses, adversarial loss and content (VGG) loss
            g_loss, _, _ = gan_model.train_on_batch([lr_imgs, hr_imgs], [real_label, image_features])
            
            #Save losses to a list so we can average and report. 
            d_losses.append(d_loss)
            g_losses.append(g_loss)
            
        #Convert the list of losses to an array to make it easy to average    
        g_losses = np.array(g_losses)
        d_losses = np.array(d_losses)
        
        #Calculate the average losses for generator and discriminator
        g_loss = np.sum(g_losses, axis=0) / len(g_losses)
        d_loss = np.sum(d_losses, axis=0) / len(d_losses)
        
        #Report the progress during training. 
        print("epoch:", e+1 ,"g_loss:", g_loss, "d_loss:", d_loss)

        if (e+1) % 10 == 0: #Change the frequency for model saving, if needed
            #Save the generator after every n epochs (Usually 10 epochs)
            generator.save("/media/abhishek/r/satproject/hilowres/gen_e_"+ str(e+1) +".h5")






