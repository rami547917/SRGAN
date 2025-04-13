import os
import cv2
import numpy as np
import numpy as np
from keras import Model
from keras.layers import Conv2D
from keras.layers import PReLU
from keras.layers import BatchNormalization 
from keras.layers import Flatten
from keras.layers import UpSampling2D
from keras.layers import LeakyReLU 
from keras.layers import Dense
from keras.layers import Input
from keras.layers import add
from tqdm import tqdm
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras import layers, Model
from sklearn.model_selection import train_test_split
from keras.applications import VGG19



# Define a residual block for the generator
def residual_block(input_tensor):

    conv1 = Conv2D(64, (3,3), padding = "same")(input_tensor)
    norm1 = BatchNormalization(momentum = 0.5)( conv1)
    act1 = PReLU(shared_axes = [1,2])(norm1)

    conv2 = Conv2D(64, (3,3), padding = "same")(act1)
    res_model = BatchNormalization(momentum = 0.5)(conv2)

    return add([input_tensor,res_model])

# Define an upscaling block for the generator

def upsample_block(input_tensor):

    conv = Conv2D(256, (3,3), padding="same")(input_tensor)
    upsampled  = UpSampling2D( size = 2 )(conv)
    activated = PReLU(shared_axes=[1,2])(upsampled )

    return activated

# Generator model construction
def build_generator(generator_input, num_residual_blocks):
    initial_conv = Conv2D(64, (9,9), padding="same")(generator_input)
    initial_act = PReLU(shared_axes=[1,2])(initial_conv)

    temp = initial_act

    for i in range(num_residual_blocks):
        initial_act = residual_block(initial_act)

    conv_mid = Conv2D(64, (3,3), padding="same")(initial_act)
    norm_mid = BatchNormalization(momentum=0.5)(conv_mid)
    merged = add([norm_mid,temp])

    upsampled1 = upsample_block(merged)
    upsampled2 = upsample_block(upsampled1)

    final_output = Conv2D(3, (9,9), padding="same")(upsampled2)

    return Model(inputs=generator_input, outputs=final_output)

# Block for constructing the discriminator
def build_discriminator_block(input_tensor, filters, strides=1, batch_norm=True):

    conv = Conv2D(filters, (3,3), strides = strides, padding="same")(input_tensor)

    if batch_norm:
        conv = BatchNormalization( momentum=0.8 )(conv)

    activated = LeakyReLU( alpha=0.2 )(conv)

    return activated


# Discriminator model construction
def build_discriminator(discriminator_input):

    filters  = 64

    block1 = build_discriminator_block(discriminator_input, filters , batch_norm=False)
    block2 = build_discriminator_block(block1, filters , strides=2)
    block3 = build_discriminator_block(block2, filters *2)
    block4 = build_discriminator_block(block3, filters *2, strides=2)
    block5 = build_discriminator_block(block4, filters *4)
    block6 = build_discriminator_block(block5, filters *4, strides=2)
    block7 = build_discriminator_block(block6, filters *8)
    block8 = build_discriminator_block(block7, filters *8, strides=2)

    flattened = Flatten()(block8)
    dense = Dense(filters *16)(flattened)
    leaky_relu = LeakyReLU(alpha=0.2)(dense)
    validity_output = Dense(1, activation='sigmoid')(leaky_relu)

    return Model(discriminator_input, validity_output)


# VGG19 model construction for feature extraction


def build_vgg_model(hr_shape):

    vgg = VGG19(weights="imagenet",include_top=False, input_shape=hr_shape)

    return Model(inputs=vgg.inputs, outputs=vgg.layers[10].output)

# Combining the generator, discriminator, and VGG19 models
def create_gan(generator_model, discriminator_model, vgg_model, lr_input, hr_input):
    generated_image = generator_model(lr_input)

    generated_features = vgg_model(generated_image)

    discriminator_model.trainable = False
    validity = discriminator_model(generated_image)

    return Model(inputs=[lr_input, hr_input], outputs=[validity, generated_features])

# Function to scale image values between 0 and 1
def  normalize_images(image_dict):
    normalized_dict = {}
    for key, image in image_dict.items():
        normalized_dict[key] = image / 255.0
    return normalized_dict

# Training function
def train_model(data_count,num_epochs):

    lr_filenames = os.listdir("drive/MyDrive/data/lr_images")[:data_count]

    lr_images_dict = {}
    for img in lr_filenames:
        img_lr = cv2.imread("drive/MyDrive/data/lr_images/" + img)
        img_lr = cv2.cvtColor(img_lr, cv2.COLOR_BGR2RGB)
        lr_images_dict[img] = img_lr


    hr_filenames = os.listdir("drive/MyDrive/data/hr_images")[:data_count]

    hr_images_dict = {}
    for img in hr_filenames:
        img_hr = cv2.imread("drive/MyDrive/data/hr_images/" + img)
        img_hr = cv2.cvtColor(img_hr, cv2.COLOR_BGR2RGB)
        hr_images_dict[img] = img_hr


    # Normalize image values
    lr_images_dict = normalize_images(lr_images_dict)
    hr_images_dict = normalize_images(hr_images_dict)

    # Prepare train and test sets
    lr_images = []
    hr_images = []

    for key in lr_images_dict:
        lr_images.append(lr_images_dict[key])
        hr_images.append(hr_images_dict[key])

    lr_images = np.array(lr_images)
    hr_images = np.array(hr_images)

    lr_train, lr_test, hr_train, hr_test = train_test_split(lr_images, hr_images,
                                                      test_size=0.1, random_state=42)
    hr_shape = (hr_train.shape[1], hr_train.shape[2], hr_train.shape[3])
    lr_shape = (lr_train.shape[1], lr_train.shape[2], lr_train.shape[3])

    lr_input = Input(shape=lr_shape)
    hr_input = Input(shape=hr_shape)

    generator = build_generator(lr_input, num_residual_blocks = 16)
    generator.summary()

    discriminator = build_discriminator(hr_input)
    discriminator.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy'])
    discriminator.summary()

    vgg_model = build_vgg_model((128,128,3))
    print(vgg_model.summary())
    vgg_model.trainable = False

    gan_model = create_gan(generator, discriminator, vgg_model, lr_input, hr_input)

# Compile the GAN model with adversarial and content losses
    gan_model.compile(loss=["binary_crossentropy", "mse"], loss_weights=[1e-3, 1], optimizer="adam")
    gan_model.summary()
 # Prepare batches for training
    batch_size = 1
    low_res_batches = []
    high_res_batches = []
    for it in range(int(hr_train.shape[0] / batch_size)):
        start_idx = it * batch_size
        end_idx = start_idx + batch_size
        high_res_batches.append(hr_train[start_idx:end_idx])
        low_res_batches.append(lr_train[start_idx:end_idx])
# Training loop over epochs
    for epoch in range(num_epochs):

        fake_label = np.zeros((batch_size, 1)) # Label for generated images
        real_label = np.ones((batch_size,1))  # Label for real images

    # Arrays to hold the losses for the generator and discriminator
        generator_losses = []
        discriminator_losses = []

    # Training loop over batches
        for batch in tqdm(range(len(high_res_batches))):
            low_res_images = low_res_batches[batch]  # Get a batch of low-res images  
            high_res_images = high_res_batches[batch] # Get a batch of high-res images  

            fake_imgs = generator.predict_on_batch(low_res_images) # Generate high-res images

        # Train the discriminator with real and generated images
            discriminator.trainable = True
            d_loss_fake = discriminator.train_on_batch(fake_imgs, fake_label)
            d_loss_real = discriminator.train_on_batch(high_res_images, real_label)

         # Train the generator while keeping the discriminator fixed
            discriminator.trainable = False

        # Average discriminator loss for reporting
            discriminator_loss = 0.5 * np.add(d_loss_fake, d_loss_real)

        # Extract VGG features for content loss calculation
            image_features = vgg_model.predict(high_res_images)

        # Use the GAN model to enhance the generator
            generator_loss, _, _ = gan_model.train_on_batch([low_res_images, high_res_images], [real_label, image_features])

        # Record losses
            discriminator_losses.append(discriminator_loss)
            generator_losses.append(generator_loss)

    # Calculate average losses
        generator_losses = np.array(generator_losses)
        discriminator_losses = np.array(discriminator_losses)

    
        avg_g_loss = np.sum(generator_losses, axis=0) / len(generator_losses)
        davg_d_loss = np.sum(discriminator_losses, axis=0) / len(discriminator_losses)

    # Report training progress
        print("epoch:", epoch+1 ,"avg_g_loss:", avg_g_loss, "davg_d_loss:", davg_d_loss)

	# Save the generator model after each epoch
        generator.save("gen_e_"+ str(epoch+1) +".h5")