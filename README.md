Super-Resolution GAN 

 This project tackle the challenging problem of image super-resolution using a Generative Adversarial Network (GAN). The primary goal is to convert low-resolution images into high-quality, detailed outputs. The project consists of three key components:

Generator:
I designed the generator with multiple residual blocks that utilize convolutional layers, batch normalization, and PReLU activation functions. These components work together to effectively upscale images, while dedicated upsampling layers double the resolution without losing critical details.

Discriminator:
The discriminator is built with several convolutional blocks that feature LeakyReLU activations and batch normalization. It differentiates between real high-resolution images and those produced by the generator, providing adversarial feedback that refines the generator’s performance.

VGG19 for Feature Extraction:
I incorporated a pre-trained VGG19 network to extract perceptual features. By comparing these features from the generated and real images, the system leverages perceptual loss, ensuring that the generated textures closely match real-world details.

This project is implemented in Python using TensorFlow/Keras, OpenCV, NumPy, and Matplotlib, combining robust deep learning techniques with efficient image processing.
