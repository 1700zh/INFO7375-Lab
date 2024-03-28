import string
from datageneration import generate_image_data
from model import Model
from activation import Activation
from loss_function import LossFunction
from training import Training

# Parameters for data generation
num_images = 1000  # Number of images to generate for training
image_size = (64, 64)  # Size of the generated images
font_path = "arial.ttf"  # Path to the font file used in generating images
font_size = 40  # Font size used in generating images

# Assuming generate_image_data returns features and labels
x, y = generate_image_data(num_images, image_size, font_path, font_size)

# Initialize the neural network model
model = Model()
# Define your model's architecture here. This is a hypothetical example.
model.add_layer("input", input_shape=image_size)
model.add_layer("conv2d", filters=32, kernel_size=(3, 3), activation=Activation.RELU)
model.add_layer("maxpool", pool_size=(2, 2))
model.add_layer("flatten")
model.add_layer("dense", units=128, activation=Activation.RELU)
model.add_layer("dense", units=len(string.ascii_uppercase), activation=Activation.SOFTMAX)  # Output layer

# Compile the model specifying the optimizer, loss function, and metrics
model.compile(optimizer='adam', loss=LossFunction.CATEGORICAL_CROSS_ENTROPY)

# Initialize the training process
training = Training(model)

# Specify the number of epochs for training
epochs = 200

# Train the model using the generated data
training.train(x, y, epochs=epochs, batch_size=32)

model.save("trained_model.h5")
