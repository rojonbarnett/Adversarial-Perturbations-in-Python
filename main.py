# RoJon Barnett
# Adversarial Perturbation
# Step 1: Import the required libraries
# Step 1: Import the required libraries
#Python library for creating charts diagram or manipulating images
import matplotlib.pyplot as plt
#Python library used for working with arrays.
import numpy as np
#Open source software library for machine learning models
import tensorflow as tf
#Python Imaging Library
# Image - module to manipulate image files
from PIL import Image

# Step 2: Load the ResNet50 model
#Neural network architecture - Microsoft
#Pretrained by the imagenet dataset
model = tf.keras.applications.ResNet50(weights='imagenet')

# Step 3: Load an image to classify
img_path = 'C:\InfoSecurityProject\cat.jpg'
#funciton from keras preprocessing module that resizes an image to
#optimal size for classification
img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))

# Step 4: Preprocess the image
#Converts the image to an array of height, width, channels -color
x = tf.keras.preprocessing.image.img_to_array(img)
#Adds an extra dimension to array for num of images, because
#Resnet expects to classify multiple images
x = np.expand_dims(x, axis=0)

x = tf.keras.applications.resnet50.preprocess_input(x)

# Step 5: Make a prediction using the ResNet50 model on the original image
preds = model.predict(x)
original_label = tf.keras.applications.resnet50.decode_predictions(preds, top=1)[0][0][1]
print("Original image label:", original_label)

# Step 6: Generate an adversarial example using the Fast Gradient Sign Method (FGSM)
#is the perturbation value
epsilon = 2
#Transforms numpy array to tensor array
#Can be used to perform operations on large datasets with parallel computing
#expects large dataset
x_tensor = tf.convert_to_tensor(x, dtype = tf.float32)

# Convert original label to one-hot encoded vector
#binary representation of categorical data - so resnet can understand
#the image
target = tf.one_hot([np.argmax(preds)], 1000)


#Gradient that records the operations that are executed on the target
#Important for cross entropy loss - diff b/n predicted probability - likelihood of correct guess based on training and true probability - likelihood based on the dataset
#Trying to increase the entropy loss when producing adversary
with tf.GradientTape() as tape:
    tape.watch(x_tensor)
    preds = model(x_tensor)
    loss = tf.keras.losses.categorical_crossentropy(target, preds)

gradients = tape.gradient(loss, x_tensor)
adv_x = x_tensor+ epsilon * tf.sign(gradients)
adv_x = tf.clip_by_value(adv_x, 0, 255)
adv_x = tf.cast(adv_x, tf.uint8).numpy()[0]

# Step 7: Preprocess the adversarial image
adv_img = Image.fromarray(adv_x)
adv_img = adv_img.resize((224, 224))
adv_x = tf.keras.preprocessing.image.img_to_array(adv_img)
adv_x = np.expand_dims(adv_x, axis=0)
adv_x = tf.keras.applications.resnet50.preprocess_input(adv_x)

# Step 8: Make a prediction using the ResNet50 model on the adversarial image
preds_adv = model.predict(adv_x)
perturbed_label = tf.keras.applications.resnet50.decode_predictions(preds_adv, top=1)[0][0][1]
print("Perturbed image label:", perturbed_label)

# Step 9: Display the original and adversarial images side by side with their predicted labels
plt.subplot(1, 2, 1)
plt.imshow(img)
plt.title("Original Image")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(adv_img)
plt.title("Perturbed Image")
plt.axis('off')

plt.suptitle(f"Original Label: {original_label}, Perturbed Label: {perturbed_label}")
plt.show()