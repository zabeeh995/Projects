from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
import numpy as np
import argparse

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
                help="path to the input image")
ap.add_argument("-o", "--output", required=True,
                help="path to output directory to store augmentation examples")
ap.add_argument("-p", "--prefix", type=str, default="image",
                help="output filename prefix")
args = vars(ap.parse_args())

# load the input image, convert it to a NumPy array, and then
# reshape it to have an extra dimension
print("[INFO] loading example image...")
image = load_img(args["image"])
image = img_to_array(image)
image = np.expand_dims(image, axis=0)

# construct the image generator for data augmentation then
# initialize the total number of images generated thus far
aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
                         height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
                         horizontal_flip=True, fill_mode="nearest")
total = 0

#constructing the actual python image generator
imageGen = aug.flow(image, batch_size=1, save_to_dir=args["output"], save_prefix=args["prefix"],
                    save_format=".jpg")

for image in imageGen:
    total +=1
    if total == 10:
        break
