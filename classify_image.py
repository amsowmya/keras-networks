from keras.applications import VGG16
from keras.applications import VGG19
from keras.applications import ResNet50
from keras.applications import InceptionV3
from keras.applications import Xception
from keras.applications import imagenet_utils
from keras.applications.inception_v3 import preprocess_input
from keras.utils import img_to_array
from keras.utils import load_img
import numpy as np
import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str, default="images/img4.jpg", help="path to the input image")
ap.add_argument("-m", "--model", type=str, default="resnet", help="path to the pre-trained network")
args = vars(ap.parse_args())

MODELS = {
    "vgg16": VGG16,
    "vgg19": VGG19,
    "resnet": ResNet50,
    "inception": InceptionV3,
    "xception": Xception
}

if args["model"] not in MODELS.keys():
    raise AssertionError("The --model command line argument should be a key in the 'MODELS' dictionary")

inputShape = (224, 224)
preprocess = imagenet_utils.preprocess_input

if args["model"] in ("inception", "xception"):
    inputShape = (299, 299)
    preprocess = preprocess_input

print(f"[INFO] loading {args['model']}...")
Network = MODELS[args["model"]]
model = Network(weights="imagenet")

print("[INFO] loading and pre-processing image...")
image = load_img(args["image"], target_size=inputShape)
image = img_to_array(image)

image = np.expand_dims(image, axis=0)

# pre-process the image using the appropriate function based on the model that has been loaded
# (i.e., mean subtraction, scaling, etc..)
image = preprocess(image)

# classify the image
print(f"[INFO] classifying image with {args['model']}")
preds = model.predict(image)
P = imagenet_utils.decode_predictions(preds)

# loop over the predictions and display the rank-5 predictions + probabilities to our terminal
for (i, (imagenetID, label, prob)) in enumerate(P[0]):
    print(f"{i+1}. {label}: {prob * 100:.2f}%")

orig = cv2.imread(args["image"])
orig = cv2.resize(orig, (640, 480))
(imagenetID, label, prob) = P[0][0]
cv2.putText(orig, f"Label: {label}, {prob:.2f}%", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
cv2.imshow("image", orig)
cv2.waitKey(0)