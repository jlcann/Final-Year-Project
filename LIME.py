import skimage.segmentation
import numpy as np
import cv2
import matplotlib.pyplot as plt
import keras
import copy
import sklearn
import sklearn.metrics
from sklearn.linear_model import LinearRegression

from PIL import Image
from skimage.color import rgb2gray
from keras.models import load_model
  


def generateSuperpixels(image, dims, kSize, mDist, ratio):
  #array to hold image
  predictArr = []

  #Open image and resize it
  img = np.array(Image.open(image))
  img = cv2.resize(img, dsize=dims,interpolation=cv2.INTER_CUBIC).astype(np.float32)
  img = np.asarray(img)
  img /= 255

  #Mode image into Numpy array with appropriate dimentions
  predictArr = np.asarray(predictArr)
  predictArr = np.append(predictArr, img)
  predictArr = predictArr.reshape(1,256,256,3)

  #Generate the superpixels over the image
  superpixels = skimage.segmentation.quickshift(predictArr[0], kernel_size=kSize, max_dist=mDist, ratio=ratio)

  #Assign number of superpixels generated to variable
  num_superpixels_brain = np.unique(superpixels).shape[0]

  #Show image with superpixel mask
  plt.imshow(skimage.segmentation.mark_boundaries(predictArr[0], superpixels))
  print(f'Number of superpixels generated: {num_superpixels_brain} \nMask:')

  return num_superpixels_brain, superpixels

def generateSuperpixelsSLIC(image, dims, numSegs, sig):
  #array to hold image
  predictArr = []

  #Open image and resize it
  img = np.array(Image.open(image))
  img = cv2.resize(img, dsize=dims,interpolation=cv2.INTER_CUBIC).astype(np.float32)
  img = np.asarray(img)/255

  #Mode image into Numpy array with appropriate dimentions
  predictArr = np.asarray(predictArr)
  predictArr = np.append(predictArr, img)
  predictArr = predictArr.reshape(1,256,256,3)

  #Generate the superpixels over the image
  superpixels = skimage.segmentation.slic(predictArr[0], numSegs, sig)

  #Assign number of superpixels generated to variable
  num_superpixels_brain = np.unique(superpixels).shape[0]

  #Show image with superpixel mask
  plt.imshow(skimage.segmentation.mark_boundaries(predictArr[0], superpixels))
  print(f'Number of superpixels generated: {num_superpixels_brain} \nMask:')

  return num_superpixels_brain, superpixels

def generateGrayImage(image, dims):

  #Open the image, resize, normalise and convert to Grayscale. 
  img = np.array(Image.open(image))
  new_img = cv2.resize(img, dsize=dims,interpolation=cv2.INTER_CUBIC).astype(np.float32)
  new_img = np.asarray(new_img)
  new_img /= 255
  new_img = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)

  #Create NumpyArray and modify shape for model input.
  to_pred_Gray = []
  to_pred_Gray = np.asarray(to_pred_Gray)
  to_pred_Gray = np.append(to_pred_Gray, new_img)
  to_pred_Gray = to_pred_Gray.reshape(1,256,256,1)

  return to_pred_Gray

def getImageClass(model, imageArr):

    #image_classes = {0: "Glioma", 1: "Meningioma", 2: "No Tumour", 3: "Pituitary"}
    preds = model.predict(imageArr)
    predClass = np.argmax(preds, axis=1)
    print(f'Image Prediction Successfull, Confidences:\n{[round(i, 5) for i in preds[0]]}\nPredicted Class:\n{predClass[0]}')

    return preds, predClass[0]


def decodePredictions(predClass, preds=[]):

    classes = {0: "Glioma Tumour", 1: "Meningioma Tumour", 2: "No Tumour", 3: "Pituitary Tumour"}
    predictedClass = classes[predClass]
    print(f'The Predicted Class is: {predictedClass}')

    return predictedClass

def perturbate(image, perturbation, superpixels):
      #Obtain an array of indexes which correspond to each one in the binary perturbation array.
      active_pixels = np.where(perturbation == 1)[0]

      #Create a mask of the same size of the original image, with all values set to 0.
      mask = np.zeros(superpixels.shape)

      #For each of the active_pixels:
      for active in active_pixels:

        #Change each value in mask to 1 if corresponding superpixel is present in the active_pixels array.
        mask[superpixels == active] = 1

      #Create a copy of the original greyscale image
      perturb_image = copy.deepcopy(image)

      #Multiply the mask by the new copied image. Any pixel multiplied by a 1 in mask will retain it's value, otherwise
      #it is turned off.
      perturb_image = perturb_image*mask[:, :, np.newaxis]
      
      #return the peturbed image
      return perturb_image

def generatePertubations(num_pertubations, image, superpixels, num_superpixels, model):

  #Array to store all pertubation perdictions
  pertubationPredictions = []
  #Array to store perturbed images
  perturbed_Images = []
  #Create the binary arrays which correspond to each of the perturbations
  perturbations = np.random.binomial(1, 0.5, size=(num_pertubations, num_superpixels))
  #Loop through each of the binary arrays created
  for perturbation in perturbations:
    #Create a perturbed image
    perturb_image = perturbate(image, perturbation, superpixels)
    #Append new image to an array
    perturbed_Images.append(perturb_image)
    #Collect classification result from model of the new image
    prediction = model.predict(perturb_image[np.newaxis, :, :, :])
    #Store the result of the models classification on the perturbated image
    pertubationPredictions.append(prediction)
  #Return the predictions, perturbated images and the binary arrays.

  return np.asarray(pertubationPredictions), np.asarray(perturbed_Images), perturbations


def generateDistances(numSuperpixels, pertubations):

  #Create an 'original image' with the length of superpixels, all turned on.
  original_image = np.ones(numSuperpixels)[np.newaxis, :]

  #Use the sklearn library to generate the Euclidean distance between original image and each perturbation.
  distances = sklearn.metrics.pairwise_distances(pertubations, original_image, metric='cosine').ravel()

  #Return the distances 
  return distances

def fitLinModel(k_width, distances, class_to_explain, 
                pertubation_predictions, perturbations):

  #Define the kernel width
  kernel_width = k_width
  #Calculate the weights for the model
  weights = np.sqrt(np.exp(-(distances**2)/kernel_width**2))
  #The index of the class to be explained
  class_to_explain = class_to_explain
  #Create a linear regression model
  simpler_model = LinearRegression()
  #Fit the model using the perturbations, the predictions associated with those 
  # predictions and the calculated weights.
  simpler_model.fit(X=perturbations, y=pertubation_predictions
                    [:,:,class_to_explain], sample_weight=weights)
  #Record the coefficients from the model
  #Each coefficient in the model represents a superpixel
  coeff = simpler_model.coef_[0]
  #Return the coefficients

  return coeff

def showExplanation(num_top_features, coeff, num_superpixels, gray_image, superpixels):

  #Sort the coefficients and keep the top number of features defined
  top_features = np.argsort(coeff)[-num_top_features:]

  #Create a new mask
  mask = np.zeros(num_superpixels)

  #Turn on the super pixels associated with the top coefficients
  mask[top_features] = True

  #Show the explained image
  plt.imshow(perturbate(gray_image, mask, superpixels))

  #return the explained image
  return perturbate(gray_image, mask, superpixels)
