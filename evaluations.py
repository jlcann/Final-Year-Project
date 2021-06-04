import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

def loadModel(modelName):
  from keras.models import load_model
  try:
    model = load_model(modelName)
    print(f'Model: {modelName} loaded successfully')
    return model
  except Exception as e: 
    print(f'Model Failed to load: \n{e}')

def plotTrainingHistory(hist):

    history = np.load(hist, allow_pickle=True).item()

    max_acc = history['accuracy'][(len(history['accuracy'])-1)]
    max_val = history['val_accuracy'][(len(history['val_accuracy'])-1)]

    min_acc = history['loss'][(len(history['loss'])-1)]
    min_val = history['val_loss'][(len(history['accuracy'])-1)]

    print(f'Final Training Accuracy: {max_acc} \n' )
    print(f'Final Validation Accuracy: {max_val} \n')

    print(f'Final Training Loss: {min_acc} \n')
    print(f'Final Validation Loss: {min_val} \n \n \n ')

    plt.plot(history['accuracy'])
    plt.plot(history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    # "Loss"
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()

def plotConfusionMatrix(true_classes, predicted_classes, problem_type = 'Multi-Class'):

    if problem_type == 'Multi-Class':

        predicted_classes = np.argmax(predicted_classes, axis=1)
        true_classes = [np.where(r==1)[0][0] for r in true_classes]

        labels = ["Glioma", "Meningioma", "None","Pituitary"]

        conf = confusion_matrix(true_classes, predicted_classes)

        df_cm = pd.DataFrame(conf, index = [i for i in labels],
                        columns = [i for i in labels])
        # plt.figure(figsize=(10,7))
        sn.set(font_scale=1.4) # for label size
        ax = sn.heatmap(df_cm, cmap='Oranges', annot=True, fmt='g')

        return conf

    if problem_type == 'Binary':

        predicted_classes = binarize(np.argmax(predicted_classes, axis=1))
        true_classes = binarize([np.where(r==1)[0][0] for r in true_classes])

        labels = ["Tumour", "No Tumour"]

        conf = confusion_matrix(true_classes, predicted_classes)

        df_cm = pd.DataFrame(conf, index = [i for i in labels],
                        columns = [i for i in labels])
        # plt.figure(figsize=(10,7))
        sn.set(font_scale=1.4) # for label size
        ax = sn.heatmap(df_cm, cmap='Oranges', annot=True, fmt='g')

        return conf

        
    
def classificationReport(true_classes, predicted_classes, problem_type = 'Multi-Class'):



    if problem_type == 'Multi-Class':

        predicted_classes = np.argmax(predicted_classes, axis=1)
        true_classes = [np.where(r==1)[0][0] for r in true_classes]

        labels = ["Glioma", "Meningioma", "None","Pituitary"]

        prediction_report = classification_report(true_classes, predicted_classes, target_names=labels)

        return prediction_report


    if problem_type == 'Binary':

        predicted_classes = binarize(np.argmax(predicted_classes, axis=1))
        true_classes = binarize([np.where(r==1)[0][0] for r in true_classes])


        labels = ["Tumour", "No Tumour"]

        prediction_report = classification_report(true_classes, predicted_classes, target_names=labels)

        return prediction_report

def generatePredictions(model_name, testing_Data, weights="None"):

    model = loadModel(model_name)

    if weights != "None":
        model.load_weights(weights)
        print(f'{weights}: weights loaded')

    predicted_classes = model.predict(testing_Data)

    return predicted_classes

def loadTestingData(testing_images, testinglabels):

    import numpy as np

    testing_im = np.load(testing_images, allow_pickle=True)
    testing_lab = np.load(testinglabels, allow_pickle=True)

    return [testing_im, testing_lab]

def binarize(label_arr):
    labels = []
    for i in range(0, len(label_arr)):
        if label_arr[i] in [0,1,3]:
            labels.append(0)
        if label_arr[i] == 2:
            labels.append(1)
    return labels
    

def calculateBinaryScore(predicted_classes, true_classes):
    
    predicted_classes = np.argmax(predicted_classes, axis=1)
    true_classes = [np.where(r==1)[0][0] for r in true_classes]

    correctly_classified = 0
    
    binary_predicted = binarize(predicted_classes)
    binary_truth = binarize(true_classes)
    
    for i in range(0, len(binary_predicted)):
        if binary_predicted[i] == binary_truth[i]:
            correctly_classified += 1
    
    score = correctly_classified / len(binary_predicted)
    
    print(f'Non-Rounded Binary Classification Score: {score}')
    
    return score
    
