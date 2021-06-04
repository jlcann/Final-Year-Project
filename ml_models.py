

def loadModel(modelName):
  from keras.models import load_model
  try:
    model = load_model(modelName)
    print(f'Model: {modelName} loaded successfully')
    return model
  except Exception as e: 
    print(f'Model Failed to load: \n{e}')




def createCustomModel():
    
  from keras.models import Model
  from keras.models import Sequential
  from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten

  model = Sequential()
  model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(256,256,1)))
  model.add(Conv2D(32, (3, 3), activation='relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Dropout(0.5))

  model.add(Conv2D(64, (3, 3), padding='valid', activation='relu'))
  model.add(Conv2D(64, (3, 3), activation='relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Dropout(0.5))

  model.add(Conv2D(128, (3, 3), padding='valid', activation='relu'))
  model.add(Conv2D(128, (3, 3), activation='relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Dropout(0.5))

  model.add(Conv2D(256, (3, 3), padding='valid', activation='relu'))
  model.add(Conv2D(256, (3, 3), activation='relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Dropout(0.5))

  model.add(Flatten())
  model.add(Dense(512, activation='relu'))
  model.add(Dropout(0.5))
  model.add(Dense(4, activation='softmax'))

  model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

  return model


def createVGGModel():

  import tensorflow.keras as Keras
  from keras.models import Sequential
  from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten, Activation, Input

  vgg_model = Keras.applications.VGG16(include_top=False, weights="imagenet", input_shape=(256, 256, 3))

  for layer in vgg_model.layers[:-5]:
    layer.trainable = False
  
  model = Sequential()
  model.add(Conv2D(3,(3,3),padding='same',input_shape=(256,256,1)))
  model.add(vgg_model)
  model.add(Flatten())
  model.add(Dropout(0.5))
  model.add(Dense(4, activation='softmax'))


  model.compile(loss='categorical_crossentropy', optimizer='SGD', metrics=['accuracy'])

  return model


def createResnetModel():

  #import resnet and libraries
  import tensorflow.keras as Keras
  from keras.models import Sequential
  from keras.layers import Dense, Conv2D, Dropout, Flatten

  #Import the ResNet50 model and assign it to variable with desired configuration
  resnet_model = Keras.applications.ResNet50(include_top=False, weights="imagenet", input_shape=(256, 256, 3))

  #Freeze the first 143 layers of the model (first four blocks)
  for layer in resnet_model.layers[:143]:
      layer.trainable = False

  model = Sequential()
  model.add(Conv2D(3,(3,3),padding='same',input_shape=(256,256,1)))
  model.add(resnet_model)
  model.add(Flatten())
  model.add(Dropout(0.5))
  model.add(Dense(4, activation='softmax'))
  

  model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

  return model