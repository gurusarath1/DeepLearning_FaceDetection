import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation, Dropout
from keras.layers.advanced_activations import LeakyReLU
import pickle
import numpy as np
from skimage import data, io, color
from skimage.transform import rescale, resize


WorkingImageSize = (100,100)

def image_center_crop_resize(img):
    
    shapeImg = img.shape
    length = min(shapeImg[0], shapeImg[1])
    half = int(length / 2)
    
    if shapeImg[0] == length:
        center = int(shapeImg[1] / 2)
        cropped_img = img[:,center-half:center+half,:]
    else:
        center = int(shapeImg[0] / 2)
        cropped_img = img[center-half:center+half,:,:]
    
    cropped_resized_img = resize(cropped_img, WorkingImageSize, anti_aliasing=True)

    return cropped_resized_img


def make_model():

    model = Sequential()

    model.add(Conv2D(16, (3,3), padding='same', input_shape=(WorkingImageSize[0],WorkingImageSize[1],3)))
    model.add(LeakyReLU(0.1))
    
    model.add(Conv2D(32, (3,3), padding='same'))
    model.add(LeakyReLU(0.1))
    
    model.add(MaxPooling2D(pool_size=(2,2)))
    
    model.add(Dropout(0.25))
    
    model.add(Conv2D(32, (3,3), padding='same'))
    model.add(LeakyReLU(0.1))
    
    model.add(Conv2D(64, (3,3), padding='same'))
    model.add(LeakyReLU(0.1))
    
    model.add(MaxPooling2D(pool_size=(2,2)))
    
    model.add(Dropout(0.25))
    
    model.add(Flatten())
    model.add(Dense(50))
    model.add(LeakyReLU(0.1))
    model.add(Dropout(0.5))
    model.add(Dense(2))
    model.add(Activation('softmax'))
    
    return model

fileX = open('TrainSet_X.pickle', 'rb')
X = pickle.load(fileX)
fileX = open('TrainSet_Y.pickle', 'rb')
Y = pickle.load(fileX)

model = make_model()
model.summary()

model.compile(
    loss='categorical_crossentropy',  # we train 10-way classification
    optimizer='rmsprop',  # for SGD
    metrics=['accuracy']  # report accuracy during training
)

model.fit(
    X, Y,  # prepared data
    batch_size=50,
    epochs=20,
)

y_pred = model.predict_classes( np.array([image_center_crop_resize(io.imread('sample.jpg'))])  ) 
print(y_pred)