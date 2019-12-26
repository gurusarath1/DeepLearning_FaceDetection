import PIL
from matplotlib import pyplot
from os import listdir
from matplotlib import image
import numpy as np
import random
import pickle
from skimage import data, io, color
from skimage.transform import rescale, resize

def image_center_crop(img):
    
    shapeImg = img.shape
    length = min(shapeImg[0], shapeImg[1])
    half = int(length / 2)
    
    if shapeImg[0] == length:
        center = int(shapeImg[1] / 2)
        cropped_img = img[:,center-half:center+half,:]
    else:
        center = int(shapeImg[0] / 2)
        cropped_img = img[center-half:center+half,:,:]
    
    return cropped_img


TrainSet_X = list()
TrainSet_Y = list()
WorkingImageSize = (100,100)

NumClasses = 2
Class = []
ClassX = [ np.array([1,0]) , np.array([0,1]) ]
for i in range(NumClasses):
	Class.append(i)

Directories = ['.\\Not_Faces\\', '.\\Faces\\']


AllFiles = []

for i,dirX in enumerate(Directories):
	files = listdir(Directories[i])
	random.shuffle(files)
	AllFiles.append(files)


def NotEmpty(AllFiles):

	for filesX in AllFiles:
		if filesX:
			return True

	return False


while NotEmpty(AllFiles):

	choiceX = random.choice(Class)
	while not AllFiles[choiceX]:
		choiceX = random.choice(Class)

	x_name = random.choice(AllFiles[choiceX])
	AllFiles[choiceX].remove(x_name)
	x = io.imread(Directories[choiceX] + x_name)
	x = image_center_crop(x)
	x = resize(x, WorkingImageSize, anti_aliasing=True)


	TrainSet_X.append(x)
	TrainSet_Y.append(ClassX[choiceX])


TrainSet_X = np.array(TrainSet_X)
TrainSet_Y = np.array(TrainSet_Y)

print(TrainSet_X.shape)
print(TrainSet_Y.shape)

pickle_file = open('TrainSet_X.pickle','wb')
pickle.dump(TrainSet_X, pickle_file)
pickle_file.close()
pickle_file = open('TrainSet_Y.pickle','wb')
pickle.dump(TrainSet_Y, pickle_file)
pickle_file.close()
