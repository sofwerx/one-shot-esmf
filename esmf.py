"""
	Author: HJ van Veen <info@mlwave.com>
	Description: Experiment with zero and one-shot learning based on the paper:
				 (2015) "An embarrassingly simply approach to zero-shot learning"
				 Bernardino Romera-Paredes, Philip H. S. Torr
				 http://jmlr.org/proceedings/papers/v37/romera-paredes15.pdf
"""

from sklearn import datasets, linear_model, model_selection, preprocessing, decomposition, manifold
from sklearn.metrics import make_scorer, accuracy_score
import numpy as np
import warnings
warnings.simplefilter('ignore', DeprecationWarning)
from PIL import Image
import cv2
import os
import numpy as np
import glob

################################ Notes #############################################
# Be carefile where you have the label encoders since they can be over written without
# you knowing



#####################################################################################
############################### Get Data ############################################
#####################################################################################

wordDirectory = '/home/david/Documents/extremely-simple-one-shot-learning/'
testPath = '/home/david/Documents/extremely-simple-one-shot-learning/get_images/dataset/test/'
trainPath = '/home/david/Documents/extremely-simple-one-shot-learning/get_images/dataset/train/'
os.chdir(wordDirectory)


imageSize = 8
imagePixelCount = imageSize * imageSize


############################# Train Data ###########################################

str = " Train Data Load "
print(str.center(80,'#'))


path, dirs, files = next(os.walk(trainPath))
trainFileCount = len(files)
print(trainFileCount)
arrayOfTrainArrays = np.ndarray(shape = (trainFileCount,), dtype = "object")
countTrain = 0

arrayOfTrainArrays = np.ndarray(shape=(trainFileCount, imagePixelCount), dtype=np.float64)




for filename in glob.glob(trainPath  + "*.jpg"):


	img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
	img = cv2.resize(img,(imageSize, imageSize))
	img = img.astype(np.float64)
	img = img.flatten()

	arrayOfTrainArrays[countTrain] = img
	countTrain = countTrain + 1

print arrayOfTrainArrays
arrayOfTrainOnes = np.ones(trainFileCount / 2 )
arrayOfTrainZeros = np.zeros(trainFileCount / 2 )
arrayOfTrainLabels = np.concatenate([arrayOfTrainOnes,arrayOfTrainZeros])
print arrayOfTrainArrays
print arrayOfTrainOnes[1].dtype



# ############################## Test Data ###########################################
str = " Test Data Load "
print(str.center(80,'#'))


path, dirs, files = next(os.walk(testPath))
testFileCount = len(files)
print(testFileCount)
arrayOfTestArrays = np.ndarray(shape=(testFileCount, imagePixelCount), dtype=np.float64)
countTest = 0
for filename in glob.glob(testPath + "*.jpg"):

	img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
	img = cv2.resize(img,(imageSize, imageSize))
	img = img.flatten()
	img = img.astype(np.float64)

	arrayOfTestArrays[countTest] = img
	countTest = countTest + 1



arrayOfTestOnes = np.ones(testFileCount / 2 )
arrayOfTestZeros = np.zeros(testFileCount / 2 )
arrayOfTestLabels = np.concatenate([arrayOfTestOnes,arrayOfTestZeros])
print arrayOfTestLabels
print arrayOfTestLabels[1].dtype
#
#
#
#
############################# Data Transformations #############################

str = " Data Transformations "
print(str.center(80,'#'))

X, y = datasets.load_digits().data, datasets.load_digits().target
# print(X.dtype, X.shape)
# print(X[0])
# print(y)
#print X[0:2]
# print type(X)


#******** Note: Need to have 3 or more classes for training *********


arrayOfTrainLabels = y[np.where((y == 0) | (y == 1) | (y == 2) )]
arrayOfTrainArrays = X[np.where((y == 0) | (y == 1) | (y == 2))]

# We have d dimensions, d=64
# We have z classes, z=6, [digit0, digit1, digit2, digit7, digit8, digit9]
lbl = preprocessing.LabelEncoder()
y_train1 = lbl.fit_transform(y[np.where((y == 0) | (y == 1) | (y == 2) | (y == 7) | (y == 8) | (y == 9))])
y_train = lbl.fit_transform(arrayOfTrainLabels)

print(y_train.dtype,y_train.shape)
print(y_train1.dtype,y_train1.shape)
print(y_train)

X_train1 = X[np.where((y == 0) | (y == 1) | (y == 2) | (y == 7) | (y == 8) | (y == 9))]
X_train = arrayOfTrainArrays
#print(arrayOfTestArrays[0][0:64])
print(X_train.dtype,X_train.shape)
print X_train[0].shape
print(X_train1.dtype,X_train1.shape)
print X_train1[0].shape

# print(X_train1.dtype,X_train1.shape)
# print[X_train1[0]]
# print(X_train)

################################ Model Stage 1 #################################

str = " Train Model "
print(str.center(80,'#'))

# We have Weight matrix, W, d x z
model = linear_model.LogisticRegression(random_state=1)
model.fit(X_train, y_train)
W = model.coef_.T

print model_selection.cross_val_score(model, X_train, y_train, scoring=make_scorer(accuracy_score))


############################# Identify Signatures ##############################

str = " Identify Signatures "
print(str.center(80,'#'))
# We have a attributes, a=4 [pca_d1, pca_d2, lle_d1, lle_d2]
# We have Signature matrix, S a x z
pca = decomposition.PCA(n_components=2)
lle = manifold.LocallyLinearEmbedding(n_components=2, random_state=1)
X_pca = pca.fit_transform(X_train)
X_lle = lle.fit_transform(X_train)

#print X_pca, X_lle

for i, ys in enumerate(np.unique(y_train)):
	if i == 0:
		S = np.r_[ np.mean(X_pca[y_train == ys], axis=0), np.mean(X_lle[y_train == ys], axis=0) ]
	else:
		S = np.c_[S, np.r_[ np.mean(X_pca[y_train == ys], axis=0), np.mean(X_lle[y_train == ys], axis=0) ]]

print(S.shape)
print(W.shape)
# From W and S calculate V, d x a
V = np.linalg.lstsq(S.T, W.T , rcond=-1)[0].T


W_new = np.dot(S.T, V.T).T

print "%f"%np.sum(np.sqrt((W_new-W)**2))

#for ys, x in zip(y_train, X_train):
#	print np.argmax(np.dot(x.T, W_new)), ys


################################## Test Model ##################################

str = " Test Model "
print(str.center(80,'#'))

# arrayOfTestLabels = lbl.fit_transform(y[np.where((y == 3) | (y == 4)  )])
# arrayOfTestArrays  = X[np.where((y == 3) | (y == 4)  )]


# INFERENCE
# Be car
lbl = preprocessing.LabelEncoder()
print(arrayOfTestLabels)
y_test = lbl.fit_transform(arrayOfTestLabels)
print(y_test)
X_test = arrayOfTestArrays

# y_test1 = lbl.fit_transform(y[np.where((y == 3) | (y == 4)  )])
# X_test1 = X[np.where((y == 3) | (y == 4)  )]



print(y_test.dtype,y_test.shape)
# print(y_test1.dtype,y_test1.shape)
print(y_test)


print(X_test.dtype,X_test.shape)
print X_test[0].shape
# print(X_test1.dtype,X_test1.shape)
# print X_test1[0].shape

# y_test = lbl.fit_transform(y[np.where((y == 3) | (y == 4)  )])
# X_test = X[np.where((y == 3) | (y == 4)  )]

# create S' (the Signature matrix for the new classes, using the old transformers)
X_test, X_sig, y_test, y_sig = model_selection.train_test_split(X_test, y_test, test_size=4, random_state=1, stratify=y_test)

X_pca = pca.transform(X_sig)
X_lle = lle.transform(X_sig)

for i, ys in enumerate(np.unique(y_sig)):
	if i == 0:
		S = np.r_[ np.mean(X_pca[y_sig == ys], axis=0), np.mean(X_lle[y_sig == ys], axis=0) ]
	else:
		S = np.c_[S, np.r_[ np.mean(X_pca[y_sig == ys], axis=0), np.mean(X_lle[y_sig == ys], axis=0) ]]

# Calculate the new Weight/Coefficient matrix
# V comes from the trained model from the training data
W_new = np.dot(S.T, V.T).T

#Check performance
correct = 0
for i, (ys, x) in enumerate(zip(y_test, X_test)):
	#print i,ys
	print lbl.inverse_transform(np.argmax(np.dot(x.T, W_new))), lbl.inverse_transform(ys)
	if np.argmax(np.dot(x.T, W_new)) == ys:
		correct +=1

print correct, i, correct / float(i)