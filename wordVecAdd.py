# File for reading Quora data and converting questions
# to vectors using Facebook's fastText vectors for each
# word, and then adding the vectors.
#
# Use cosine distance to test if sentances are the same.
#
# Author: Peter Plantinga
# Date: April 2017

import csv
import numpy as np
import scipy.spatial.distance as dist
import string
#from nltk.stem import WordNetLemmatizer
from sklearn import svm
from sklearn import preprocessing

# Read the FaceBook fastText word vectors into a
# - dict of words -> indexes in the numpy array
# - numpy array of word vectors
def readVecs(filename):

    words = dict()
    wordVecs = []
    with open(filename) as f:
        dimensions = list(map(int, f.readline().split()))
        wordVecs = np.zeros(dimensions, dtype=float)
        for i in range(dimensions[0]):
            line = f.readline().split()
            if len(line) >= 300:
                words[''.join(line[0:-300])] = i
                wordVecs[i,:] = line[-300:len(line)]
            else:
                print(i)

    return words, wordVecs

# Read a csv data file and convert sentences to vectors
# output 4 numpy arrays:
# - train data
# - train labels
# - dev data
# - dev labels
def readFile(filename, words, wordVecs):
    data0 = []
    data1 = []
    with open(filename) as f:
        reader = csv.DictReader(f)
        for line in reader:
            q1Vec = np.zeros(300, dtype=float)
            q2Vec = np.zeros(300, dtype=float)
            translator = str.maketrans('', '', string.punctuation)
            q1 = line['question1'].lower().translate(translator)
            q2 = line['question2'].lower().translate(translator)
            for word in q1.split():
                if word in words:
                    q1Vec += wordVecs[words[word]]

            for word in q2.split():
                if word in words:
                    q2Vec += wordVecs[words[word]]

            if int(line['is_duplicate']) == 0:
                data0.append([q1Vec, q2Vec])
            else:
                data1.append([q1Vec, q2Vec])

    cutoff = int(len(data1)*0.75)

    trainX = np.concatenate((data0[0:cutoff], data1[0:cutoff]))
    trainY = np.concatenate((-np.ones(cutoff), np.ones(cutoff)))
    np.random.seed(100)
    np.random.shuffle(trainX)
    np.random.seed(100)
    np.random.shuffle(trainY)

    devX = np.concatenate((data0[cutoff:len(data1)], data1[cutoff:len(data1)]))
    devY = np.concatenate((-np.ones(len(data1)-cutoff), np.ones(len(data1)-cutoff)))
    np.random.seed(100)
    np.random.shuffle(devX)
    np.random.seed(100)
    np.random.shuffle(devY)

    return trainX, trainY, devX, devY


########
# Main #
########

words, wordVecs = readVecs('wiki.en.vec')
trainX, trainY, devX, devY = readFile('train.csv', words, wordVecs)

# implement SGD to train perceptron
eta = 0.0001
np.random.seed(12321)
w = np.random.random(601)
for epoch in range(10):
    for i in range(len(trainX)):
        x = np.concatenate((trainX[i][0], trainX[i][1], [1]))
        prediction = w.dot(x)
        if prediction * trainY[i] < 0:
            w += eta * trainY[i] * x


# Count the number of correct examples in the dev data
correctCount = 0
for i in range(len(devX)):
    if devY[i] * w.dot(np.concatenate((devX[i][0], devX[i][1], [1]))) >= 0:
        correctCount += 1

print("Perceptron accuracy: ", correctCount / len(devX))

X = np.concatenate((trainX[:,0], trainX[:,1]), axis=1)
scaler = preprocessing.StandardScaler().fit(X)
scaler.transform(X)

clf = svm.LinearSVC(C=0.01, dual=False)
clf.fit(X, trainY)

X = np.concatenate((devX[:,0], devX[:,1]), axis=1)
scaler.transform(X)
print("SVM accuracy: ", clf.score(X, devY))
