import os
import io
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

save_path = "model/mymodel"
checkpoint_path = "model/checkpoint.ckpt"
checkpont_dir = os.path.dirname(checkpoint_path)
max_length = 500
embedding_dim = 48
vocab_size = 30000
trainSize = 350000
testSize = 80000
num_epochs = 4 

def readData(file, reviews, ratings, rmax):
    review = ""
    prevLine = ""
    n = 0
    for line in file:
        if n < rmax:
            if  prevLine == "<review_text>\n":
                while line != "</review_text>\n":
                    line = line[:-1]
                    review = review + line + " "
                    line = file.readline()
                reviews.append(review)
                n += 1
                review = ""
            if prevLine == "<rating>\n":
                ratings.append(float(line))
            prevLine = line
    return n

def ratingsToBinary(ratings):
	bRatings = []
	for rating in ratings:
		bRatings.append(rating / 5.0)
	return bRatings
    
def ratingsToList(ratings):
    lst = []
    for rating in ratings:
        rList = []
        for i in range(1, int(rating)):
            rList.append(0)
        rList.append(1)
        for i in range(5 - int(rating)):
            rList.append(0)
        lst.append(np.array(rList))
    return lst
        
def takeData(data, labels, size):
    path = "./sorted_data/books/all.review"
    f=open(path, "r", encoding="ISO-8859-1")
    print("Loading data from:  " + path)
    n = readData(f, data, labels, size)
    print("Added ", str(n))
    
print("1 - Load and Evaluate")
print("2 - Train and Evaluate")
val = input();
mode = int(val)

reviews= []
ratings = []

trainReviews = []
trainRatings = []

testReviews = []
testRatings = []

size = 0
midpoint = 0

if mode == 1:
	trainSize = 0
	size = testSize
	midpoint = 0
elif mode == 2:
	size = trainSize + testSize
	midpoint = trainSize
elif (mode == 3):
    size = 0
    midpoint = 0

print("Reading data from hard disk...")
takeData(reviews, ratings, size);
trainReviews = reviews[:midpoint]
trainRatings = ratings[:midpoint]
testReviews = reviews[midpoint:]
testRatings = ratings[midpoint:]

print("Size of training data:  " + str(len(trainRatings)))
print("Size of testing data:   " + str(len(testRatings)))
print("Preprocessing data...")

trainLabels = np.array(ratingsToList(trainRatings))
testLabels = np.array(ratingsToList(testRatings))

tokenizer = Tokenizer(num_words = vocab_size, oov_token="<OOV>")
tokenizer.fit_on_texts(trainReviews)
word_index = tokenizer.word_index

trainSeq = tokenizer.texts_to_sequences(trainReviews)
testSeq = tokenizer.texts_to_sequences(testReviews)
del trainReviews
del testReviews

trainSeqPadded = pad_sequences(trainSeq, padding='post', truncating='post', maxlen=max_length)
testSeqPadded = pad_sequences(testSeq, padding='post', truncating='post', maxlen=max_length)
del trainSeq
del testSeq

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only = False, verbose = 1)

model = tf.keras.Sequential()
model.add(tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length = max_length))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(86, activation='relu'))
model.add(tf.keras.layers.Dense(5, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

if (mode == 3):
    model.summary()
elif (mode == 2):
    model.fit(
    trainSeqPadded,
    trainLabels,
    epochs=num_epochs,
   	callbacks=[cp_callback]
    )
    del trainSeqPadded
    del trainLabels
elif (mode == 1):
    print("Loading model...")
    model.load_weights(checkpoint_path)

print("Evaluating model...")
test_loss, test_acc = model.evaluate(testSeqPadded, testLabels)
del testSeqPadded
del testLabels

print("Loss= " + str(test_loss))
print("Accuracy= " + str(test_acc))


inReview = [""]
while (inReview[0] != "exit"):
	print("Insert a review:");
	inReview = [input()]
	inSeq = tokenizer.texts_to_sequences(inReview)
	inSeqPadded = pad_sequences(inSeq, padding='post', truncating='post', maxlen=max_length)
	prediction = model.predict(inSeqPadded) 

	print("Prediction= ", prediction[0])
	sol = 0
	j = 1
	for pred in prediction[0]:
	    sol += j * pred
	    j+=1
	print("That is: " + str(sol))
