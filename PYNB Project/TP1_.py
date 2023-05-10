# import nltk
# from nltk import data
# from nltk.tokenize import word_tokenize
# from nltk.stem import PorterStemmer, WordNetLemmatizer
# from nltk.corpus import stopwords
# import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# import json

# nltk.download('wordnet')
# nltk.download('punkt')
# nltk.download('stopwords')
# lemmatizer = WordNetLemmatizer()
# stemmer = PorterStemmer()

# file = pd.read_json('snli_1.0_test.jsonl', lines=True)
# fileClass = file['gold_label']
# fileLines = file['sentence2']
# #print(fileClass)

# linesFilt = []
# for i in range(len(fileLines)):
#     if (i % 1000 == 0):
#         print(i)
#     tok=word_tokenize(fileLines[i]) #Separa la oración en strings.
#     alpha=[x for x in tok if x.isalpha()] #Saca palabras con números.
#     lem=[lemmatizer.lemmatize(x,pos='v') for x in alpha] #Pasa de plural a singular y generaliza el género.
#     stop=[x for x in lem if x not in stopwords.words('english')] #Saca las palabras comunes.
#     stem=[stemmer.stem(x) for x in stop] #Verbos a infinitivo y pasa todo a minuscula.
#     linesFilt.append(" ".join(stem))

# with open('test_processed.jsonl', 'w') as file:
#     for i in range(len(fileClass)):
#         data2jsonl = {'gold_label': fileClass[i], 'sentence2': linesFilt[i]}
#         json.dump(data2jsonl, file)
#         file.write('\n')

######################################################################################################################################################################
# import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# import json

# file = pd.read_json('train_processed_.jsonl', lines=True)
# fileClass = file['gold_label']
# fileLines = file['sentence2']
# #print(fileClass)

# with open('train_sentence2.jsonl', 'w') as file:
#     for i in range(len(fileClass)):
#         data2jsonl = {'gold_label': fileClass[i]}
#         json.dump(data2jsonl, file)
#         file.write('\n')

# with open('train_sentence2.jsonl', 'w') as file:
#     for i in range(len(fileClass)):
#         data2jsonl = {'sentence2': fileLines[i]}
#         json.dump(data2jsonl, file)
#         file.write('\n')

######################################################################################################################################################################
import numpy as np
import pandas as pd #Implementación de clasificador bayesiano.
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import precision_score
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
# https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html
# https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html

trainFile = pd.read_json('train_processed_.jsonl', lines=True)
trainClass = trainFile['gold_label'].tolist()
trainLines = trainFile['sentence2'].tolist()
# Nlines = 300000
# for i in range(Nlines, len(trainClass)):
#   trainClass.pop(Nlines) #Erase lines to free space.
#   trainLines.pop(Nlines) #Erase lines to free space.
# print(len(trainClass))
valFile = pd.read_json('val_processed_.jsonl', lines=True)
valClass = valFile['gold_label'].tolist()
valLines = valFile['sentence2'].tolist()

score1 = 0
score2 = 0
for mindf in range(5,20):
   for alpha in np.arange(0.1,2,0.1):
        # countVect = CountVectorizer(max_df=0.8,min_df=10) #Recibe todos los artículos y arma los vectores de cuenta. Check "ngram_range"
        tfidfVect = TfidfVectorizer(max_df=0.8,min_df=mindf,ngram_range = (1,2)) #As tf–idf is very often used for text features

        # trainData = countVect.fit_transform(trainLines) #Learn a vocabulary dictionary of all tokens in the raw documents and return document-term matrix.
        # print(trainData.shape) #En este sparse matrix se muestran las ocurrencias de cada palabra en cada linea.
        trainData = tfidfVect.fit_transform(trainLines) #Learn vocabulary and idf from training set, return document-term matrix.
        #print(trainData.shape) #En este sparse matrix se muestran las ocurrencias de cada palabra en cada linea. Con tfidf las ocurrencias de las palabras no se representan con números enteros.

        # valData = countVect.transform(valLines) #Transform documents to document-term matrix. Extract token counts out of raw text documents using the vocabulary fitted with fit.
        # print(valData.shape)
        valData = tfidfVect.transform(valLines) #Transform documents to document-term matrix. Extract token counts out of raw text documents using the vocabulary fitted with fit.
        #print(valData.shape)

        multiNB = MultinomialNB(alpha=alpha)
        multiNB.fit(trainData, trainClass)
        score1_ = multiNB.score(trainData, trainClass)
        score2_ = multiNB.score(valData, valClass)

        if score2_ > score2:
            score1 = score1_
            score2 = score2_
            print('trainScore: {} | valScore: {} | min_df = {} | alpha = {}'.format(score1, score2, mindf, alpha))

valPred = multiNB.predict(valData)
#print(valPred)

enc = LabelEncoder()
enc.fit(valClass)
valClass = enc.transform(valClass)
valClass_ = np_utils.to_categorical(valClass-1) #Convert integers to dummy variables (i.e. one hot encoded).
enc.fit(valPred)
valPred = enc.transform(valPred)
valPred_ = np_utils.to_categorical(valPred) #Convert integers to dummy variables (i.e. one hot encoded).

print(precision_score(valClass_, valPred_, average=None))
print(precision_score(valClass_, valPred_, average='micro'))
print(precision_score(valClass_, valPred_, average='macro'))
print(precision_score(valClass_, valPred_, average='weighted'))

            
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam, SGD
#from keras.layers import Dropout
from keras.wrappers.scikit_learn import KerasClassifier
#from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
#from sklearn.preprocessing import LabelEncoder
#from sklearn.pipeline import Pipeline
from sklearn.decomposition import TruncatedSVD

trunSVD = TruncatedSVD(n_components=1000)
trainData_ = trunSVD.fit_transform(trainData)
valData_ = trunSVD.transform(valData)
#print(trainData_.shape)

enc = LabelEncoder()
enc.fit(trainClass)
trainClass = enc.transform(trainClass)
trainClass_ = np_utils.to_categorical(trainClass-1) #Convert integers to dummy variables (i.e. one hot encoded).
#print(trainClass_)
enc.fit(valClass)
valClass = enc.transform(valClass)
valClass_ = np_utils.to_categorical(valClass-1) #Convert integers to dummy variables (i.e. one hot encoded).

def neuralNetwork():
    Nwords = trainData_.shape[1] #cantidad de datos (palabras del vocabulario). ###
    model = Sequential()
    model.add(Dense(200, input_shape=(Nwords,), activation='relu'))
    model.add(Dense(trainClass_.shape[1], activation='softmax')) #La salida de la función softmax puede ser utilizada para representar una distribución categórica. 
                                                #Es empleada en varios métodos de clasificación multiclase tales como Regresión Logística Multinomial.
                                                #La función softmax es utilizada como capa final de los clasificadores basados en redes neuronales.
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

#Cross validation.
# estClass = KerasClassifier(build_fn=neuralNetwork, epochs=50, batch_size=256, verbose=0) #Estimator.
# Kfold = KFold(n_splits=10, shuffle=True)
# scores = cross_val_score(estClass, trainData_, trainClass_, cv=Kfold)
# print("Score: %.2f%% (%.2f%%)" % (scores.mean()*100, scores.std()*100))
#Hold out
model = neuralNetwork()
model.fit(trainData_, trainClass_, epochs=10, batch_size=256, verbose=1)
loss, acc = model.evaluate(valData_, valClass_)
print('Test Accuracy: %f' % (acc*100))

valPred = model.predict(valData_, verbose = 1).round()
#print(valPred.round())
print(precision_score(valClass_, valPred, average=None))
print(precision_score(valClass_, valPred, average='micro'))
print(precision_score(valClass_, valPred, average='macro'))
print(precision_score(valClass_, valPred, average='weighted'))

######################################################################################################################################################################
# from keras.models import Sequential
# from keras.layers import Dense
# from keras.optimizers import Adam, SGD
# #from keras.layers import Dropout
# from keras.wrappers.scikit_learn import KerasClassifier
# from keras.utils import np_utils
# from sklearn.model_selection import cross_val_score
# from sklearn.model_selection import KFold
# from sklearn.preprocessing import LabelEncoder
# #from sklearn.pipeline import Pipeline
# from sklearn.decomposition import TruncatedSVD
# from dataGen import DataGenerator

# params = {'dim': (32,32,32),
#           'batch_size': 64,
#           'n_classes': 6,
#           'n_channels': 1,
#           'shuffle': True}



# trunSVD = TruncatedSVD(n_components=100)
# trainData_ = trunSVD.fit_transform(trainData)
# valData_ = trunSVD.transform(valData)
# #print(trainData_.shape)

# enc = LabelEncoder()
# enc.fit(trainClass)
# trainClass = enc.transform(trainClass)
# trainClass_ = np_utils.to_categorical(trainClass-1) #Convert integers to dummy variables (i.e. one hot encoded).
# #print(trainClass_)
# valClass = enc.transform(valClass)
# valClass_ = np_utils.to_categorical(valClass-1) #Convert integers to dummy variables (i.e. one hot encoded).

# def neuralNetwork():
#     Nwords = trainData_.shape[1] #cantidad de datos (palabras del vocabulario). ###
#     model = Sequential()
#     model.add(Dense(200, input_shape=(Nwords,), activation='relu'))
#     model.add(Dense(trainClass_.shape[1], activation='softmax')) #La salida de la función softmax puede ser utilizada para representar una distribución categórica. 
#                                                 #Es empleada en varios métodos de clasificación multiclase tales como Regresión Logística Multinomial.
#                                                 #La función softmax es utilizada como capa final de los clasificadores basados en redes neuronales.
#     model.summary()
#     model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#     return model

# #Cross validation.
# # estClass = KerasClassifier(build_fn=neuralNetwork, epochs=50, batch_size=256, verbose=0) #Estimator.
# # Kfold = KFold(n_splits=10, shuffle=True)
# # scores = cross_val_score(estClass, trainData_, trainClass_, cv=Kfold)
# # print("Score: %.2f%% (%.2f%%)" % (scores.mean()*100, scores.std()*100))
# #Hold out
# model = neuralNetwork()
# model.fit(trainData_, trainClass_, epochs=10, batch_size=256, verbose=1)
# loss, acc = model.evaluate(valData_, valClass_)
# print('Test Accuracy: %f' % (acc*100))