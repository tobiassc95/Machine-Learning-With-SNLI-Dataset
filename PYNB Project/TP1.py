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

# file = pd.read_json('test_processed.jsonl', lines=True)
# fileClass = file['gold_label']
# fileLines = file['sentence2']
# #print(fileClass)

######################################################################################################################################################################
import numpy as np
import pandas as pd #Implementación de clasificador bayesiano.
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
# https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html
# https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html

trainFile = pd.read_json('train_processed_.jsonl', lines=True)
trainClass = trainFile['gold_label'].tolist()
trainLines = trainFile['sentence2'].tolist()
#print(trainClass)
# Nlines = 300000
# for i in range(Nlines, len(trainClass)):
#   trainClass.pop(Nlines) #Erase lines to free space.
#   trainLines.pop(Nlines) #Erase lines to free space.
# print(len(trainClass))

valFile = pd.read_json('val_processed_.jsonl', lines=True)
valClass = valFile['gold_label'].tolist()
valLines = valFile['sentence2'].tolist()
#print(fileClass)
#print(len(valClass))

countVect = CountVectorizer(max_df=0.145,min_df=10) #Recibe todos los artículos y arma los vectores de cuenta. Check "ngram_range"
tfidfVect = TfidfVectorizer(max_df=0.145,min_df=10)#,ngram_range=(1,5),norm='l2',use_idf=True, smooth_idf=True, sublinear_tf=False) #Probar con 'l1'

#countVect.fit(trainLines) #Aprende el vocabulario y le asigna un codigo a cada palabra (código = número de índice).
#print(countVect.vocabulary_) #Estos son los índices de cada una de las palabras (El _ significa que aprendió el vocabulario luego de hacer el fit).
##print(len(countVect.vocabulary_))
#print(countVect.get_feature_names()) #Devuelve el vocabulario aprendido luego del fit.
trainData = countVect.fit_transform(trainLines) #Aprende el vocabulario y le asigna un codigo a cada palabra (código = número de índice).
print(trainData.shape) #(550153, 7566). tiene 550153 filas (lineas) y 7566 columnas (palabras del vocabulario). En este sparse matrix se muestran las ocurrencias de cada palabra en cada linea.
#print(trainData)
##print(trainData.toarray())

trainData = tfidfVect.fit_transform(trainLines)
print(trainData.shape) #(550153, 7566). tiene 550153 filas (lineas) y 7566 columnas (palabras del vocabulario). En este sparse matrix se muestran las ocurrencias de cada palabra en cada linea.
#print(trainData) #con tfidf las ocurrencias de las palabras no se representan con números enteros.
#print(len(trainData[0]))

valData = tfidfVect.transform(valLines)
print(valData.shape)
#print(valData) #con tfidf las ocurrencias de las palabras no se representan con números enteros.
#print(len(valData[0]))

col = tfidfVect.get_feature_names()
df = pd.DataFrame(trainData.toarray(), columns=col)
df['gold_label'] = trainClass
#print(df.head())

# alpha = 1.0 #para el smoothing laplaciano (alpha = hiperparámetro).
# Plikelyh = list() #Probabilidad de cada palabra para cada categoría.
# Pprior = list() #Probabilidad de ocurrencia de cada categoría.
# Nlines = df.values.shape[0] #Cantidad de lineas.
# Nwords = df.values.shape[1] #Cantidad de palabras en el vocabulario.
# for i in range(3):
#     Ntimes = sum(df.loc[df['gold_label'] == i].drop('gold_label',axis=1).values, alpha) #Me da un vector con la suma de las ocurrencias de las palabras dado la clase i.
#     Plikelyh.append(np.log(Ntimes/sum(Ntimes)))
#     Pprior.append(np.log(df.loc[df['gold_label'] == i].shape[0]/Nlines))
# # for i in range(3): #Cantidad de clases.
# #     Ntimes = sum(1 for j in trainClass if j == i) + alpha ##
# #     print('La clase {} apareció {} veces. Por lo tanto Ppriori = {}'.format(i,Ntimes,Ntimes/Nlines))
# line = 1005 #Agarramos una línea y calculamos el likelyhood de cada una de las clases.
# maxlogL = -float('inf')
# for i in range(3): #3 clases.
#     logL = np.dot(trainData.toarray()[line],Plikelyh[i]) + Pprior[i] #producto a punto (dot product).
#     print('Clase {} con log-likelyhood {}'.format(i, logL))
#     if logL > maxlogL:
#         maxlogL = logL
#         maxI = i
# print('El log-likelyhood mayor de la línea {} pertenece a la clase {} con un valor de {}'.format(line, maxI, maxlogL))

clf = MultinomialNB(alpha=0.65)
clf.fit(trainData, trainClass)
#print('La predicción para la línea {} es la clase {}'.format(line, clf.predict(trainData.toarray()[line:line+1])))

# P = sum(np.array(clf.predict(trainData.toarray())) == np.array(trainClass))/len(trainClass)
# print('El porentaje de artículos clasificados correctamente (score) es: {}'.format(P))
print(clf.score(trainData, trainClass))
print(clf.score(valData, valClass))

#################################################################################
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam, SGD

hidden_units = 3 #trainData[1] #cantidad de datos (palabras del vocabulario).
input_dim = 2#trainData[1]
activ = 'sigmoid'
model = Sequential()
model.add(Dense(hidden_units, input_shape=(input_dim,), activation=activ))
#model.add(Dense(10, activation='tanh'))
model.add(Dense(1, activation=activ))
model.summary()

#model.compile(Adam(lr=0.01), loss='binary_crossentropy', metrics=['accuracy']) #Loss function and optimizer.
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) #Loss function and optimizer.
#model.fit(X, y, epochs=500, verbose=0) #entrenamos
#accuracy = model.evaluate(X, y)

# predictions = model.predict_classes(X) #make class predictions with the model
# for i in range(5): #summarize the first 5 cases
# 	print('%s => %d (expected %d)' % (X[i].tolist(), predictions[i], y[i]))


#model.evaluate(X, y) #load values (?).
##weights = model.get_weights()
##print(weights)

#model.get_weights()