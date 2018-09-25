import pandas as pd
import gensim.models as g
from collections import defaultdict
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Sequential,Model, Input
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional
from keras_contrib.layers import CRF
from keras_contrib.utils import save_load_utils
from keras.callbacks import EarlyStopping,ModelCheckpoint
from keras.optimizers import Adam
from keras.metrics import categorical_accuracy, mean_squared_error, mean_absolute_error, logcosh
import jellyfish as jf
import stringdist

#Sklearn - keras wrapper
#from sklearn.model_selection import GridSearchCV
#from keras.wrappers.scikit_learn import KerasClassifier
#from keras_pickle_wrapper import KerasPickleWrapper

import ast
import numpy as np
import time
#import pickle
path = "C:\\MyStuff\\ASU\\Spring_2018\\NLP\\Project\\Clinical_NLP\\Dataset\\Train_Test_Data\\"

numOfTrainRecords = 10
#numOfTestRecords = 1000


trainSetDF_FULL=pd.read_csv(path+'trainSetDF_FULL.csv',header = None)
trainSetDF_FULL.columns = ['0','FileName','Order']
trainSetDF_FULL = trainSetDF_FULL.drop(['0'], axis=1)
#
train_Dict=trainSetDF_FULL.head(numOfTrainRecords).set_index('FileName').T.to_dict('list')
del trainSetDF_FULL

testSetDF=pd.read_csv(path+'testSetDF.csv',header = None)
testSetDF.columns = ['0','FileName','Order']
testSetDF = testSetDF.drop(['0'], axis=1)
#
test_Dict=testSetDF.head(5).set_index('FileName').T.to_dict('list')
del testSetDF

#Loading Doc2Vec
model_doc2vec= g.Doc2Vec.load(path+"model_doc_2_vec.bin")

numDoc = 0
num_docs = 1500000 
max_len = 21
vector_dim = 321

sectionHeaders = ['history_present_illness','activity','discharge_condition','past_medical_history','chief_complaint','follow_up','discharge_instructions','allergies_and_adverse_reactions','admission_date','hospital_course','findings','review_of_systems','family_history','laboratory_and_radiology_data','diagnoses','physical_examination','assessment_and_plan','social_history','medications','history_source','assessment']

t1=time.time()

embedding_matrix = np.zeros((num_docs, vector_dim))
def sectionHeaderSimilarities(fileName_SectionHeader_Tag):
    global embedding_matrix
    global numDoc
    numDoc=numDoc+1
    SectionSim=[model_doc2vec.docvecs.similarity(sectionHeader,fileName_SectionHeader_Tag) for sectionHeader in sectionHeaders]
    embedding_matrix[numDoc]=list(model_doc2vec.docvecs[fileName_SectionHeader_Tag])+SectionSim
    return numDoc

def createListOfClinicalNotes_Doc2vec_And_SectionSeq(trainOrTestDict):
    counter = 0
    listOfClinicalNotes_Doc2vec = []
    listOfClinicalNotes_SectionSeq = []
    
    for fileName, seqOfSecHeader in trainOrTestDict.items():
        counter = counter+1
        seqOfSecHeader=ast.literal_eval(seqOfSecHeader[0])
        if len(seqOfSecHeader)>1:
            listOfClinicalNotes_Doc2vec.append([sectionHeaderSimilarities(fileName+'_'+sectionHeader) for sectionHeader in seqOfSecHeader])#np.array([list(model_doc2vec.docvecs[fileName+'_'+sectionHeader])+sectionHeaderSimilarities(fileName+'_'+sectionHeader) for sectionHeader in seqOfSecHeader])
            listOfClinicalNotes_SectionSeq.append(np.array([sectionHeaders.index(sectionHeader)+1 for sectionHeader in seqOfSecHeader]))
        if counter%1000 == 0:
            print (str(counter)+ ' files processed')
    return listOfClinicalNotes_Doc2vec,listOfClinicalNotes_SectionSeq
    
def getTrainTestXY(train_Dict, test_Dict):
    
    listOfClinicalNotes_Doc2vec_TRAIN,listOfClinicalNotes_SectionSeq_TRAIN = createListOfClinicalNotes_Doc2vec_And_SectionSeq(train_Dict)
    X_Train = pad_sequences(maxlen=max_len, dtype='int32',sequences=listOfClinicalNotes_Doc2vec_TRAIN, padding="post", value=0)
    y1_Train = pad_sequences(maxlen=max_len, sequences=listOfClinicalNotes_SectionSeq_TRAIN, padding="post", value=0)
    y_Train = [to_categorical(i, num_classes=max_len+1) for i in y1_Train]#21 section headers + 1 token for no input
    
    listOfClinicalNotes_Doc2vec_TEST,listOfClinicalNotes_SectionSeq_TEST = createListOfClinicalNotes_Doc2vec_And_SectionSeq(test_Dict)
    X_Test = pad_sequences(maxlen=max_len, dtype='int32',sequences=listOfClinicalNotes_Doc2vec_TEST, padding="post", value=0)
    y1_Test = pad_sequences(maxlen=max_len, sequences=listOfClinicalNotes_SectionSeq_TEST, padding="post", value=0)
    y_Test = [to_categorical(i, num_classes=max_len+1) for i in y1_Test]#21 section headers + 1 token for no input
    
    return X_Train,y_Train,X_Test,y_Test
    

def getF1score(inp):
    eps = 0.000000001
    return (2*(float(inp[0]*inp[1])))/(float(inp[0]+inp[1]+eps))

def mapHeadersListToString(headersToCharMap, listToMap):
    return "".join(list(map(lambda x: headersToCharMap.get(x), listToMap)))

def getSequenceDistances(trueSequence, predSequence, headersToCharMap):
    trueMapped = mapHeadersListToString(headersToCharMap, trueSequence)
    predMapped = mapHeadersListToString(headersToCharMap, predSequence)
    editDistance = stringdist.levenshtein(trueMapped, predMapped)
#    dlDistance = jf.damerau_levenshtein_distance(trueMapped,predMapped)
    return editDistance, 0
    
def evaluateModel(model,X_Test,y_Test):
    counter = 0
    numCorrectSeqPred=0
    numCorrectPredPerSeq = [0]*len(sectionHeaders)
    numAppearancesPerSeq = [0]*len(sectionHeaders)
    numTotalPredPerSeq = [0]*len(sectionHeaders)
    editDistance = 0
    editDistanceWithSwap = 0
    
    numToChar = {}
    for x in range(26):
        numToChar[x] = str(chr(x+97))
    
    headersToCharMap = {value:numToChar.get(index) for index,value in enumerate(sectionHeaders)}
    
    for i in range(len(X_Test)):
        counter = counter +1
        p = model.predict(np.array([X_Test[i]]))
        p = np.argmax(p, axis=-1)
        true = np.argmax(y_Test[i], -1)
        predictedSeq=[sectionHeaders[pred-1] for pred in p[0] if pred!=0]
        trueSeq=[sectionHeaders[t-1] for t in true if t!=0]
        
        editDistances = getSequenceDistances(trueSeq, predictedSeq, headersToCharMap)
        editDistance += editDistances[0]
        editDistanceWithSwap += editDistances[1]
        
        pairwise = zip (trueSeq, predictedSeq)
        matched_sections = [pair[0] for idx, pair in enumerate(pairwise) if pair[0] == pair[1]]
        unMatched_sections = [pair[1] for idx, pair in enumerate(pairwise) if pair[0] != pair[1]]
        for corrSec in matched_sections:
            numCorrectPredPerSeq[sectionHeaders.index(corrSec)]=numCorrectPredPerSeq[sectionHeaders.index(corrSec)]+1
            numTotalPredPerSeq[sectionHeaders.index(corrSec)]=numTotalPredPerSeq[sectionHeaders.index(corrSec)]+1
        for inCorrSec in unMatched_sections:
           numTotalPredPerSeq[sectionHeaders.index(inCorrSec)]=numTotalPredPerSeq[sectionHeaders.index(inCorrSec)]+1
        for section in trueSeq:
            numAppearancesPerSeq[sectionHeaders.index(section)]=numAppearancesPerSeq[sectionHeaders.index(section)]+1
        if str(predictedSeq) == str(trueSeq):
            numCorrectSeqPred=numCorrectSeqPred+1
        if counter%10 == 0:
            print (str(counter)+ ' files processed')
    
    editDistance = editDistance/float(len(X_Test))
    editDistanceWithSwap = editDistanceWithSwap/float(len(X_Test))
    
    eps = 0.000000001
    perSections_Prec = [float(pair[1])/float(pair[0]+eps) for idx, pair in enumerate(zip(numTotalPredPerSeq, numCorrectPredPerSeq))] 
    perSections_Rec = [float(pair[1])/float(pair[0]+eps) for idx, pair in enumerate(zip(numAppearancesPerSeq, numCorrectPredPerSeq))]
    perSections_F1Score = list(map(getF1score,zip(perSections_Prec,perSections_Rec)))
    return pd.DataFrame(list(zip(sectionHeaders,perSections_Prec,perSections_Rec,numAppearancesPerSeq,numCorrectPredPerSeq,perSections_F1Score)),columns = ['sectionHeaders','perSections_Prec','perSections_Rec','numAppearancesPerSeq','numCorrectPredPerSeq','F1-Score-PerSection']),float(numCorrectSeqPred)/float(len(X_Test)), editDistance, editDistanceWithSwap
def printTrueSeqVsPredictionSeq(model,X_Test,y_Test):
    for i in range(len(X_Test)):
        p = model.predict(np.array([X_Test[i]]))
        p = np.argmax(p, axis=-1)
        true = np.argmax(y_Test[i], -1)
        
        
        print("{:50}{}".format("True", "Pred"))
        print(80 * "=")
        for t, pred in zip(true, p[0]):
            if t!=0:
                print("{:50} {}".format(sectionHeaders[t-1], sectionHeaders[pred-1]))
        print(80 * "=")
        
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
               

num_LSTM_Units = 500
dropout = 0.5
learning_rate = 0.0001 


X_Train,y_Train,X_Test,y_Test=getTrainTestXY(train_Dict, test_Dict)

del train_Dict
del test_Dict

doc_input = Input(shape=(max_len,),dtype='float32', name='doc_input')

print("Creating Embedding Layer...")
#embedding layer intialized with the matrix created earlier
embedded_doc_input = Embedding(output_dim=vector_dim, input_dim=num_docs,weights=[embedding_matrix], trainable=False,mask_zero=True)(doc_input)
#Dumping the Embedding Matrix
embedding_matrix.dump(path+'EmbeddingDump.dat')

print("Embedding Layer Created, Creating BiDirectional Layer...")
model=Bidirectional(LSTM(units=num_LSTM_Units, return_sequences=True,recurrent_dropout=dropout))(embedded_doc_input) # variational biLSTM

print("BiDirectional Layer Created, Creating TimeDistributed Layer...")
model=(TimeDistributed(Dense(vector_dim, activation="relu")))(model)  # a dense layer as suggested by neuralNer

print("TimeDistributed Layer Created, Creating CRF Layer...")
crf = CRF(max_len+1)  # CRF layer
out = crf(model)
print("CRF Layer Created, Training...")
model = Model(doc_input, out)

#optimizer = Adam(lr=learning_rate)
#callbacks=[EarlyStopping(patience=2, monitor='val_loss')]
model.compile(loss=crf.loss_function, optimizer="rmsprop", metrics=[crf.accuracy])
model.summary()

history = model.fit(X_Train, np.array(y_Train), batch_size=5000, epochs=10,
                    validation_split=0.15, verbose=1)

save_load_utils.save_all_weights(model,path+'test_model.h5')


embedding_matrix1 = np.load(path+'EmbeddingDump.dat')
embedded_doc_input1 = Embedding(output_dim=vector_dim, input_dim=num_docs,weights=[embedding_matrix1], trainable=False,mask_zero=True)(doc_input)

print("Embedding Layer Created, Creating BiDirectional Layer...")
model1=Bidirectional(LSTM(units=num_LSTM_Units, return_sequences=True,recurrent_dropout=dropout))(embedded_doc_input1) # variational biLSTM

print("BiDirectional Layer Created, Creating TimeDistributed Layer...")
model1=(TimeDistributed(Dense(vector_dim, activation="relu")))(model1)  # a dense layer as suggested by neuralNer

print("TimeDistributed Layer Created, Creating CRF Layer...")
crf1 = CRF(max_len+1)  # CRF layer
out1 = crf1(model1)
print("CRF Layer Created, Training...")
model1 = Model(doc_input, out1)

#optimizer = Adam(lr=learning_rate)
#callbacks=[EarlyStopping(patience=2, monitor='val_loss')]
model1.compile(loss=crf.loss_function, optimizer="rmsprop", metrics=[crf.accuracy])



save_load_utils.load_all_weights(model1, path+'test_model.h5',include_optimizer=False)
model1.summary()

summaryResult,percCorrectSeqPred,editDistance, editDistanceWithSwap = evaluateModel(model1,X_Test,y_Test)



#with open('Model_Performace.txt')

print("Percentage of Correctly Predicted Sequences :: ", percCorrectSeqPred)
print("Average Edit Distance :: ", editDistance)
print("Average Edit Distance with swaps allowed :: ", editDistanceWithSwap)
print(summaryResult)



#printTrueSeqVsPredictionSeq(model,X_Test,y_Test)


#hist = pd.DataFrame(history.history)
#
#import matplotlib.pyplot as plt
#plt.style.use("ggplot")
#plt.figure(figsize=(12,12))
#plt.plot(hist["viterbi_acc"])
#plt.plot(hist["val_viterbi_acc"])
#plt.show()