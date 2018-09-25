# -*- coding: utf-8 -*-
"""
Created on Sun Apr 15 13:44:15 2018

@author: ankit
"""

import pandas as pd
from gensim.models.doc2vec import TaggedDocument, Doc2Vec 
#import gensim.models as g
import ast
#import _pickle as pickle
import time

t1=time.time()
path = ""
data = pd.read_csv(path+'allClinicalNotes_header_text.csv')
data = data.drop(['Unnamed: 0'], axis=1)

dataDict=data.to_dict('records')
del data

print ('Read All data into List of Dict')
taggedDocuments = []
counter = 0
def createTaggedDocument(oneDict):
    global taggedDocuments
    global counter
    counter = counter + 1
    fileName = oneDict.get('FileName')
    sequence = oneDict.get('Order of Section Header Appearence')
    for sectionHeader in ast.literal_eval(sequence):
        if sectionHeader != 'Unknown':
             [taggedDocuments.append(TaggedDocument(oneDict.get(sectionHeader).split(), [fileName +'_'+ sectionHeader]) ) ]
        else:
            unknText = oneDict.get(sectionHeader).split('**Unknown**')
            [taggedDocuments.append(TaggedDocument(unknText(i).split(), [fileName +'_'+ sectionHeader+str(i+1)])) for i in range(unknText) ]
    if counter%10000 == 0:
        print(str(counter)+ ' files processed')
    return
temp=[createTaggedDocument(oneDict) for oneDict in dataDict]

print('Created Tagged Documents of length '+str(len(taggedDocuments)))

model = Doc2Vec(size=300, sample = 10e-5, window=15, negative = 5, min_count=5, workers=3, alpha=0.01, min_alpha=0.001, dm = 0,dbow_words=0)

model.build_vocab(taggedDocuments,progress_per=5)

print('Training Doc2Vec....')
model.train(taggedDocuments,total_examples=1797394, epochs=20)
#    
model.save(path+"model_dbow0_nopretrained.bin")

print('Doc2Vec Training finished and model is saved')
t2=time.time()
print(t2-t1)
#
#model= Doc2Vec.load("model1.bin")