# -*- coding: utf-8 -*-
"""
Created on Sun Apr 15 14:26:36 2018

@author: ankit
"""

import pandas as pd
from gensim.models.doc2vec import TaggedDocument, Doc2Vec 
import gensim.models as g
import ast

path = "C:\\Geet\\college\\Dataset\\"


data_full = pd.read_csv(path+'allClinicalNotes_header_text.csv')
data_full = data_full.drop(['Unnamed: 0'], axis=1)

data1 = pd.read_csv(path + 'trainSetDF_FULL.csv', header = None)
data1 = data1[1].tolist()
data = data_full[data_full['FileName'].isin(data1)]

dataDict=data.to_dict('records')
del data

print('Read All data into List of Dict')
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

pretrained_emb = path + "mimic-i2b2-model.bin"

model = g.Doc2Vec(taggedDocuments, vector_size=300, epochs= 20, sample = 10e-5, window=15, negative = 5, min_count=5, workers=24, alpha=0.01, min_alpha=0.001, dm = 0, pretrained_emb=pretrained_emb)
    
model.save("model_dbow0_pretrained_trained_full.bin")