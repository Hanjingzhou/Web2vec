# Load Libraries - Make sure to run this cell!

from keras.engine import Layer
from sklearn import model_selection

#import gensim
import tensorflow as tf
from keras.models import Sequential, Model, model_from_json, load_model
from keras import regularizers
from keras.layers.core import Dense, Dropout, Activation, Lambda, Flatten
from keras.layers import Input, ELU, LSTM, Embedding, Convolution2D, MaxPooling2D, \
    BatchNormalization, Convolution1D, MaxPooling1D, concatenate, ReLU, PReLU, initializers, constraints,Bidirectional
from keras.preprocessing import sequence
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils
from keras import backend as K

from HTML_word1 import load_data
from dom import load_data_dom
from URL_word import load_data_url
from URL_char import load_data_url_c
from HTML_senctence import load_data_sentences
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import auc,roc_auc_score,roc_curve
import matplotlib.pyplot as plt
from Attention_layer import Attention_layer
from pathlib import Path
import json

import warnings
warnings.filterwarnings("ignore")

epochs = 10             #epochs
batch_size = 64         #batch
lstm_output_size=128    #LSTM Unit
Embedding_dim=128       #dimension
lr=1e-4                 #Learning Rate
kernel_size=5           #CNN kernel_size
filters=256             #CNN filters
pool_size=4             #CNN pool_size




#input url，char
c_x_url, c_y_url,c_vocabulary_url, c_vocabulary_inv_url = load_data_url_c()

print("c_x_url.shape",c_x_url.shape)
print("c_y_url.shape",c_y_url.shape)

print("c_x_url",c_x_url)

X_train1, X_test1, target_train1, target_test1  =model_selection.train_test_split( c_x_url, c_y_url, test_size=0.25, random_state=33)#必须固定随机取数据方式

c_sequence_length_url = c_x_url.shape[1]
c_vocabulary_size_url = len(c_vocabulary_inv_url) 
print("target1",target_train1)

#input DOM
x, y, vocabulary, vocabulary_inv = load_data_dom()
print("x.shape",x.shape)
print("y.shape",y.shape)

print("x",x)

X_train2, X_test2, target_train2, target_test2  =model_selection.train_test_split( x, y, test_size=0.25, random_state=33)

sequence_length = x.shape[1] 
vocabulary_size = len(vocabulary_inv) 
print("target2",target_train2)

#input context，word
x1, y1,vocabulary1, vocabulary_inv1 = load_data()

print("x1.shape",x1.shape)
print("y1.shape",y1.shape)

print("x1",x1)

X_train3, X_test3, target_train3, target_test3  =model_selection.train_test_split( x1, y1, test_size=0.25, random_state=33)

sequence_length1 = x1.shape[1] 
vocabulary_size1 = len(vocabulary_inv1) 
print("target3",target_train3)


#input url，word
x_url, y_url,vocabulary_url, vocabulary_inv_url = load_data_url()

print("x_url.shape",x_url.shape)
print("y_url.shape",y_url.shape)

print("x_url",x_url)

X_train4, X_test4, target_train4, target_test4  =model_selection.train_test_split( x_url, y_url, test_size=0.25, random_state=33)

sequence_length_url = x_url.shape[1] 
vocabulary_size_url = len(vocabulary_inv_url) 
print("target4",target_train4)



#input context，sentences
x_sent, y_sent,vocabulary_sent, vocabulary_inv_sent = load_data_sentences()

print("x_sent.shape",x_sent.shape)
print("y_sent.shape",y_sent.shape)

print("x_sent",x_sent)

X_train5, X_test5, target_train5, target_test5  =model_selection.train_test_split( x_sent, y_sent, test_size=0.25, random_state=33)

sequence_length_sent = x_sent.shape[1] 
vocabulary_size_sent = len(vocabulary_inv_sent) 
print("target5",target_train5)




def lstm_conv(emb_dim=Embedding_dim,lstm_output_size=lstm_output_size, W_reg=regularizers.l2(1e-4)):
    # Input url_char
    input_url_char = Input(shape=(c_sequence_length_url,), dtype='int32', name='url_char_input')
    # Embedding layer
    emb_url_char = Embedding(input_dim=c_vocabulary_size_url, output_dim=emb_dim, input_length=c_sequence_length_url,W_regularizer=W_reg)(input_url_char)
    #input url_word
    input_url_word = Input(shape=(sequence_length_url,), dtype='int32', name='url_word_input')
    # Embedding layer
    emb_url_word = Embedding(input_dim=vocabulary_size_url, output_dim=emb_dim, input_length=sequence_length_url,W_regularizer=W_reg)(input_url_word)

    #url_char_model
    emb_url_char = Dropout(0.5)(emb_url_char)
    conv1 = Convolution1D(kernel_size=kernel_size, filters=filters, border_mode='same')(emb_url_char)
    conv1 = ELU()(conv1)
    conv1 = MaxPooling1D(pool_size=pool_size)(conv1)
    conv1 = Dropout(0.5)(conv1)
    lstm1 =Bidirectional(LSTM(lstm_output_size,return_sequences=True))(conv1)

    lstm1= Dropout(0.5)(lstm1)
    lstm1 = Attention_layer()(lstm1)

    # url_word_model

    emb_url_word = Dropout(0.5)(emb_url_word)
    conv2 = Convolution1D(kernel_size=kernel_size, filters=filters, border_mode='same')(emb_url_word)
    conv2 = ELU()(conv2)
    conv2 = MaxPooling1D(pool_size=pool_size)(conv2)
    conv2 = Dropout(0.5)(conv2)
    lstm2 = Bidirectional(LSTM(lstm_output_size, return_sequences=True))(conv2)

    lstm2 = Dropout(0.5)(lstm2)
    lstm2 = Attention_layer()(lstm2)

    #concatenate
    x_url_output = concatenate([lstm1, lstm2], axis=1)
    #x_url_output = Dense(128, activation='relu')(x_url_output)



    #DOM model
    input_dom = Input(shape=(sequence_length,), dtype='int32', name='dom_input')
    # Embedding layer
    emb_dom = Embedding(input_dim=vocabulary_size, output_dim=emb_dim, input_length=sequence_length,W_regularizer=W_reg)(input_dom)
    emb_dom = Dropout(0.5)(emb_dom)
    # Conv layer
    conv3 = Convolution1D(kernel_size=kernel_size, filters=filters, border_mode='same')(emb_dom)
    conv3 = ELU()(conv3)
    conv3 = MaxPooling1D(pool_size=pool_size)(conv3)
    conv3 = Dropout(0.5)(conv3)
    # LSTM layer
    lstm3 =Bidirectional(LSTM(lstm_output_size,return_sequences=True))(conv3)

    lstm3 = Dropout(0.5)(lstm3)
    lstm3 = Attention_layer()(lstm3)



    #text__word
    input_text_word = Input(shape=(sequence_length1,), dtype='int32', name='text_word_input')
    # Embedding layer
    emb_text_word = Embedding(input_dim=vocabulary_size1, output_dim=emb_dim, input_length=sequence_length1,W_regularizer=W_reg)(input_text_word)
    emb_text_word = Dropout(0.5)(emb_text_word)

    conv4 = Convolution1D(kernel_size=kernel_size, filters=filters, border_mode='same')(emb_text_word)
    conv4 = ELU()(conv4)
    conv4 = MaxPooling1D(pool_size=pool_size)(conv4)
    conv4 = Dropout(0.5)(conv4)
    lstm4 = Bidirectional(LSTM(lstm_output_size, return_sequences=True))(conv4)
    lstm4 = Dropout(0.5)(lstm4)
    lstm4 = Attention_layer()(lstm4)


    #text_sentence
    input_text_sent = Input(shape=(sequence_length_sent,), dtype='int32', name='text_sent_input')
    # Embedding layer
    emb_text_sent = Embedding(input_dim=vocabulary_size_sent, output_dim=emb_dim, input_length=sequence_length_sent,
                        W_regularizer=W_reg)(input_text_sent)



    emb_text_sent=Dropout(0.5)(emb_text_sent)

    # Conv layer
    conv5 = Convolution1D(kernel_size=kernel_size, filters=filters, border_mode='same')(emb_text_sent)
    conv5 = ELU()(conv5)
    conv5 = MaxPooling1D(pool_size=pool_size)(conv5)
    conv5 = Dropout(0.5)(conv5)
    # LSTM layer
    lstm5 =Bidirectional( LSTM(lstm_output_size, return_sequences=True))(conv5)

    lstm5 = Dropout(0.5)(lstm5)
    lstm5 = Attention_layer()(lstm5)

    
    x_text_output = concatenate([lstm4, lstm5], axis=1)
    #x_text_output = Dense(128, activation='relu')(x_text_output)


    x=concatenate([x_url_output,lstm3,x_text_output],axis=1)
    print('x.shape',x.shape)
    #x=Flatten()(x)
    x=Dense(256,activation='relu')(x)
    x=Dense(128, activation='relu')(x)
    x=Dense(64, activation='relu')(x)
    print('x.shape', x.shape)
    output = Dense(1, activation='sigmoid', name='output')(x)

    # Compile model and define optimizer
    model = Model(input=[input_url_char,input_url_word,input_dom,input_text_word,input_text_sent], output=[output])

    adam = Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
    return model



model = lstm_conv()
model.fit([X_train1,X_train4,X_train2,X_train3,X_train5], target_train1, epochs=epochs, batch_size=batch_size,validation_data=([X_test1,X_test4,X_test2,X_test3,X_test5],target_test1))
loss, accuracy = model.evaluate([X_test1,X_test4,X_test2,X_test3,X_test5], target_test1,verbose=1)




y_pred = model.predict([X_test1,X_test4,X_test2,X_test3,X_test5])

print("y_pred",y_pred)
m=[]
for i in y_pred:
    for j in i:
        if(j>=0.5):
            m.append(1)
        else:
            m.append(0)


y_pred1= m


print('\nFinal Cross-Validation Accuracy', accuracy, '\n')
recall = recall_score(target_test1, y_pred1 , average="binary")
precision = precision_score(target_test1, y_pred1 , average="binary")
f1 = f1_score(target_test1, y_pred1, average="binary")

print("racall")
print("%.6f" %recall)
print("precision")
print("%.6f" %precision)
print("f1score")
print("%.6f" %f1)


FPR,TPR,thresholds=roc_curve(target_test1,y_pred1)
roc_auc=auc(FPR,TPR)
print('FPR:',FPR)
print('TPR',TPR)

plt.plot(FPR,TPR,lw=1,label='Roc(area=%0.2f)'%(roc_auc))
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title("ROC(AUC=%0.2f)"%(roc_auc))
plt.show()

auc_score=roc_auc_score(target_test1,y_pred1)
print('auc:',auc_score)
confusion=confusion_matrix(y_true=target_test1,y_pred=y_pred1)
print(confusion)