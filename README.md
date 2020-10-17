# Web2Vec Phishing Webpage Detection Method Based on Multidimensional Features Driven by Deep Learning
## 1.Overview
This is about the web2vec paper code and data description.We will describe the operating environment, data set source and program description
## 2.Operating environment
Windows7, 10 or Linux  
Anaconda3  
Python3.5.4  
Keras2.2.4  
numpy1.13.1  
TensorFlow1.2.1  
sklearn0.19.2  
## 3.Datasets
Normal Pages：from PhishTank.com  
Phishing Pages：from Alexa.com  
Since the original data set is large, we will give the cloud disk link: https://pan.baidu.com/s/19HAxZmfsP_fd-g9v20PNog      Extraction code：ipk8  
preprocessing data of link: https://pan.baidu.com/s/1BFf2wXpV-JXyK2N9KPSODQ  Extraction code：zf11  
## 4.Programs
Input normal-page and phishing-page datasets -> Parsing out required page-features and saving them into files -> Build a deep learning model ->Feature extraction and feature learning-> Output evaluation index
### 4.1.Input File
Two folders are going to be required for program input. One folder includes the known phishing websites, and another folder includes the known normal websites.  
Each folder includes two types of files:  
*URL files: URL information for each webpage  
*HTML files: HTML source code for each webpage  
Notice: Same page are renamed by the same prefix(page1.url, page2.html)  
### 4.2.Python Programs
The web2vec model input file is the text content, URL, and DOM structure obtained after the original webpage is parsed。E.g：data2  
## 5.Usage
### 5.1.dom.py 
It is DOM structure preprocessing code, used to realize data preprocessing and encoding  
### 5.2.HTML_sentence.py and HTML_word1.py 
They are text content word and sentence preprocessing codes, used to realize data preprocessing and encoding  
### 5.3.URL_char.py and URL_word.py 
they are URL character and word preprocessing codes, which implement data preprocessing and encoding  
### 5.4.Attention_layer.py
It is the attention mechanism code  
### 5.5.Web2vec.py
It is the web2vec model code, which implements feature extraction, feature learning and category prediction
