import numpy as np
import re
import itertools
from collections import Counter


data_nor="./data2/text/norContext.txt"
data_ph="./data2/text/ph.txt"
sequence_lengths=10  


def clean_str(string):
   

    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " , ", string)
    string = re.sub(r"\?", " , ", string)
    string = re.sub(r"\s{2,}", " , ", string)
    string = re.sub(r":", " , ", string)
    string = re.sub(r";", " , ", string)
    string = re.sub(r"\.", " , ", string)
    string = re.sub(r"!", " , ", string)
    return string.strip().lower()


def load_data_and_labels():
    
    # Load data from files
    positive_examples = list(open(data_nor, "r", encoding='utf-8').readlines())
    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = list(open(data_ph, "r", encoding='utf-8').readlines())
    negative_examples = [s.strip() for s in negative_examples]
    # Split by words
    x_text = positive_examples + negative_examples
    x_text = [clean_str(sent) for sent in x_text]
    x_text = [s.split(",") for s in x_text]

    # Generate labels
    positive_labels = [0 for _ in positive_examples]
    negative_labels = [1 for _ in negative_examples]
    y = np.concatenate([positive_labels, negative_labels])
    return [x_text, y]


def pad_sentences(sentences, padding_word="<PAD/>"):
    
    sequence_length = sequence_lengths
    padded_sentences = []
    for i in range(len(sentences)):
        sentence = sentences[i]
        num_padding = sequence_length - len(sentence)
        if num_padding>=0:
            new_sentence = sentence + [padding_word] * num_padding
            padded_sentences.append(new_sentence)
        else:
            padded_sentences.append(sentence[:sequence_lengths])
    return padded_sentences


def build_vocab(sentences):
    
    # Build vocabulary
    word_counts = Counter(itertools.chain(*sentences))
    # Mapping from index to word
    vocabulary_inv = [x[0] for x in word_counts.most_common()]
    vocabulary_inv = list(sorted(vocabulary_inv))
    # Mapping from word to index
    vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
    return [vocabulary, vocabulary_inv]


def build_input_data(sentences, labels, vocabulary):
  
    x = np.array([[vocabulary[word] for word in sentence] for sentence in sentences])
    y = np.array(labels)
    return [x, y]


def load_data_sentences():
    
    # Load and preprocess data
    sentences, labels = load_data_and_labels()
    sentences_padded = pad_sentences(sentences)
    vocabulary, vocabulary_inv = build_vocab(sentences_padded)
    x, y = build_input_data(sentences_padded, labels, vocabulary)
    return [x, y, vocabulary, vocabulary_inv]


load_data_sentences()