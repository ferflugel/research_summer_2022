from spacy.lang.en import English
from spacy.lang.en.stop_words import STOP_WORDS
import unidecode
import gensim.downloader as api
import numpy as np


def get_processed_inputs(df, column='text', mode='list_of_strings'):
    # copies the dataframe for a temporary variable
    temp = df.loc[:]

    # strip accents and use lowercase for all the text
    temp[column] = [unidecode.unidecode(review_text).lower() for review_text in temp[column]]

    # tokenize the reviews using spaCy and remove stop words
    spacy_tokenizer = English()
    temp['spacy_token'] = [[token.text for token in spacy_tokenizer(review_text)] for review_text in temp[column]]
    temp['spacy_token'] = [list(filter(lambda word: word not in STOP_WORDS, list_of_tokens)) for list_of_tokens in
                           temp['spacy_token']]

    if mode == 'list_of_strings':
        # create a list with inputs in the ideal format for BoW and TF-IDF
        return [' '.join(review) for review in temp['spacy_token'].tolist()]

    elif mode == 'list_of_lists':
        return temp['spacy_token'].tolist()

    else:
        return [' '.join(review) for review in temp['spacy_token'].tolist()]


def sum_to_one(input_vector):
    return [vector/sum(vector) if sum(vector) != 0 else vector for vector in input_vector]


def word2vec(df, column='text', mode='vector'):

    # getting the pre-trained word2vec model
    w2v = api.load('word2vec-google-news-300')

    # processing inputs
    w2v_inputs = get_processed_inputs(df, column=column, mode='list_of_lists')

    # create the word2vec list of vectors
    w2v_array = []

    if mode == 'vector':
        for text in w2v_inputs:
            text_array = np.zeros(300)
            len_count = 0
            for token in text:
                if token in w2v:
                    text_array += w2v[token]
                    len_count += 1
            if len_count != 0:
                text_array /= len_count
            w2v_array.append(text_array)

    elif mode == 'matrix':
        for text in w2v_inputs:
            text_matrix = []
            count = 0

            if len(text) >= 300:
                text = text[:300]
                for token in text:
                    if token in w2v:
                        text_matrix.append(w2v[token])
                    else:
                        count += 1
            else:
                for token in text:
                    if token in w2v:
                        text_matrix.append(w2v[token])
                    else:
                        count += 1
                for i in range(300 - len(text)):
                    text_matrix.append(np.zeros(300))

            for i in range(count):
                text_matrix.append(np.zeros(300))

            w2v_array.append(np.array(text_matrix))

    return w2v_array
