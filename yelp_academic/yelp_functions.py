from spacy.lang.en import English
from spacy.lang.en.stop_words import STOP_WORDS
import unidecode
import pickle


def gini_index_sample(list_of_values):
    list_of_values = sorted(list_of_values)
    n = len(list_of_values)
    top_sum, bottom_sum = 0, 0
    for i, value in enumerate(list_of_values):
        top_sum += i * value
        bottom_sum += value
    return 1 - (2 / (n - 1)) * (n - top_sum / bottom_sum)


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


def get_restaurant_vectors():
    with open("processed_data/restaurant_vectors", "rb") as pickle_file:
        restaurant_vectors = pickle.load(pickle_file)
    ids, embedding, price_range = restaurant_vectors[0], restaurant_vectors[1], restaurant_vectors[2]
    return ids, embedding, price_range

