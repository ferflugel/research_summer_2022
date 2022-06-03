import pandas as pd


def get_bikes_data():

    # import data for rentals
    rentals = pd.read_csv('../testing_methods/datasets/bike_sharing/day.csv')

    # create the vector for h and use one-hot-encoding
    hidden_variables = rentals[['mnth', 'temp', 'hum', 'windspeed', 'holiday', 'weekday']]
    hidden_variables = pd.get_dummies(hidden_variables, columns=['mnth'])

    # create the vector for x and use one-hot-encoding
    observations = rentals[['season', 'weathersit', 'workingday']]
    observations = pd.get_dummies(observations, columns=['season', 'weathersit'])

    # create vector for labels y and normalize the labels
    rental_labels = rentals[['casual', 'registered', 'cnt']]
    rental_labels /= rental_labels.max()

    return hidden_variables, observations, rental_labels    # h, x, y
