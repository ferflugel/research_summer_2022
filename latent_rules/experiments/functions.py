from representation_learning.metrics import multiple_correlation, inverted_kruskals_stress
import numpy as np
import pandas as pd
from sklearn.decomposition import FactorAnalysis, PCA, FastICA
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt

h_variables = [f'h_{i}' for i in range(2)]
x_variables = [f'x_{i}' for i in range(3)]
z_variables = [f'z_{i}' for i in range(2)]
methods = [FactorAnalysis, PCA, FastICA]


def generate_hidden_factors(samples=500, seed=0):

    # Generate Gaussian hidden factors
    np.random.seed(seed)
    hidden = np.random.normal(0, 1, size=(2, samples))

    # Create a dataframe to carry the data
    data = pd.DataFrame(hidden.T, columns=h_variables)

    # Generate labels from a linear combination of hidden factors
    A = np.random.normal(1, 1, size=(1, 2))
    labels = (A @ hidden).T
    data['y'] = labels

    return data


def generate_observations(data):

    # Generate observations from non-linear transformations
    data['x_0'] = random_transformation(data['h_0'])
    data['x_1'] = random_transformation(data['h_1'])
    data['x_2'] = random_transformation(data['h_1'] + data['h_1'])

    # Normalize all observations to have mean of 0 and variance of 1
    for i in range(3):
        data[f'x_{i}'] = (data[f'x_{i}'] - data[f'x_{i}'].mean()) / data[f'x_{i}'].std()

    # Split the data into train and test sets
    data_train, data_test = train_test_split(data, test_size=0.25, random_state=0)
    data_train = data_train.reset_index(drop=True)
    data_test = data_test.reset_index(drop=True)

    return data_train, data_test


def run_experiment(data_train, data_test, noise_range, smoothness='kruskal', verbose=True):

    results = [[[], [], []] for _ in range(len(noise_range))]

    for trial, std in enumerate(noise_range):

        # Add noise to the data after creating a deep copy
        noisy_data_train = data_train.copy()
        noisy_data_test = data_test.copy()
        for i in range(3):
            noisy_data_train[f'x_{i}'] = noisy_data_train[f'x_{i}'] + np.random.normal(0, std, data_train.shape[0])
            noisy_data_test[f'x_{i}'] = noisy_data_test[f'x_{i}'] + np.random.normal(0, std, data_test.shape[0])

        # Obtain the individual latent spaces
        (fa, pca, ica), jacobians = get_latent_spaces(noisy_data_train, noisy_data_test, methods)

        # Get metrics for each method
        for index, latent_space in enumerate([fa, pca, ica]):
            results[trial][0].append(multiple_correlation(latent_space, 'y', z_variables))
            if smoothness == 'kruskal':
                results[trial][1].append(inverted_kruskals_stress(latent_space, x_variables,
                                                                  z_variables, 0.01, silent=True))
            elif smoothness == 'jacobian':
                results[trial][1].append(1 / np.linalg.norm(jacobians[index]))
            else:
                print('Invalid smoothness method')
                exit()

        if verbose:
            print(f'Done std {std}')

    return results


def get_correlations(results):
    correlation = []
    for result in results:
        correlation.append(np.corrcoef(result[0], result[1])[0, 1])
    initial_predictivity = np.array(results[0][0]).mean()
    return correlation, initial_predictivity


def plot_correlations(results, noise_range, trial):
    correlation = []
    for result in results:
        correlation.append(np.corrcoef(result[0], result[1])[0, 1])

    sns.set_style('white'); sns.set_palette('pastel')
    sns.scatterplot(x=noise_range, y=correlation)
    sns.despine()
    plt.xlabel('Noise Magnitude'); plt.ylabel('Correlation')
    plt.savefig(f'results/noise_vs_correlation_{trial}.png', dpi=200)
    plt.show()


"""HELPER FUNCTIONS"""


def get_latent_spaces(df_train, df_test, latent_methods):

    # List to store dataframes for each latent representation
    latent_spaces, jacobians = [], []

    # Create dataframes for each method
    for latent_method in latent_methods:
        latent_space, jacobian = create_latent_space(df_train, df_test, latent_method)
        latent_spaces.append(latent_space)
        jacobians.append(jacobian)

    return latent_spaces, jacobians


def create_latent_space(df_train, df_test, latent_method):

    # Train the method to create latent representations and fit test data
    if latent_method == FastICA:
        mapping = latent_method(n_components=2, random_state=0, max_iter=2000).fit(df_train[x_variables])
    else:
        mapping = latent_method(n_components=2, random_state=0).fit(df_train[x_variables])
    latent = mapping.transform(df_test[x_variables])

    # Add created representation to the dataframe
    latent = pd.DataFrame(latent, columns=z_variables)
    latent = pd.concat([df_test, latent], axis=1)

    return latent, mapping.components_


# Apply a random non-linear transformation to the data
def random_transformation(x):
    functions = {0: np.sin, 1: np.exp, 2: np.tanh,
                 3: np.sign, 4: np.abs, 5: relu, 6: sigmoid,
                 7: np.cos, 8: np.round, 9: np.ceil}
    f_choice = np.random.randint(low=0, high=len(functions), size=1)[0]
    return functions[f_choice](x)


def relu(x):
    return np.maximum(0, x)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


