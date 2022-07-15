import numpy as np
from sklearn.feature_selection import mutual_info_regression
from tqdm.notebook import tqdm


def multiple_correlation(df, y_name, z_list):
    c = df[[y_name] + z_list].corr()['y'].iloc[1:].to_numpy()
    R = df[[y_name] + z_list].corr().loc[z_list, z_list].to_numpy()
    return c.T @ np.linalg.inv(R) @ c


def mutual_information_gap(df, h_list, z_list):
    mig = [individual_mig(df, h_k, z_list) for h_k in h_list]
    return np.mean(mig)


def separated_attribute_predictability(df, h_list, z_list):
    sap = [individual_sap(df, h_k, z_list) for h_k in h_list]
    return np.mean(sap)


def inverted_kruskals_stress(df, x_list, z_list, accuracy=0.01, silent=False):
    x, z = df[x_list].to_numpy(), df[z_list].to_numpy()
    min_stress = minimum_stress(x, z, accuracy, silent)
    return 1 - min_stress


"""HELPER FUNCTIONS"""


# define the distance that will be used for the kruskal's stress
def distance(a, b):
    return np.linalg.norm(a - b)


# calculate the MIG for a single h_k
def individual_mig(df, h_k, z_list):
    mutual_information = mutual_info_regression(df[z_list], df[h_k])
    first = np.max(mutual_information)
    second = np.max(mutual_information[mutual_information != first])
    return first - second


# calculate the SAP for a single h_k
def individual_sap(df, h_k, z_list):
    R_squared = (df[[h_k] + z_list].corr() ** 2).iloc[0, 1:]
    first = R_squared.max()
    second = R_squared[R_squared != first].max()
    return first - second


# define the stress function
def stress_function(x, z, scale_factor):
    top_sum, bottom_sum = 0, 0
    for i in range(x.shape[0]):
        for j in range(i):
            top_sum += (distance(x[i], x[j]) - scale_factor * distance(z[i], z[j])) ** 2
            bottom_sum += distance(x[i], x[j]) ** 2
    return np.sqrt(top_sum / bottom_sum)


# search for the minimum kruskal's stress
def minimum_stress(x, z, accuracy, silent):

    lower_bound, upper_bound, window_size = 0.2, 5, 4.8

    while window_size > accuracy:
        if not silent:
            print(f'Search Range: [{lower_bound:.5f}, {upper_bound:.5f}]')
            search_range, stress_list = np.linspace(lower_bound, upper_bound, 10), []
            for scale in tqdm(search_range):
                stress_list.append(stress_function(x, z, scale))
        else:
            search_range, stress_list = np.linspace(lower_bound, upper_bound, 10), []
            for scale in search_range:
                stress_list.append(stress_function(x, z, scale))

        argmin = np.argmin(stress_list)

        if argmin == 0:
            lower_bound = max(0, lower_bound - window_size)
            upper_bound = search_range[1]
        elif argmin == 9:
            lower_bound = search_range[8]
            upper_bound = upper_bound + window_size
        else:
            lower_bound = search_range[argmin - 1]
            upper_bound = search_range[argmin + 1]

        window_size = upper_bound - lower_bound
        if not silent:
            print(f'Inverted Stress: {1 - stress_list[argmin]:.3f}, New Window Size: {window_size:.5f}')

    return stress_function(x, z, 0.5 * (upper_bound + lower_bound))
