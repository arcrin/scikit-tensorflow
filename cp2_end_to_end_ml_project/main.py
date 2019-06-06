#%% imports
import os
import tarfile
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from six.moves import urllib
from zlib import crc32
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.impute import SimpleImputer
from pandas.plotting import scatter_matrix



#%%
DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

#%% download data
def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()


fetch_housing_data()

#%% load data
def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")

    return pd.read_csv(csv_path)


housing = load_housing_data()
housing.head()

#%%
housing.hist(bins=50, figsize=(20, 15))
plt.show()

#%% Create a test
def split_train_test(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]


def test_set_check(identifier, test_ratio):
    return crc32(np.int64(identifier)) & 0xffffffff < test_ratio * 2**32


def split_train_test_by_id(data, test_ratio, id_column):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio))
    return data.loc[~in_test_set], data.loc[in_test_set]

#%%
housing_with_id = housing.reset_index()
# train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "index")


housing_with_id['id'] = housing['longitude'] * 1000 + housing['latitude']
train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, 'id')


#%% Strat
housing["income_cat"] = pd.cut(housing["median_income"],
                               bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                               labels=[1, 2, 3, 4, 5])
#%%
housing["income_cat"].hist()
plt.show()

#%%
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

#%% Drop income_cat
for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)

#%%
housing = strat_train_set.copy()

#%%
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)
plt.show()


#%%
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
             s=housing["population"]/100, label="population", figsize=(10,7),
             c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True)
plt.legend()
plt.show()

#%% Looking for Correlations
corr_matrix = housing.corr()

#%%
attributes = ["median_house_value", "median_income", "total_rooms", "housing_median_age"]
scatter_matrix(housing[attributes], figsize=(12, 8))
plt.show()

#%%
housing.plot(kind="scatter", x="median_income", y="median_house_value", alpha=0.1)
plt.show()

#%%
housing["rooms_per_household"] = housing["total_rooms"]/housing["household"]
housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
housing["popylation_per_household"] = housing["population"]/housing["households"]

#%%
corr_matrix = housing.corr()

#%% Data cleaning
housing.dropna(subset=["total_bedrooms"])
housing.drop("total_bedrooms", axis=1)
median = housing["total_bedrooms"].median()
housing["total_bedrooms"].fillna(median, inplace=True)

#%%
imputer = SimpleImputer(strategy="median")


#%%
housing_num = housing.drop("ocean_proximity", axis=1)

#%%
imputer.fit(housing_num)

#%%
X = imputer.transform(housing_num)

#%%
housing_tr = pd.DataFrame(X, columns=housing_num.columns)