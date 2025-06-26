# import libraries
import gzip
import json
import pickle

import pandas as pd


# wrangle function
def wrangle(filename):
    
    # unzip and load the json file
    with gzip.open(filename) as f:
        data = json.load(f)

    # read the data to dataframe
    df = pd.DataFrame().from_dict(data["observations"]).set_index("id")

    return df

def make_predictions(data_filepath, model_filepath):
    X_test = wrangle(data_filepath)
    
    # load the model
    with open(model_filepath, "rb") as f:
        model = pickle.load(f)

    # Generate predictions
    y_pred = model.predict(X_test)

    # Put predictions into Series with name "bankrupt", and same index as X_test
    y_pred = pd.Series(y_pred, index=X_test.index, name="bankrupt")

    return y_pred
    



