import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE, RandomOverSampler

def load_train_test_data():
    """
    Loads the data, selects the features, normalizes the values, and splits it into train/validation/test set
    """
    no_show_df = pd.read_csv("data/no_show_feature_engineered_no_extreme_locations_no_SMS_issues.csv")

    # encode the gender as a binary
    # NOTE: the gender didn't seem to affect no show by itself, but we're going to keep it and
    #       and verify if the model can still use it in conjunction with other variables
    no_show_df["isFemale"] = (no_show_df["gender"] == "F")

    # select the columns that we want to keep
    FEATURE_COLS = ["age","scholarship","hypertension","diabetes","alcoholism","handicap","smsSent",
                    "daysInAdvance","lat","lon","isFemale","distanceFromCenterLat","scheduledDayHour",
                    "otherAppointmentsOnSameDay","previouslyMissed"]
    days_of_weeks_cols = [col_name for col_name in no_show_df.columns if "appointmentDayDOW__" in col_name]
    FEATURE_COLS += days_of_weeks_cols

    # target column
    TARGET_COLUMN = "noShow"

    # prepare dataset for models by normalizing and scaling the features
    no_show_df["age"] = (no_show_df["age"]-no_show_df["age"].mean())/no_show_df["age"].std()

    no_show_df["daysInAdvance"] = (no_show_df["daysInAdvance"]-no_show_df["daysInAdvance"].min())/\
                                    (no_show_df["daysInAdvance"].max() - no_show_df["daysInAdvance"].min())

    no_show_df["scheduledDayHour"] = (no_show_df["scheduledDayHour"]-no_show_df["scheduledDayHour"].min())/\
                                    (no_show_df["scheduledDayHour"].max() - no_show_df["scheduledDayHour"].min())

    no_show_df["previouslyMissed"] = (no_show_df["previouslyMissed"]-no_show_df["previouslyMissed"].min())/\
                                    (no_show_df["previouslyMissed"].max() - no_show_df["previouslyMissed"].min())

    no_show_df["noShow"] = (no_show_df["noShow"])*1


    no_show_df = no_show_df.sort_values(['appointmentDay','scheduledDay'])

    # turns the DataFrame into np.array
    X = no_show_df[FEATURE_COLS].values
    y = no_show_df[TARGET_COLUMN].values

    SEQUENTIAL_SPLIT = False
    USE_SMOTE = False
    # get train = 60%, validation = 20%, test = 20%
    if not SEQUENTIAL_SPLIT:
        train_X, test_X, train_y, test_y = train_test_split(X,y,test_size=0.2,random_state=127) # split between train and test
        train_X, val_X, train_y, val_y = train_test_split(train_X,train_y,train_size=0.75,random_state=127) # split train to get validation

    else:
        train_X = X[:int(X.shape[0]*(0.8))]
        train_y = y[:int(y.shape[0]*(0.8))]
        test_X = X[int(X.shape[0]*(0.8)):]
        test_y = y[int(y.shape[0]*(0.8)):]
        
        # extracts validation from training set
        val_X = train_X[int(train_X.shape[0]*(0.75)):]
        val_y = train_y[int(train_y.shape[0]*(0.75)):]

        # remove validation from training set
        train_X = train_X[:int(train_X.shape[0]*(0.75))]
        train_y = train_y[:int(train_y.shape[0]*(0.75))]

        # sanity check
        assert len(train_X) + len(val_X) + len(test_X) == len(X), "Something went wrong while splitting up the sets"
    
    # Generating additional samples of the minority class to help with class imbalance
    if USE_SMOTE:
        oversample = SMOTE(random_state=127)
    else:
        oversample = RandomOverSampler(random_state=127)
    train_X, train_y = oversample.fit_resample(train_X, train_y)

    train_X = train_X.astype(np.float32)
    val_X = val_X.astype(np.float32)
    test_X = test_X.astype(np.float32)
    train_y = train_y.astype(np.int8)
    val_y = val_y.astype(np.int8)
    test_y = test_y.astype(np.int8)

    return train_X, train_y, val_X, val_y, test_X, test_y