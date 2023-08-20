import pandas as pd


def preprocess_data(df):
    df["FF"] = df["FF1 (lb/hr)"] + df["FF2 (lb/hr)"]
    df['Throttle'] = df['Throttle 2'] + df['Throttle 1']
    df["Horizontal Force"] = df['Thrust (N)'] + df['Drag (N)']
    keys = ["Ground Speed Dot (kt/s2)", "Mass (kg)",
            "FF", 'Throttle', "Horizontal Force"]
    processed_data = df[keys]
    return processed_data


def load_and_preprocess_data(file_path):
    df = pd.read_pickle(file_path)
    return preprocess_data(df)


def load_train_test_data(train_data_path, test_data_path):
    tr_data = load_and_preprocess_data(train_data_path)
    X_train = tr_data.drop(columns=["FF"])
    y_train = tr_data["FF"]

    test_data = load_and_preprocess_data(test_data_path)
    X_test = test_data.drop(columns=["FF"])
    y_test = test_data["FF"]

    return X_train, y_train, X_test, y_test


if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_train_test_data(
        "DataSet1_Q", "DataSet2_Q")
