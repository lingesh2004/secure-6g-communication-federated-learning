import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def preprocess_data(file_path):
    data = pd.read_csv(file_path)

    data = data.dropna()

    scaler = MinMaxScaler()
    data.iloc[:, :-1] = scaler.fit_transform(data.iloc[:, :-1])

    return data

if __name__ == "__main__":
    dataset = preprocess_data("dataset.csv")
    print(dataset.head())
