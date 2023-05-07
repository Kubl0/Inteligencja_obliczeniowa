from sklearn.preprocessing import LabelEncoder, MinMaxScaler


def withDeletion(df):
    label_encoder = LabelEncoder()
    df["type"] = label_encoder.fit_transform(df["type"])
    df = df.dropna()
    return df


def withMean(df):
    label_encoder = LabelEncoder()
    df["type"] = label_encoder.fit_transform(df["type"])
    df = df.fillna(df.mean(numeric_only=True))
    return df


def normalization(df):
    scaler = MinMaxScaler()
    columns_to_normalize = df.columns[:-1]
    df[columns_to_normalize] = scaler.fit_transform(df[columns_to_normalize])
    return df
