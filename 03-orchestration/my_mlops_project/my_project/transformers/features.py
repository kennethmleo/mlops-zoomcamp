# features.py
from mage_ai.data_preparation.decorators import custom
from sklearn.feature_extraction import DictVectorizer

@transformer
def features(data, **kwargs):
    """
    data: dict returned by load_data (df_train, df_val, ...)
    Returns: dict with matrices and labels ready for training.
    """
    df_train = data['df_train']
    df_val = data['df_val']

    categorical = ['PU_DO']
    numerical = ['trip_distance']
    target = 'duration'

    # Build dictionaries
    train_dicts = df_train[categorical + numerical].to_dict(orient='records')
    val_dicts = df_val[categorical + numerical].to_dict(orient='records')

    # Fit/transform vectorizer
    dv = DictVectorizer(sparse=True)
    X_train = dv.fit_transform(train_dicts)
    X_val = dv.transform(val_dicts)

    y_train = df_train[target].values
    y_val = df_val[target].values

    return {
        'X_train': X_train,
        'X_val': X_val,
        'y_train': y_train,
        'y_val': y_val,
        'dv': dv,
        # bubble up months for logging
        'train_year': data['train_year'],
        'train_month': data['train_month'],
        'val_year': data['val_year'],
        'val_month': data['val_month'],
    }