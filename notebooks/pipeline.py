# pipeline.py
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder

def create_wildfire_pipeline(numeric_features, categorical_features, *, encoding="ordinal"):
    """
    encoding: "ordinal" (best for trees) or "onehot" (for linear models)
    """

    # Trees don’t need scaling, but harmless if kept. You can drop scaler if you want.
    num = Pipeline(steps=[('scaler', StandardScaler())])

    if encoding == "ordinal":
        cat = Pipeline(steps=[
            ('enc', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
        ])
    else:
        from sklearn.preprocessing import OneHotEncoder
        cat = Pipeline(steps=[
            ('enc', OneHotEncoder(handle_unknown='ignore', sparse_output=True))
        ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num, numeric_features),
            ('cat', cat, categorical_features)
        ]
    )
    return preprocessor
