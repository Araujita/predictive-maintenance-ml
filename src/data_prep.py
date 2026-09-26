import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE


def load_and_clean(path):
    """
    Loads the dataset and creates engineered features based on
    mechanical engineering domain knowledge (material fatigue).
    """
    df = pd.read_csv(path)

    # Fatigue features: combine physical variables to capture
    # accumulated tool wear, which is a strong predictor of failure
    df['Torque_Fatigue'] = df['Torque [Nm]'] * df['Tool wear [min]']
    df['Temp_Diff'] = df['Process temperature [K]'] - df['Air temperature [K]']
    df['Thermal_Fatigue'] = df['Temp_Diff'] * df['Tool wear [min]']

    columns_to_drop = ['UDI', 'Product ID', 'Type', 'Target', 'Failure Type']
    X = df.drop(columns_to_drop, axis=1, errors='ignore')
    y = df['Target']
    return X, y


def balance_data(X, y, test_size=0.2, random_state=42):
    """
    Performs a stratified train/test split (to preserve the failure
    ratio in both sets) and applies SMOTE only on the training set,
    avoiding data leakage into the test set.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    smote = SMOTE(random_state=random_state)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

    return X_train_smote, X_test, y_train_smote, y_test
