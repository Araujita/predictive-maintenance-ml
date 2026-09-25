import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE


def carregar_e_limpar(caminho):
    """
    Carrega o dataset e cria features de engenharia baseadas em
    conhecimento de engenharia mecânica (fadiga do material).
    """
    df = pd.read_csv(caminho)

    # Features de fadiga: combinam variáveis físicas para capturar
    # o desgaste acumulado da ferramenta, que é um bom preditor de falha
    df['Fadiga_Torque'] = df['Torque [Nm]'] * df['Tool wear [min]']
    df['Diferenca_Temp'] = df['Process temperature [K]'] - df['Air temperature [K]']
    df['Fadiga_Termica'] = df['Diferenca_Temp'] * df['Tool wear [min]']

    colunas_ignorar = ['UDI', 'Product ID', 'Type', 'Target', 'Failure Type']
    X = df.drop(colunas_ignorar, axis=1, errors='ignore')
    y = df['Target']
    return X, y


def balancear_dados(X, y, test_size=0.2, random_state=42):
    """
    Faz o split treino/teste de forma estratificada (para preservar a
    proporção de falhas em ambos os conjuntos) e aplica SMOTE apenas
    no treino, evitando fuga de informação (data leakage) para o teste.
    """
    X_treino, X_teste, y_treino, y_teste = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    smote = SMOTE(random_state=random_state)
    X_treino_smote, y_treino_smote = smote.fit_resample(X_treino, y_treino)

    return X_treino_smote, X_teste, y_treino_smote, y_teste
