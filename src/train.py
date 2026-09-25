import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score

MODELOS_DISPONIVEIS = {
    'RandomForest': lambda: RandomForestClassifier(random_state=42, n_jobs=-1),
    'LogisticRegression': lambda: LogisticRegression(max_iter=1000, random_state=42),
    'GradientBoosting': lambda: GradientBoostingClassifier(random_state=42),
}


def comparar_modelos(X_treino, y_treino, cv=5, scoring='f1'):
    """
    Compara vários modelos com validação cruzada estratificada (em vez de
    confiar num único split treino/teste, que pode dar resultados enganosos
    por sorte/azar na divisão dos dados).

    Retorna um DataFrame ordenado pelo score médio, para facilitar a escolha
    do modelo final.
    """
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    resultados = []

    for nome, construtor in MODELOS_DISPONIVEIS.items():
        modelo = construtor()
        scores = cross_val_score(
            modelo, X_treino, y_treino, cv=skf, scoring=scoring, n_jobs=-1
        )
        resultados.append({
            'modelo': nome,
            f'{scoring}_medio': scores.mean(),
            f'{scoring}_std': scores.std(),
        })
        print(f"{nome}: {scoring} = {scores.mean():.4f} (+/- {scores.std():.4f})")

    df_resultados = pd.DataFrame(resultados).sort_values(
        f'{scoring}_medio', ascending=False
    ).reset_index(drop=True)

    return df_resultados


def treinar_modelo(X_treino, y_treino, tipo='RandomForest'):
    """
    Treina o modelo final (já escolhido via comparar_modelos) no conjunto
    de treino completo.
    """
    if tipo not in MODELOS_DISPONIVEIS:
        raise ValueError(
            f"Modelo '{tipo}' não reconhecido. Opções: {list(MODELOS_DISPONIVEIS)}"
        )
    modelo = MODELOS_DISPONIVEIS[tipo]()
    modelo.fit(X_treino, y_treino)
    return modelo


def obter_importancia_features(modelo, nomes_features):
    """
    Extrai a importância de cada feature (apenas para modelos baseados em
    árvores, como RandomForest e GradientBoosting). Ajuda a interpretar
    quais leituras de sensores mais contribuem para prever falhas.
    """
    if not hasattr(modelo, 'feature_importances_'):
        raise ValueError(
            f"O modelo {type(modelo).__name__} não expõe feature_importances_"
        )

    importancias = pd.DataFrame({
        'feature': nomes_features,
        'importancia': modelo.feature_importances_
    }).sort_values('importancia', ascending=False).reset_index(drop=True)

    return importancias
