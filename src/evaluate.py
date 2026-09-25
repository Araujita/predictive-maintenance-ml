import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

# --- Pressupostos de custo (ajustar consoante o contexto real da fábrica) ---
#
# CUSTO_FALSO_POSITIVO: custo de uma inspeção/manutenção preventiva feita
#   sem necessidade (mão de obra + paragem curta da máquina).
#
# CUSTO_FALSO_NEGATIVO: custo de uma falha não detetada a tempo — paragem
#   não planeada, reparação de emergência, perda de produção e possíveis
#   danos adicionais ao equipamento. Tipicamente muito superior ao custo
#   de uma inspeção preventiva.
#
# BENEFICIO_VERDADEIRO_POSITIVO: poupança obtida ao evitar a falha através
#   de manutenção preventiva planeada, já descontado o custo dessa
#   manutenção (ou seja, é um valor líquido).
#
# Estes números são estimativas ilustrativas baseadas em cenários típicos
# de manutenção industrial em PMEs; num caso real, viriam de dados de
# manutenção da própria fábrica (custo médio de paragem por hora, custo
# de peças, custo de mão de obra de emergência, etc.).
CUSTO_FALSO_POSITIVO = -500
CUSTO_FALSO_NEGATIVO = -15000
BENEFICIO_VERDADEIRO_POSITIVO = 14000


def calcular_roi_unico(y_real, y_previsto):
    tn, fp, fn, tp = confusion_matrix(y_real, y_previsto).ravel()
    return (
        fp * CUSTO_FALSO_POSITIVO
        + fn * CUSTO_FALSO_NEGATIVO
        + tp * BENEFICIO_VERDADEIRO_POSITIVO
    )


def encontrar_melhor_limiar(modelo, X_teste, y_teste):
    """
    Procura o limiar de decisão (threshold) que maximiza o ROI financeiro,
    em vez de usar o limiar padrão de 0.5, que otimiza para métricas
    estatísticas (accuracy/F1) e não para o impacto económico real.

    Retorna também o histórico completo, útil para gerar o gráfico de
    ROI vs. limiar.
    """
    probabilidades = modelo.predict_proba(X_teste)[:, 1]

    melhor_limiar = 0.5
    lucro_maximo = -float('inf')
    historico = []

    for limiar in np.arange(0.01, 1.0, 0.01):
        previsoes = (probabilidades >= limiar).astype(int)
        lucro = calcular_roi_unico(y_teste, previsoes)
        historico.append((limiar, lucro))

        if lucro > lucro_maximo:
            lucro_maximo = lucro
            melhor_limiar = limiar

    df_historico = pd.DataFrame(historico, columns=['limiar', 'lucro'])
    return melhor_limiar, lucro_maximo, df_historico


def calcular_roi_limiar_padrao(modelo, X_teste, y_teste):
    """
    ROI usando o limiar padrão (0.5), para servir de linha de base e
    quantificar o ganho obtido apenas por otimizar o limiar de decisão.
    """
    previsoes = (modelo.predict_proba(X_teste)[:, 1] >= 0.5).astype(int)
    return calcular_roi_unico(y_teste, previsoes)
