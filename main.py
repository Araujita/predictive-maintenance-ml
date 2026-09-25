import os
import matplotlib.pyplot as plt

from src.data_prep import carregar_e_limpar, balancear_dados
from src.train import comparar_modelos, treinar_modelo, obter_importancia_features
from src.evaluate import encontrar_melhor_limiar, calcular_roi_limiar_padrao

os.makedirs('images', exist_ok=True)

# 1. Preparar os dados
X, y = carregar_e_limpar('data/raw/predictive_maintenance.csv')
X_treino, X_teste, y_treino, y_teste = balancear_dados(X, y)

# 2. Comparar modelos com validação cruzada (evita escolher um modelo
#    apenas com base num único split, que pode ser enganoso)
print("=== Comparação de modelos (validação cruzada, 5-fold) ===")
resultados_cv = comparar_modelos(X_treino, y_treino, cv=5)
melhor_modelo_nome = resultados_cv.iloc[0]['modelo']
print(f"\nMelhor modelo: {melhor_modelo_nome}\n")

# 3. Treinar o modelo final escolhido
modelo = treinar_modelo(X_treino, y_treino, tipo=melhor_modelo_nome)

# 4. Avaliar financeiramente: comparar limiar padrão (0.5) vs. limiar otimizado
roi_padrao = calcular_roi_limiar_padrao(modelo, X_teste, y_teste)
limiar_ideal, lucro_estimado, historico_limiares = encontrar_melhor_limiar(
    modelo, X_teste, y_teste
)

print("=== Resultado financeiro ===")
print(f"ROI com limiar padrão (0.5):   {roi_padrao:,.0f} EUR")
print(f"ROI com limiar otimizado ({limiar_ideal:.2f}): {lucro_estimado:,.0f} EUR")
print(f"Ganho da otimização do limiar:  {lucro_estimado - roi_padrao:,.0f} EUR")

# 5. Importância das features (apenas para modelos baseados em árvores)
if hasattr(modelo, 'feature_importances_'):
    importancias = obter_importancia_features(modelo, X.columns)
    print("\n=== Importância das features ===")
    print(importancias.to_string(index=False))

    plt.figure(figsize=(8, 5))
    plt.barh(importancias['feature'][::-1], importancias['importancia'][::-1])
    plt.xlabel('Importância')
    plt.title(f'Importância das Features ({melhor_modelo_nome})')
    plt.tight_layout()
    plt.savefig('images/feature_importance.png', dpi=150)
    print("\nGráfico guardado em images/feature_importance.png")

# 6. Gráfico de ROI vs. limiar de decisão
plt.figure(figsize=(8, 5))
plt.plot(historico_limiares['limiar'], historico_limiares['lucro'])
plt.axvline(limiar_ideal, color='red', linestyle='--',
            label=f'Limiar ótimo = {limiar_ideal:.2f}')
plt.axhline(roi_padrao, color='gray', linestyle=':',
            label=f'ROI limiar padrão (0.5)')
plt.xlabel('Limiar de decisão')
plt.ylabel('ROI estimado (EUR)')
plt.title('ROI em função do limiar de decisão')
plt.legend()
plt.tight_layout()
plt.savefig('images/roi_vs_limiar.png', dpi=150)
print("Gráfico guardado em images/roi_vs_limiar.png")

print("\nO modelo está pronto a ser usado na fábrica!")
print(f"Limiar ideal: {limiar_ideal:.2f} | Poupança estimada: {lucro_estimado:,.0f} EUR")
