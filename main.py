import os
import matplotlib.pyplot as plt

from src.data_prep import load_and_clean, balance_data
from src.train import compare_models, train_model, get_feature_importance
from src.evaluate import find_best_threshold, calculate_default_threshold_roi

os.makedirs('images', exist_ok=True)

# 1. Prepare the data
X, y = load_and_clean('data/raw/predictive_maintenance.csv')
X_train, X_test, y_train, y_test = balance_data(X, y)

# 2. Compare models with cross-validation (avoids picking a model based
#    solely on a single split, which can be misleading)
print("=== Model comparison (5-fold cross-validation) ===")
cv_results = compare_models(X_train, y_train, cv=5)
best_model_name = cv_results.iloc[0]['model']
print(f"\nBest model: {best_model_name}\n")

# 3. Train the chosen final model
model = train_model(X_train, y_train, model_type=best_model_name)

# 4. Financial evaluation: compare default threshold (0.5) vs. optimized threshold
default_roi = calculate_default_threshold_roi(model, X_test, y_test)
best_threshold, estimated_profit, threshold_history = find_best_threshold(
    model, X_test, y_test
)

print("=== Financial result ===")
print(f"ROI with default threshold (0.5):    {default_roi:,.0f} EUR")
print(f"ROI with optimized threshold ({best_threshold:.2f}): {estimated_profit:,.0f} EUR")
print(f"Gain from threshold optimization:     {estimated_profit - default_roi:,.0f} EUR")

# 5. Feature importance (tree-based models only)
if hasattr(model, 'feature_importances_'):
    importance_df = get_feature_importance(model, X.columns)
    print("\n=== Feature importance ===")
    print(importance_df.to_string(index=False))

    plt.figure(figsize=(8, 5))
    plt.barh(importance_df['feature'][::-1], importance_df['importance'][::-1])
    plt.xlabel('Importance')
    plt.title(f'Feature Importance ({best_model_name})')
    plt.tight_layout()
    plt.savefig('images/feature_importance.png', dpi=150)
    print("\nChart saved to images/feature_importance.png")

# 6. ROI vs. decision threshold chart
plt.figure(figsize=(8, 5))
plt.plot(threshold_history['threshold'], threshold_history['profit'])
plt.axvline(best_threshold, color='red', linestyle='--',
            label=f'Optimal threshold = {best_threshold:.2f}')
plt.axhline(default_roi, color='gray', linestyle=':',
            label=f'Default threshold ROI (0.5)')
plt.xlabel('Decision threshold')
plt.ylabel('Estimated ROI (EUR)')
plt.title('ROI as a function of the decision threshold')
plt.legend()
plt.tight_layout()
plt.savefig('images/roi_vs_threshold.png', dpi=150)
print("Chart saved to images/roi_vs_threshold.png")

print("\nThe model is ready to be used on the factory floor!")
print(f"Optimal threshold: {best_threshold:.2f} | Estimated savings: {estimated_profit:,.0f} EUR")
