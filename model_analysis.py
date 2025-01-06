from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def evaluation_metrics(test, pred):
    mae = mean_absolute_error(test, pred)
    mse = mean_squared_error(test, pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(test, pred)

    print(f"Ortalama Mutlak Hata {mae}")
    print(f"Ortalama Karesel Hata: {mse}")
    print(f"Kök Ortalama Karesel Hata: {rmse}")
    print(f"R² Skor: {r2}")


def residual_analysis(test, pred):
    residuals = test - pred

    plt.scatter(pred, residuals)
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel("Predicted")
    plt.ylabel("Residuals")
    plt.title("Residuals vs. Predicted")
    plt.show()

    sns.histplot(residuals, kde=True)
    plt.title("Histogram of Residuals")
    plt.show()