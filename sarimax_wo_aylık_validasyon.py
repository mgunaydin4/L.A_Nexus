from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from itertools import product
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning
warnings.simplefilter('ignore', ConvergenceWarning)


df = pd.read_excel("proje.xlsx")
df["Order Date"] = pd.to_datetime(df["Order Date"], format="%d/%m/%Y")
df["Ship Date"] = pd.to_datetime(df["Ship Date"], format="%d/%m/%Y")

def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False


def remove_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    df_without_outliers = dataframe[~((dataframe[col_name] < low_limit) | (dataframe[col_name] > up_limit))]
    return df_without_outliers

check_outlier(df, "Sales")
df = remove_outlier(df, "Sales")

df = df.sort_values('Order Date')
df.set_index('Order Date', inplace=True)

sales_series = df['Sales'].resample('M').sum()

train_size = int(len(sales_series) * 0.8)
train, test = sales_series[:train_size], sales_series[train_size:]

p_values = range(0, 3)  
d_values = range(0, 2)  
q_values = range(0, 3)  
P_values = range(0, 3)  
D_values = range(0, 2)  
Q_values = range(0, 3)  
S_values = [12]  

# Grid search 
def hyperparameter_search_sarimax(train, test, p_values, d_values, q_values, P_values, D_values, Q_values, S_values):
    best_mse = float('inf')
    best_params = None
    best_model = None

    param_grid = list(product(p_values, d_values, q_values, P_values, D_values, Q_values, S_values))

    for param in param_grid:
        p, d, q, P, D, Q, S = param
        try:
            
            model = SARIMAX(train,
                            order=(p, d, q),
                            seasonal_order=(P, D, Q, S))
            model_fit = model.fit(disp=False)
            forecast = model_fit.forecast(len(test))

            mse = mean_squared_error(test, forecast)
            if mse < best_mse:
                best_mse = mse
                best_params = param
                best_model = model_fit
        except Exception as e:
            continue 

    return best_model, best_params, best_mse

best_model_sarimax, best_params_sarimax, best_mse_sarimax = hyperparameter_search_sarimax(
    train, test, p_values, d_values, q_values, P_values, D_values, Q_values, S_values)

print(f"En iyi SARIMAX Parametreler: {best_params_sarimax}")
print(f"En iyi SARIMAX Model Hatası (MSE): {best_mse_sarimax}")

forecast_sarimax = best_model_sarimax.forecast(len(test))

plt.figure(figsize=(12, 6))
plt.plot(test, label='Gerçek Değerler', marker='o')
plt.plot(test.index, forecast_sarimax, label='Tahminler', marker='x', linestyle='--', color='red')
plt.title('SARIMAX Modeli: Gerçek ve Tahmin Değerleri Kıyaslaması')
plt.xlabel('Tarih')
plt.ylabel('Satışlar')
plt.legend()
plt.show()

def plot_actual_vs_forecast(test, forecast_sarimax):
    plt.figure(figsize=(12, 6))
    plt.plot(test, label='Gerçek Değerler', marker='o')
    plt.plot(test.index, forecast_sarimax, label='Tahminler', marker='x', linestyle='--', color='red')
    plt.title('SARIMAX Modeli: Gerçek ve Tahmin Değerleri Kıyaslaması')
    plt.xlabel('Tarih')
    plt.ylabel('Satışlar')
    plt.legend()
    plt.show()
    return plt

plot_actual_vs_forecast(test, forecast_sarimax)

test_error_sarimax = mean_squared_error(test, forecast_sarimax)
print(f"SARIMAX Modeli Test Hatası (MSE): {test_error_sarimax}")
mae = mean_absolute_error(test, forecast_sarimax)
print(f"SARIMAX Modeli Test Hatası (MAE): {mae}")

future_steps = 12 
future_forecast_sarimax = best_model_sarimax.forecast(future_steps)
future_index = pd.date_range(start=sales_series.index[-1], periods=future_steps + 1, freq='M')[1:]

plt.figure(figsize=(12, 6))
plt.plot(future_index, future_forecast_sarimax, label='Gelecekteki Tahmin', marker='x', linestyle='--', color='green')
plt.title('SARIMAX Modeli: Gelecekteki Satış Tahmini (12 Ay)')
plt.xlabel('Tarih')
plt.ylabel('Satışlar')
plt.legend()
plt.show()

# CSV'ye kaydetme
future_df_sarimax = pd.DataFrame({'Tarih': test.index, 'Tahmin Satışlar': forecast_sarimax, 'Gerçek Satışlar': test})
future_df_sarimax.to_csv('gelecekteki_satis_tahminleri_sarimax_validasyonlu.csv', index=False)

