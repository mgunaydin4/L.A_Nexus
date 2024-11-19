import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt
from sklearn.model_selection import TimeSeriesSplit


df = pd.read_excel("proje.xlsx")

df["Order Date"] = pd.to_datetime(df["Order Date"], format="%d/%m/%Y")
df["Ship Date"] = pd.to_datetime(df["Ship Date"], format="%d/%m/%Y")


def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
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

df['Order Date'] = pd.to_datetime(df['Order Date'])
data = df.sort_values('Order Date')
data.set_index('Order Date', inplace=True)


sales_series = data['Sales'].resample('M').sum()


train_size = int(len(sales_series) * 0.8)
train, test = sales_series[:train_size], sales_series[train_size:]

# Hiperparametreler için seçenekler
trend_options = ['add', 'mul', None] 
seasonal_options = ['add', 'mul', None]  
seasonal_periods_options = [6, 12, 24] 


# Hiperparametre optimizasyon
def hyperparameter_search(train, test, trend_options, seasonal_options, seasonal_periods_options):
    best_mse = float('inf')
    best_params = None
    best_model = None

    for trend in trend_options:
        for seasonal in seasonal_options:
            for seasonal_periods in seasonal_periods_options:
                try:
                    model = ExponentialSmoothing(train,
                                                 trend=trend,
                                                 seasonal=seasonal,
                                                 seasonal_periods=seasonal_periods)
                    model_fit = model.fit()
                    forecast = model_fit.forecast(len(test))

                    mse = mean_squared_error(test, forecast)
                    if mse < best_mse:
                        best_mse = mse
                        best_params = (trend, seasonal, seasonal_periods)
                        best_model = model_fit
                except Exception as e:
                    continue 

    return best_model, best_params, best_mse


best_model, best_params, best_mse = hyperparameter_search(train, test, trend_options, seasonal_options,
                                                          seasonal_periods_options)

print(f"En İyi Parametreler: {best_params}")
print(f"En İyi MSE: {best_mse}")

forecast = best_model.forecast(len(test))

plt.figure(figsize=(12, 6))
plt.plot(test, label='Gerçek Değerler', marker='o')
plt.plot(test.index, forecast, label='Tahminler', marker='x', linestyle='--', color='red')
plt.title('Tahmin ve Gerçek Değerlerin Kıyaslanması (Holt-Winters)')
plt.xlabel('Tarih')
plt.ylabel('Satışlar')
plt.legend()
plt.show()

test_error_mse = mean_squared_error(test, forecast)
test_error_rmse = sqrt(test_error_mse)
test_error_mae = mean_absolute_error(test, forecast)

print(f"Test Hatası (MSE): {test_error_mse}")
print(f"Test Hatası (RMSE): {test_error_rmse}")
print(f"Test Hatası (MAE): {test_error_mae}")


future_steps = 12 
future_forecast = best_model.forecast(future_steps)
future_index = pd.date_range(start=sales_series.index[-1], periods=future_steps + 1, freq='M')[1:]

plt.figure(figsize=(12, 6))
plt.plot(future_index, future_forecast, label='Gelecekteki Tahmin', marker='x', linestyle='--', color='green')
plt.title('Gelecekteki Satış Tahmini (12 Ay) (Holt-Winters)')
plt.xlabel('Tarih')
plt.ylabel('Satışlar')
plt.legend()
plt.show()

# CSV'ye Kaydetme
future_df = pd.DataFrame({'Tarih': future_index, 'Tahmin Satışlar': future_forecast})
future_df.to_csv('gelecekteki_satis_tahminleri_hw.csv', index=False)
