import datetime as dt
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from lifetimes.plotting import plot_period_transactions
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = "browser"

warnings.filterwarnings("ignore")

pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.2f' % x)
pd.set_option('display.width', 500)

df_copy = pd.read_csv('train.csv')
df = df_copy.copy()
##################################################### DATA #####################################################
#DEĞİŞKENLER
"""
ORDER ID:      SİPARİŞ ID
Order Date:    SİPARİŞ TARİHİ
Ship Date:     GÖNDERİM TARİHİ
Ship Mode:     GÖNDERİM ŞEKLİ
Customer ID:   KULLANICI ID
Customer Name: KULLANICI İSMİ
Segment:       MÜŞTERİNİN TÜRÜ
Country:       ÜLKE
CİTY:          ŞEHİR
STATE:         İLÇE
POSTAL:        POSTA KODU
REGİON:        BÖLGE (GÜNEY, KUZEY VS)
Category:      KATEGORİ
Sub-Category:  ALT KATAGORİ
Product Name:  ÜRÜN ADI
SALES:         SATIŞ
"""


######################### Veriye İlk Bakış #########################
def check_df(dataframe, count=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(count))
    print("##################### Tail #####################")
    print(dataframe.tail(count))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.describe([0, 0.05, 0.50, 0.95, 0.99, 1]).T)
    print("##################### Unique #####################")
    print(dataframe.nunique())
    print("##################### Total Unique #####################")
    print(dataframe.nunique().sum())

check_df(df)




########################### NaN Değerlerin kaldırılması
df = df.dropna()

check_df(df)
############################ Gereksiz Columnların Kaldırılması ############################

#drop_col = ["Row ID", "Customer Name", "Country", "Postal Code", "Product ID"]
drop_col = ["Row ID", "Country", "Postal Code"]


df.drop(drop_col, axis=1, inplace=True)

df.shape

#Değişken Düzenlemesi
df['Order Date'] = pd.to_datetime(df['Order Date'], format='%d/%m/%Y')
df['Ship Date'] = pd.to_datetime(df['Ship Date'], format='%d/%m/%Y')

##################### Kategorik Değişkenlerin Analizi #####################
def cat_summary(dataframe, plot=False):
    # Kategorik sütunları tespit etme
    cat_columns = dataframe.select_dtypes(include=['object', 'category','datetime64[ns]']).columns

    for col_name in cat_columns:
        # Kategorik değişkenin değer sayımlarını ve oranlarını yazdırma
        value_counts = dataframe[col_name].value_counts()
        ratio = 100 * value_counts / len(dataframe)
        print(f"Summary for column: {col_name}")
        print(pd.DataFrame({col_name: value_counts, "Ratio": ratio}))
        print("##########################################")

        # Eğer plot=True ise, barplot ve piechart grafiklerini çizme
        if plot:
            # Bar plot oluşturma
            bar_fig = go.Figure()
            bar_fig.add_trace(go.Bar(
                y=value_counts.index,
                x=value_counts.values,
                orientation='h',
                text=ratio.round(2).astype(str) + '%',  # Oranları etiket olarak ekliyoruz
                textposition='inside',
                marker=dict(color='rgba(58, 71, 80, 0.6)', line=dict(color='rgba(58, 71, 80, 1)', width=1)),
            ))
            bar_fig.update_layout(
                title=f'{col_name} - Bar Plot',
                xaxis_title='Count',
                yaxis_title=col_name,
                template='plotly_white'
            )
            bar_fig.show()

            # Pie chart oluşturma
            pie_fig = px.pie(
                values=value_counts.values,
                names=value_counts.index,
                title=f'{col_name} - Pie Chart',
                labels={col_name: 'Categories'},
                hole=0.3  # Ortasında boşluk olan bir pie chart
            )
            pie_fig.update_traces(textinfo='percent+label',
                                  pull=[0.05] * len(value_counts))  # Daha interaktif ve okunabilir
            pie_fig.show()

cat_summary(df)

################################ Nümerik Degiskenlerin analizi ##########################################
def num_summary(dataframe, plot=False):
    num_columns = dataframe.select_dtypes(include=['number']).columns

    for col_name in num_columns:
        print(f"Summary for numeric column: {col_name}")
        print(dataframe[col_name].describe())  # Verinin istatistiksel özetini yazdırıyoruz
        print("##########################################")

        # Eğer plot=True ise, boxplot ve histogram grafiklerini çizme
        if plot:
            # Box plot oluşturma
            box_fig = go.Figure()
            box_fig.add_trace(go.Box(
                y=dataframe[col_name],
                boxmean='sd',  # Ortalama ve standart sapma değerlerini gösterebiliriz
                marker=dict(color='rgba(17, 157, 255, 0.7)', line=dict(color='rgba(17, 157, 255, 1)', width=1)),
                name=col_name
            ))
            box_fig.update_layout(
                title=f'{col_name} - Box Plot',
                yaxis_title=col_name,
                template='plotly_white'
            )
            box_fig.show()  # Grafik tarayıcıda açılacak

            # Histogram oluşturma
            hist_fig = px.histogram(dataframe, x=col_name, title=f'{col_name} - Histogram')
            hist_fig.update_traces(
                marker=dict(color='rgba(255, 87, 34, 0.7)', line=dict(color='rgba(255, 87, 34, 1)', width=1)))
            hist_fig.show()  # Grafik tarayıcıda açılacak

num_summary(df)

df.info()

df.head()

sales_customer_groupby = df.groupby(["Customer ID" ,"City", "State", "Region"]).aggregate({"Sales": ("sum", "max")}).sort_values(
    ("Sales", "sum"), ascending=False).round(2)

sales_city_groupby = df.groupby(["City","State"]).aggregate({"Sales": ("sum", "max")}).sort_values(("Sales", "sum"), ascending=False).round(2)

sales_product_groupby = df.groupby("Product Name").aggregate({"Sales": ("sum", "count")}).sort_values(("Sales", "sum"), ascending=False).round(2)

# Mevsimlerin Oluşturulması
def get_season(date):
    # Tarihlerin ayını alıyoruz
    month = date.month
    # Ay numarasına göre mevsim belirliyoruz
    if month in [12, 1, 2]:
        return 'Kış'
    elif month in [3, 4, 5]:
        return 'İlkbahar'
    elif month in [6, 7, 8]:
        return 'Yaz'
    else:
        return 'Sonbahar'

# Mevsim sütunu ekliyoruz
df['Season'] = df['Order Date'].apply(get_season)

df.head()

#df.to_excel("proje.xlsx")
df.to_csv("marketsales_data.csv")
df.to_excel("marketsales_data.xlsx")

##################################################### RFM #####################################################
def create_rfm(dataframe, csv=False):
    df["Order Date"].max()
    today_date = dt.datetime(2019, 1, 10)

    rfm = df.groupby('Customer ID').agg({'Order Date': lambda date: (today_date - date.max()).days,
                                         'Order ID': lambda order: order.nunique(),
                                         'Sales': lambda sales: sales.sum()})

    rfm.columns = ["recency", "frequency", "monetary"]

    # 5. RFM Skorlarının Hesaplanması (Calculating RFM Scores)

    rfm["recency_score"] = pd.qcut(rfm['recency'], 5, labels=[5, 4, 3, 2, 1])
    rfm["frequency_score"] = pd.qcut(rfm['frequency'].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
    rfm["monetary_score"] = pd.qcut(rfm['monetary'], 5, labels=[1, 2, 3, 4, 5])

    rfm["RFM_SCORE"] = (rfm['recency_score'].astype(str) +
                        rfm['frequency_score'].astype(str))


    # 6. RFM Segmentlerinin Oluşturulması ve Analiz Edilmesi (Creating & Analysing RFM Segments)

    # RFM isimlendirmesi
    seg_map = {
        r'[1-2][1-2]': 'hibernating',
        r'[1-2][3-4]': 'at_Risk',
        r'[1-2]5': 'cant_loose',
        r'3[1-2]': 'about_to_sleep',
        r'33': 'need_attention',
        r'[3-4][4-5]': 'loyal_customers',
        r'41': 'promising',
        r'51': 'new_customers',
        r'[4-5][2-3]': 'potential_loyalists',
        r'5[4-5]': 'champions'
    }

    rfm['segname'] = rfm['RFM_SCORE'].replace(seg_map, regex=True)

    if csv:
        rfm.to_csv("rfm.csv")

    return rfm
rfm_new = create_rfm(df, csv=True)

rfm_new.groupby("segname").aggregate({"monetary": ("mean", "sum")})

##################################################### CLTV #####################################################
# Tüm İşlemlerin Fonksiyonlaştırılması
def create_cltv_c(dataframe, profit=0.10):
    # Müşteri bazlı olarak fatura ve satış verilerinin getirilmesi
    cltv_c = df.groupby('Customer ID').agg({'Order ID': lambda x: x.nunique(),
                                            'Sales': lambda x: x.sum()})

    cltv_c.columns = ["Total_transaction", "Total_price"]

    ## 2. Ortalama Sipariş Değeri ( Average Order Value ) (average_order_value = total_price / total_transaction)
    cltv_c["average_order_value"] = cltv_c["Total_price"] / cltv_c["Total_transaction"]

    # 3. Satın Alma Sıklığı ( Purchase Frequency ) (total_transaction / total_number_of_customers)
    cltv_c["purchase_frequency"] = cltv_c["Total_transaction"] / cltv_c.shape[0]

    # 4. Repeat Rate & Churn Rate (birden fazla alışveriş yapan müşteri sayısı / tüm müşteriler)
    repeat_rate = cltv_c[cltv_c["Total_transaction"] > 1].shape[0] / cltv_c.shape[0]
    churn_rate = 1 - repeat_rate

    # 5. Profit Margin (profit margin = total_price * 0.10)
    cltv_c["profit_margin"] = cltv_c["Total_price"] * 0.10

    # 6. Customer Value  (customer_value = average_order_value * purchase_frequency)
    cltv_c["customer_value"] = cltv_c["average_order_value"] * cltv_c["purchase_frequency"]

    # 7. Customer Life Time Value (CLTV = (customer_value / churn_rate) * profit_margin)
    cltv_c["cltv"] = (cltv_c["customer_value"] / churn_rate) * cltv_c["profit_margin"]

    # 8. Segmentlerin Oluşturulması

    cltv_c["segment"] = pd.qcut(cltv_c["cltv"], 7, labels=["E", "D", "C", "B", "A-", "A", "A+"])
    return cltv_c

clv = create_cltv_c(df)

clv.head()

clv.groupby("segment").agg({"count", "mean", "sum"})