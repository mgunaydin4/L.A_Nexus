import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning
warnings.simplefilter('ignore', ConvergenceWarning)

from main import sales_customer_groupby, create_rfm,sales_city_groupby,sales_product_groupby
#from sarimax_wo_aylık_validasyon import *
df = pd.read_excel("proje.xlsx")
rfm = pd.read_csv("rfm.csv")
df.drop("Unnamed: 0", axis=1, inplace=True)
df["Order Date"] = pd.to_datetime(df["Order Date"]).dt.date
df["Ship Date"] = pd.to_datetime(df["Ship Date"]).dt.date

average_sales = df["Sales"].mean()
columns_to_include = [col for col in df.columns if df[col].nunique() <= 500 or pd.api.types.is_numeric_dtype(df[col])]


# Sidebar başlık
image = Image.open("logo.png")

# Resmi Streamlit'te gösterin
st.sidebar.image(image, width=50)

# Başlık ve logo için HTML kodu
st.sidebar.markdown("<h1>LA Nexus</h1>", unsafe_allow_html=True)

# Sidebar Alt Kümeler
main_section = st.sidebar.selectbox(
    "Select Main Section:",
    ["Home", "Data", "RFM", "Prediction"]
)

if main_section == "Home":
    st.title("MarketSales Analizi: CRM ve Sarımax Modelleri ile Stratejik Kararlar")
    st.write("""
    Birbirlerini tanımayan ve farklı nedenlerle Amerika'ya göç eden 5 kişi, Los Angeles yeni hayatlarına alışmaya çalışmaktadır. İlk aylarda farklı işler yaparak hayatlarını sürdüren bu kişiler, kaderin bir cilvesiyle bir gece vakti L.A.'de bir barda karşılaşırlar. Sohbet ilerledikçe, her birinin hayalinin kendi işini kurmak olduğu ortaya çıkar.
    
    """)
    st.image("photo.png",use_container_width=True)
    st.write("""
    Aralarından biri, elinde bir veri seti bulunduğunu ve bu veriyi analiz ederek potansiyel bir e-ticaret işine dönüştürebileceklerini önerir. Bu fikir diğerlerinin de ilgisini çeker ve birlikte bu veri seti üzerinden bir iş modeli geliştirmeye karar verirler. Böylece ortak bir amaç doğrultusunda çalışmaya başlarlar: veriyi anlamlı bir şekilde analiz edip, e-ticarette başarı yakalayacak bir strateji oluşturmak.
    """)

elif main_section == "Data":
    data_subsection = st.sidebar.radio(
        "Data:",
        ["Data Information", "Data Visualization"]
    )
    if data_subsection == "Data Information":
        st.subheader("Data Information")
        st.write(df)
        st.subheader("First 5 Observations")
        st.write(df.head())
        st.subheader("Last 5 Observations")
        st.write(df.tail())
        st.subheader("Descriptive Statistics")
        st.write(df.describe())
        st.subheader("Top Spending Customers ($)")
        st.write(sales_customer_groupby)
        st.subheader("Top Spending City ($)")
        st.write(sales_city_groupby)
        st.subheader("Top Spending Product ($)")
        st.write(sales_product_groupby)

    elif data_subsection == "Data Visualization":
        st.subheader("Data - Visualization")


        if columns_to_include:
            variable = st.selectbox("Select a variable to visualize:", columns_to_include)

            # Görselleştirme
            if pd.api.types.is_numeric_dtype(df[variable]):
                plot_type = st.radio("Choose a plot type:", ["Boxplot", "Histogram"])
                if plot_type == "Boxplot":
                    fig = px.box(df, y=variable, title=f"{variable} Boxplot")
                elif plot_type == "Histogram":
                    fig = px.histogram(df, x=variable, title=f"{variable} Histogram")
            else:
                plot_type = st.radio("Choose a plot type:", ["Barplot", "Pie Chart"])
                if plot_type == "Barplot":
                    counts = df[variable].value_counts()
                    fig = px.bar(
                        counts,
                        #df[variable].value_counts().reset_index(),
                        x=counts.index,
                        y=counts.values,
                        labels={"x": variable, "y": "Count"},
                        title=f"{variable} Barplot"
                    )
                elif plot_type == "Pie Chart":
                    fig = px.pie(
                        df,
                        names=variable,
                        title=f"{variable} Pie Chart",
                        hole=0.3,
                        template="plotly_dark"
                    )

            # Grafiği Streamlit üzerinde göster
            st.plotly_chart(fig)
        else:
            st.warning("No variables with unique values <= 500 are available for visualization.")

            # Streamlit başlığı
        st.title("Dynamic Groupby Visualization")

        # Gruplama için değişkenleri seç
        group_by_columns = st.multiselect(
            "Select columns to group by:",
            options=df.columns,
            default=["Category"]
        )

        # Metrik seçimi: Toplam veya Ortalama
        metric = st.radio(
            "Select aggregation metric:",
            options=["Total", "Average"],
            index=0  # Varsayılan olarak "Total" seçili
        )

        # Eğer kullanıcı bir grup seçtiyse işlem yap
        if group_by_columns:
            # Gruplama ve metrik işlemi
            if metric == "Total":
                grouped = df.groupby(group_by_columns, as_index=False)["Sales"].sum()
            elif metric == "Average":
                grouped = df.groupby(group_by_columns, as_index=False)["Sales"].mean()

            # Bar grafiği oluştur
            fig = px.bar(
                grouped,
                x=group_by_columns[0],  # İlk grup kolonunu X ekseni olarak al
                y="Sales",
                color=group_by_columns[1] if len(group_by_columns) > 1 else None,
                # İkinci grup kolonunu renklendirme için kullan
                barmode="group" if len(group_by_columns) > 1 else None,
                title=f"Sales by {', '.join(group_by_columns)}"
            )

            # Grafiği göster
            st.plotly_chart(fig)
        else:
            st.warning("Please select at least one column to group by.")

        # Streamlit arayüzü
        st.title("Sales by Time")

        # Tarih aralığı seçimi
        start_date = st.date_input("Start Date", min_value=df["Order Date"].min(),
                                   max_value=df["Order Date"].max(), value=df["Order Date"].min())
        end_date = st.date_input("End Date", min_value=df["Order Date"].min(), max_value=df["Order Date"].max(),
                                 value=df["Order Date"].max())

        # Tarih aralığına göre veriyi filtreleme
        #filtered_data = df[
        #    (df["Order Date"] >= pd.to_datetime(start_date)) & (df["Order Date"] <= pd.to_datetime(end_date))]
        filtered_data = df[
            (df["Order Date"] >= pd.Timestamp(start_date)) &
            (df["Order Date"] <= pd.Timestamp(end_date))
            ]
        # Zaman serisi grafiği için satışları toplama
        sales_by_date = filtered_data.groupby('Order Date')['Sales'].sum().reset_index()

        # Zamanla değişen satışları görselleştirme
        fig_sales = px.line(sales_by_date, x='Order Date', y='Sales',
                            title="Total Sales by Time Range",
                            labels={'Order Date': 'Date', 'Sales': 'Total Sales ($)'})

        st.plotly_chart(fig_sales)

        # Seçilen tarih aralığı ve toplam satışları yazdırma
        # Seçilen tarih aralığındaki toplam satış
        total_sales = sales_by_date['Sales'].sum()

        # Seçilen tarih aralığındaki ortalama satış
        average_sales = sales_by_date['Sales'].mean()

        # Textbox içinde verileri göstermek
        summary_text = f"""
                        Date Range: {start_date} - {end_date}

                        Total Sales ($): {total_sales:.2f}

                        Average Sales ($): {average_sales:.2f}
                        """

        # Textbox kullanarak veriyi görüntüleme
        st.text_area("Satış Özeti", summary_text, height=200)

elif main_section == "RFM":
    st.title("RFM Analysis with CSV")
    # CSV dosyasını yükleme
    uploaded_file = st.file_uploader("CSV dosyasını yükleyin", type=["csv"])

    if uploaded_file is not None:
        # CSV dosyasını yükle
        df = pd.read_csv(uploaded_file)

        # RFM analizi yap
        rfm_new = create_rfm(df, csv=True)

        # RFM Tablosunu Streamlit'te göster
        st.subheader("RFM Tablosu")
        st.write(rfm_new)

        # Segmentlere göre dağılımı görselleştir
        st.subheader("RFM Segment Dağılımı")

        # Segmentlerin sayısını hesapla
        segment_counts = rfm_new['segname'].value_counts().reset_index()
        segment_counts.columns = ['Segment', 'Count']

        # Plotly ile bar grafiği oluştur
        fig = px.bar(segment_counts, x='Segment', y='Count',
                     title="RFM Segment Dağılımı",
                     labels={'Segment': 'RFM Segment', 'Count': 'Müşteri Sayısı'},
                     color='Segment',
                     color_discrete_sequence=px.colors.qualitative.Set1)

        # Grafiği Streamlit'te göster
        st.plotly_chart(fig)

        st.subheader("RFM Segment Harcama Dağılımı")

        # Kullanıcıdan metrik seçimini alma
        metric = st.radio("Hangi metriği görmek istersiniz?", ("Ortalama Harcama", "Toplam Harcama"))

        if metric == "Ortalama Harcama":
            # Her segment için ortalama sales değerleri
            segment_avg_sales = rfm_new.groupby('segname')['monetary'].mean().reset_index()

            fig_avg_sales = px.bar(segment_avg_sales, x='segname', y='monetary',
                                   title="RFM Segmentlerine Göre Ortalama Harcama",
                                   labels={'segname': 'RFM Segment', 'monetary': 'Ortalama Harcama'},
                                   color='segname',  # Segmentlere göre renklendirme
                                   template='plotly_dark')  # Görselleştirme için karanlık tema
            st.plotly_chart(fig_avg_sales)

        elif metric == "Toplam Harcama":
            # Her segment için toplam sales değerleri
            segment_total_sales = rfm_new.groupby('segname')['monetary'].sum().reset_index()

            fig_total_sales = px.bar(segment_total_sales, x='segname', y='monetary',
                                     title="RFM Segmentlerine Göre Toplam Harcama",
                                     labels={'segname': 'RFM Segment', 'monetary': 'Toplam Harcama'},
                                     color='segname',  # Segmentlere göre renklendirme
                                     template='plotly_dark')  # Görselleştirme için karanlık tema
            st.plotly_chart(fig_total_sales)

        st.subheader("Segment Seçimi")
        selected_segment = st.selectbox("Bir segment seçin:", rfm_new['segname'].unique())

        if selected_segment:
            # Seçilen segmentteki kullanıcıların bilgilerini filtreleme
            segment_users = rfm_new[rfm_new['segname'] == selected_segment]

            st.subheader(f"Seçilen Segment: {selected_segment}")
            st.write(f"Toplam Kullanıcı Sayısı: {segment_users.shape[0]}")

            # Kullanıcı bilgilerini göster
            st.dataframe(segment_users)

            # Excel dosyasını okuma ve ön işleme
            df = pd.read_excel("proje.xlsx")
            df.drop("Unnamed: 0", axis=1, inplace=True)
            df["Order Date"] = pd.to_datetime(df["Order Date"]).dt.date
            df["Ship Date"] = pd.to_datetime(df["Ship Date"]).dt.date

            # Segment kullanıcılarının Customer ID'lerini almak
            segment_users_ids = segment_users.index  # segment_users'deki Customer ID'leri

            # df'deki Customer ID'leri segment_users_ids ile eşleştir
            original_data = df[df['Customer ID'].isin(segment_users_ids)]

            # Filtrelenen verileri gösterme
            st.write(f"{selected_segment} segmentine ait kullanıcıların verileri:")
            st.write(original_data)



    else:
        st.info("Lütfen bir CSV dosyası yükleyin.")


elif main_section == "Prediction":
    # Başlık
    # CSV Dosyasını Yükleme
    uploaded_file = st.file_uploader("Tahmin sonuçlarını içeren CSV dosyasını yükleyin", type=["csv"])

    if uploaded_file is not None:
        # CSV dosyasını okuma
        forecast_df = pd.read_csv(uploaded_file)

        # # Tarih sütununu datetime formatına dönüştürme
        forecast_df['Tarih'] = pd.to_datetime(forecast_df['Tarih'])

        # Veriyi gösterme
        st.subheader("Prediction Results")
        st.write(forecast_df)

        # Tarih aralığı seçimi
        start_date = st.date_input("Start Date", value=forecast_df['Tarih'].min(),
                                   min_value=forecast_df['Tarih'].min(), max_value=forecast_df['Tarih'].max())
        end_date = st.date_input("End Date", value=forecast_df['Tarih'].max(),
                                 min_value=forecast_df['Tarih'].min(), max_value=forecast_df['Tarih'].max())

        # Tarih aralığına göre filtreleme
        filtered_df = forecast_df[(forecast_df['Tarih'] >= pd.to_datetime(start_date)) &
                                  (forecast_df['Tarih'] <= pd.to_datetime(end_date))]
        
        # Tahmin Satışlar Grafiği
        fig = px.line(filtered_df, x='Tarih', y='Tahmin Satışlar',
                      title="Tahmin Satışlar Zaman Serisi",
                      labels={'Tarih': 'Date', 'Tahmin Satışlar': 'Sales'})
        fig.update_traces(line=dict(dash='dash', color='blue'), mode='lines+markers')
        st.plotly_chart(fig)

        # Eğer Gerçek Satışlar sütunu varsa
        if 'Gerçek Satışlar' in forecast_df.columns:
            fig_comparison = px.line(filtered_df, x='Tarih',
                                     y=['Tahmin Satışlar', 'Gerçek Satışlar'],
                                     title="Predict and Actual Sales",
                                     labels={'value': 'Sales', 'variable': 'Date'})
            fig_comparison.update_traces(mode='lines+markers')
            st.plotly_chart(fig_comparison)
    else:
        st.info("Lütfen tahmin sonuçlarını içeren bir CSV dosyası yükleyin.")
