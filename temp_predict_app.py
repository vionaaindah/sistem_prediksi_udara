from winreg import KEY_ENUMERATE_SUB_KEYS
import streamlit as st
import numpy as np  # linear algebra
# data processing, CSV file I/O (e.g. pd.read_csv), data manipulation seperti SQL
import pandas as pd
from sklearn.preprocessing import MinMaxScaler  # normalisasi data
# check error dan akurasi model
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Deep-learing:
from keras.models import load_model

#---------------------------------#
# Page layout
# Page expands to full width
st.set_page_config(page_title='Sistem Prediksi Suhu Udara',
                   layout='wide')

#---------------------------------#
# Model building


def build_model(df_new):
    # Menemukan semua kolom yang memiliki nilai NaN
    droping_list_all = []
    for j in range(0, 1):
        if not df_new.iloc[:, j].notnull().all():
            droping_list_all.append(j)
    droping_list_all

    # Mengisi NaN dengan nilai mean
    for j in range(0, 1):
        df_new.iloc[:, j] = df_new.iloc[:, j].fillna(df_new.iloc[:, j].mean())

    def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
        n_vars = 1 if type(data) is list else data.shape[1]
        df_newf = pd.DataFrame(data)
        cols, names = list(), list()
        # input sequence (t-n, ... t-1)
        for i in range(n_in, 0, -1):
            cols.append(df_newf.shift(i))
            names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
        # prediksi sequence (t, t+1, ... t+n)
        for i in range(0, n_out):
            cols.append(df_newf.shift(-i))
            if i == 0:
                names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
            else:
                names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
        # Menggabungkan semua
        agg = pd.concat(cols, axis=1)
        agg.columns = names
        # drop rows dengan NaN values
        if dropnan:
            agg.dropna(inplace=True)
        return agg

    # normalisasi data
    scaler_new = MinMaxScaler(feature_range=(0, 1))
    scaled_new = scaler_new.fit_transform(df_new)

    # frame sebagai supervised learning
    reframed = series_to_supervised(scaled_new, 1, 1)

    # Membagi data menjadi data train dan data test
    values = reframed.values

    n_train_time = len(values) - 1
    n_train_time = round(n_train_time)
    train = values[:n_train_time, :]
    test = values[n_train_time:, :]

    # Membagi data menjadi input dan outputs
    train_X, train_y = train[:, :-1], train[:, -1]
    test_X, test_y = test[:, :-1], test[:, -1]

    # reshape input menjadi 3D [samples, timesteps, features]
    train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
    test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))

    # menggunakan model yang telah dibangu sebelumnya
    model = load_model('temp_predict_model.h5')

    model.fit(train_X, train_y, epochs=100, batch_size=50,
              validation_data=(test_X, test_y), verbose=2, shuffle=False)

    # membuat prediksi
    yhat = model.predict(test_X)
    test_X = test_X.reshape((test_X.shape[0], 1))

    # invert scaling untuk prediksi
    inv_yhat = np.concatenate((yhat, test_X[:, 0:]), axis=1)
    inv_yhat = scaler_new.inverse_transform(inv_yhat)
    inv_yhat = inv_yhat[:, 0]

    hasil = []
    df2 = pd.DataFrame({"Tavg": [inv_yhat[0]]})
    df_new = df_new.append(df2, ignore_index=True)

    for i in range(7):
        # normalisasi data
        scaler_new = MinMaxScaler(feature_range=(0, 1))
        scaled_new = scaler_new.fit_transform(df_new)

        # frame sebagai supervised learning
        reframed = series_to_supervised(scaled_new, 1, 1)

        # Membagi data menjadi data train dan data test
        values = reframed.values

        n_train_time = len(values) - 1
        n_train_time = round(n_train_time)
        train = values[:n_train_time, :]
        test = values[n_train_time:, :]

        # Membagi data menjadi input dan outputs
        train_X, train_y = train[:, :-1], train[:, -1]
        test_X, test_y = test[:, :-1], test[:, -1]

        # reshape input menjadi 3D [samples, timesteps, features]
        train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
        test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))

        model.fit(train_X, train_y, epochs=100, batch_size=50,
                  validation_data=(test_X, test_y), verbose=2, shuffle=False)

        # membuat prediksi
        yhat = model.predict(test_X)
        test_X = test_X.reshape((test_X.shape[0], 1))

        # invert scaling untuk prediksi
        inv_yhat = np.concatenate((yhat, test_X[:, 0:]), axis=1)
        inv_yhat = scaler_new.inverse_transform(inv_yhat)
        inv_yhat = inv_yhat[:, 0]

        hasil.append(inv_yhat[0])

        # menambah hasil prediksi untuk menjadi data latih
        df_new = pd.DataFrame(
            np.insert(df_new.values, -1, values=[inv_yhat[0]], axis=0))

    st.subheader('2. Hasil Prediksi')
    st.write('Suhu Udara Hari Pertama (°C):')
    st.info('%.2f' % hasil[0])
    st.write('Suhu Udara Hari Kedua (°C):')
    st.info('%.2f' % hasil[1])
    st.write('Suhu Udara Hari Ketiga (°C):')
    st.info('%.2f' % hasil[2])
    st.write('Suhu Udara Hari Keempat (°C):')
    st.info('%.2f' % hasil[3])
    st.write('Suhu Udara Hari Kelima (°C):')
    st.info('%.2f' % hasil[4])
    st.write('Suhu Udara Hari Keenam (°C):')
    st.info('%.2f' % hasil[5])
    st.write('Suhu Udara Hari Ketujuh (°C):')
    st.info('%.2f' % hasil[6])


#---------------------------------#
st.write("""
# Sistem Prediksi Suhu Udara
Dengan memasukkan data suhu udara harian anda bisa mendapatkan prediksi suhu udara selama tujuh hari kedepan
""")

#---------------------------------#
# Sidebar - Collects user input features into dataframe
with st.sidebar.header('1. Pilih file CSV anda:'):
    uploaded_file = st.sidebar.file_uploader(
        "Upload input file CSV anda", type=["csv"])
    st.sidebar.markdown("""
[Example CSV input file](https://github.com/vionaaindah/sistem_prediksi_udara/blob/main/suhu_udara_2022.csv)
""")

#---------------------------------#
# Main panel

    # Displays the dataset
st.subheader('1. Dataset')

if uploaded_file is not None:
    df_new = pd.read_csv(uploaded_file, sep=',', parse_dates=['Tanggal'],
                         infer_datetime_format=True, low_memory=False, na_values=['nan', '?'],
                         index_col='Tanggal')
    st.markdown('**1.1. Dataset Suhu Udara**')
    st.write(df_new.head())
    build_model(df_new)
else:
    st.info('Menunggu file CSV ter-upload.')
    if st.button('Klik disini untuk menggunakan contoh dataset'):

        url = 'https://drive.google.com/file/d/1vCHYM0FKlX_7PqikSpvNmD-iwxZpqv6d/view?usp=sharing'
        url = 'https://drive.google.com/uc?id=' + url.split('/')[-2]

        df_new = pd.read_csv(url, sep=',', parse_dates=['Tanggal'],
                             infer_datetime_format=True, low_memory=False, na_values=['nan', '?'],
                             index_col='Tanggal')

        st.markdown('Dataset Suhu Udara 2022 digunakan sebagai contoh dataset.')
        st.write(df_new.head(5))

        build_model(df_new)