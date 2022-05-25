# Import Packages

# data processing, CSV file I/O (e.g. pd.read_csv), data manipulation seperti SQL
import pandas as pd
from sklearn.preprocessing import MinMaxScaler  # normalisasi data

# Deep-learing:
from keras.models import Sequential
from keras.layers import LSTM, GRU, Dense

# Import data suhu udara
url = 'https://drive.google.com/file/d/1uXSxH1m-AangHyehR_Xl-11QXgH-WgeH/view?usp=sharing'
url = 'https://drive.google.com/uc?id=' + url.split('/')[-2]

df = pd.read_csv(url, sep=',', parse_dates=['Tanggal'],
                 infer_datetime_format=True, low_memory=False, na_values=['nan', '?'],
                 index_col='Tanggal')

# Menemukan semua kolom yang memiliki nilai NaN
droping_list_all = []
for j in range(0, 1):
    if not df.iloc[:, j].notnull().all():
        droping_list_all.append(j)
droping_list_all

# Mengisi NaN dengan nilai mean
for j in range(0, 1):
    df.iloc[:, j] = df.iloc[:, j].fillna(df.iloc[:, j].mean())


def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    dff = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(dff.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # prediksi sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(dff.shift(-i))
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
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(df)

# frame sebagai supervised learning
reframed = series_to_supervised(scaled, 1, 1)

# Membagi data menjadi data train dan data test
values = reframed.values

n_train_time = len(values) * 0.8  # jumlah % data
n_train_time = round(n_train_time)
train = values[:n_train_time, :]
test = values[n_train_time:, :]

# Membagi data menjadi input dan outputs
train_X, train_y = train[:, :-1], train[:, -1]
test_X, test_y = test[:, :-1], test[:, -1]

# reshape input menjadi 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))

# Membangun model hybrid GRU-LSTM
model = Sequential()
# menggunakan 20 neurons GRU untuk layer pertama
model.add(GRU(20, input_shape=(
    train_X.shape[1], train_X.shape[2]), return_sequences=True))
model.add(LSTM(256))  # Menggunakan 256 neuron LSTM
model.add(Dense(64))  # Menggunakan 64 neuron Dense Layer
model.add(Dense(1))  # Menggunakan 1 neuron Dense Layer
# Menggunakan MSE sebagai loss function dan Adam sebagai optimizer
model.compile(loss='mean_squared_error', optimizer='adam')

# Fit network
# Menggunakan 100 training epochs dengan ukuran batch size sebesar 50
history = model.fit(train_X, train_y, epochs=100, batch_size=50,
                    validation_data=(test_X, test_y), verbose=2, shuffle=False)
model.save('temp_predict_model.h5')
