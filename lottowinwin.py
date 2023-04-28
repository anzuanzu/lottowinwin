import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler
import requests
from bs4 import BeautifulSoup

def fetch_lotto_data(url):
    res = requests.get(url)
    soup = BeautifulSoup(res.text, 'html.parser')

    result = []

    for td in soup.find_all('td'):
        if '開出順序' in td.text:
            for num in td.find_next_siblings('td'):
                span = num.find('span')
                if span is not None:
                    result.append(int(span.text))

    result = [result[i:i + 7] for i in range(0, len(result), 7)]

    return result

def create_lstm_model():
    model = Sequential()
    model.add(LSTM(128, activation='relu', input_shape=(1, 7), return_sequences=True))
    model.add(LSTM(64, activation='relu'))
    model.add(Dense(7, activation='linear'))
    model.compile(loss='mse', optimizer='adam')
    return model

def preprocess_data(data):
    X_train = data[:-1]
    y_train = data[1:]

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    y_train = scaler.fit_transform(y_train)

    X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])

    return X_train, y_train, scaler

def predict_next_periods(model, data, scaler, periods=5):
    predictions = []
    X_input = data[1:]

    for _ in range(periods):
        X_scaled = scaler.transform(X_input)
        X_scaled = X_scaled[-1].reshape(1, 1, 7)
        y_pred = model.predict(X_scaled)
        y_pred = scaler.inverse_transform(y_pred).round(0).astype(int)
        y_pred = np.clip(y_pred, 1, 49)
        while True:
            if len(np.unique(y_pred)) == 7:
                break
            y_pred[np.where(np.diff(np.sort(y_pred)) == 0)] = np.random.randint(1, 50, 1)
        predictions.append(y_pred)
        X_input = np.vstack((X_input, y_pred))

    return predictions

# Fetch recent lotto data
url = 'https://www.taiwanlottery.com.tw/lotto/lotto649/history.aspx'
recent_data = fetch_lotto_data(url)
data = np.array(recent_data)

# Preprocess data
X_train, y_train, scaler = preprocess_data(data)

# Train model
model = create_lstm_model()
model.fit(X_train, y_train, epochs=1000, verbose=0)

# Predict next periods
predictions = predict_next_periods(model, data, scaler, periods=5)

# 1. Show the frequency of numbers in recent 10 draws
st.title("近10期大樂透號碼出現次數")
recent_numbers = np.array(recent_data)[-10:].flatten()
count = np.bincount(recent_numbers)
count_df = pd.DataFrame({"號碼": range(1, 50), "次數": count[1:]})
count_df = count_df.sort_values("次數", ascending=False)
st.write(count_df)

# 2. Show the numbers not drawn in recent 10 draws
st.title("近10期未出現的號碼")
not_drawn_numbers = np.setdiff1d(np.arange(1, 50), np.unique(recent_numbers))
st.write(not_drawn_numbers)

# 3. Show the predicted numbers
st.title("預測結果")
for i, pred in enumerate(predictions, 1):
    st.write(f"Y{i}: {pred.flatten()}")

st.write("注意：以上預測結果僅供參考，實際開獎結果可能有所不同。")

