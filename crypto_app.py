import streamlit as st
import ccxt
import pandas as pd
import numpy as np
import plotly.express as px
import warnings

warnings.filterwarnings('ignore', category=RuntimeWarning)

exchange = ccxt.kucoin()

@st.cache_data(ttl=86400)  
def fetch_top_crypto():
    try:
        markets = exchange.fetch_markets()
        data = []
        for market in markets:
            if market['symbol'].endswith('/USDT') and market['active']:
                ticker = exchange.fetch_ticker(market['symbol'])
                change = ticker['percentage']
                data.append((market['symbol'], ticker['last'], change))

        df = pd.DataFrame(data, columns=['Symbol', 'Price', 'Change'])
        df = df.sort_values(by='Change', ascending=False).head(5)
        return df

    except Exception as e:
        st.error(f"Terjadi kesalahan saat mengambil data: {e}")
        return pd.DataFrame(columns=['Symbol', 'Price', 'Change'])

def visualize_top_crypto(df):
    try:
        fig = px.bar(
            df, x='Symbol', y='Change', 
            title='Top 5 Cryptocurrency Berdasarkan Kenaikan Harga',
            labels={'Change': 'Persentase Kenaikan (%)', 'Symbol': 'Pasangan Mata Uang'},
            color='Change',
            color_continuous_scale='Viridis')
        st.plotly_chart(fig)
    except Exception as e:
        st.error(f"Terjadi kesalahan saat membuat visualisasi: {e}")

def calculate_sma(df, window):
    return df['close'].rolling(window=window).mean()

def calculate_ema(df, window):
    return df['close'].ewm(span=window, adjust=False).mean()

def calculate_rsi(df, window=14):
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

class HMM:
    def __init__(self, n_states, n_iter=50):
        self.n_states = n_states
        self.n_iter = n_iter
        self.transition_matrix = None
        self.means = None
        self.variances = None
        self.initial_probs = None

    def initialize_params(self, data):
        self.transition_matrix = np.ones((self.n_states, self.n_states)) / self.n_states
        self.means = np.linspace(data.min(), data.max(), self.n_states)
        self.variances = np.ones(self.n_states) * np.var(data)
        self.initial_probs = np.ones(self.n_states) / self.n_states

    def gaussian_prob(self, x, mean, var):
        return (1 / np.sqrt(2 * np.pi * var)) * np.exp(-(x - mean)**2 / (2 * var))

    def e_step(self, data):
        T = len(data)
        alpha = np.zeros((T, self.n_states))
        beta = np.zeros((T, self.n_states))
        gamma = np.zeros((T, self.n_states))

        for t in range(T):
            for j in range(self.n_states):
                if t == 0:
                    alpha[t, j] = self.initial_probs[j] * self.gaussian_prob(data[t], self.means[j], self.variances[j])
                else:
                    alpha[t, j] = self.gaussian_prob(data[t], self.means[j], self.variances[j]) * \
                                  np.sum(alpha[t-1, :] * self.transition_matrix[:, j])
            alpha[t, :] /= np.sum(alpha[t, :])

        for t in range(T-2, -1, -1):
            for i in range(self.n_states):
                beta[t, i] = np.sum(beta[t+1, :] * self.transition_matrix[i, :] *
                                    [self.gaussian_prob(data[t+1], self.means[k], self.variances[k])
                                     for k in range(self.n_states)])
            beta[t, :] /= np.sum(beta[t, :])

        gamma = alpha * beta
        gamma /= gamma.sum(axis=1, keepdims=True)

        return gamma

    def m_step(self, data, gamma):
        self.means = np.sum(gamma * data[:, np.newaxis], axis=0) / np.sum(gamma, axis=0)
        self.variances = np.sum(gamma * (data[:, np.newaxis] - self.means)**2, axis=0) / np.sum(gamma, axis=0)
        self.transition_matrix = np.dot(gamma[:-1].T, gamma[1:]) / np.sum(gamma[:-1], axis=0)[:, np.newaxis]

    def fit(self, data):
        self.initialize_params(data)
        for epoch in range(self.n_iter):
            gamma = self.e_step(data)
            self.m_step(data, gamma)

    def predict_states(self, data):
        gamma = self.e_step(data)
        return np.argmax(gamma, axis=1)

    def predict_next_state(self, current_state):
        return np.argmax(self.transition_matrix[current_state, :])

def analyze(symbol, n_days):
    try:
        status_placeholder = st.empty()  

        status_placeholder.text('Mengunduh data...')

        ohlcv = exchange.fetch_ohlcv(symbol, timeframe='1h', limit=2000)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

        status_placeholder.text('Menghitung indikator teknikal...')

        df['SMA_50'] = calculate_sma(df, 50)
        df['EMA_50'] = calculate_ema(df, 50)
        df['RSI_14'] = calculate_rsi(df, 14)

        min_val = df['close'].min()
        max_val = df['close'].max()
        scaled_data = (df['close'] - min_val) / (max_val - min_val)

        train_data = scaled_data.values

        status_placeholder.text('Melatih model HMM...')

        hmm = HMM(n_states=12, n_iter=50)
        hmm.fit(train_data)
        train_states = hmm.predict_states(train_data)

        predicted_prices = []
        state_means = []
        for i in range(hmm.n_states):
            state_data = train_data[train_states == i]
            state_means.append(state_data.mean() if len(state_data) > 0 else np.mean(train_data))

        for i in range(len(train_data)):
            predicted_state = hmm.predict_next_state(train_states[i])
            predicted_prices.append(state_means[predicted_state])

        for i in range(1, len(train_data)):
            predicted_prices[i] = (train_data[i-1] + state_means[train_states[i]]) / 2

        predicted_prices_denorm = [(price * (max_val - min_val)) + min_val for price in predicted_prices]

        status_placeholder.text('Prediksi harga ke depan...')

        future_prices = predicted_prices_denorm[-n_days:]
        future_dates = pd.date_range(start=df['timestamp'].iloc[-1], periods=n_days + 1, freq='D')

        st.subheader(f"Prediksi Harga {n_days} Hari Ke Depan untuk {symbol}")

        fig = px.line(
            df,
            x='timestamp',
            y='close',
            title=f'Harga Aktual dan Prediksi {symbol}',
            labels={'timestamp': 'Tanggal', 'close': 'Harga'}
        )
        fig.add_scatter(
            x=future_dates,
            y=[predicted_prices_denorm[-1]] + future_prices[:n_days],
            mode='lines+markers',
            name='Prediksi Ke Depan',
            line=dict(dash='dash', color='red')
        )

        st.plotly_chart(fig)

        return f"Analisis selesai untuk {symbol}"

    except Exception as e:
        st.error(f"Terjadi kesalahan: {e}")

def prediction_page():
    symbols = [market['symbol'] for market in exchange.fetch_markets() if market['active'] and market['symbol'].endswith('/USDT')]

    st.sidebar.title("Pengaturan")
    symbol = st.sidebar.selectbox("Pilih pasangan mata uang", symbols)
    n_days = st.sidebar.slider("Pilih jumlah hari untuk prediksi", min_value=1, max_value=10, value=10)

    if st.button("Mulai Analisis"):
        result = analyze(symbol, n_days)
        st.success(result)

def main():
    st.set_page_config(page_title="Crypto Predictor", layout="wide")

    menu = ["Landing Page", "Prediksi Koin"]
    choice = st.sidebar.selectbox("Pilih Halaman", menu)

    if choice == "Landing Page":
        st.title("Selamat Datang di Crypto Predictor")
        st.markdown("**Aplikasi untuk menganalisis dan memprediksi harga cryptocurrency.**")

        st.subheader("Top 5 Cryptocurrency yang Sedang Bagus")
        top_crypto = fetch_top_crypto()
        if not top_crypto.empty:
            visualize_top_crypto(top_crypto)

    elif choice == "Prediksi Koin":
        st.title("Halaman Prediksi Koin")
        prediction_page()

if __name__ == "__main__":
    main()
