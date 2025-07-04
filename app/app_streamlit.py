import streamlit as st
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# ---------- Load Historical Data ----------
@st.cache_data
def load_data():
    data = yf.download('KO', start='2015-01-01', end='2023-12-31')
    # Flatten multi-level columns if needed
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = ['_'.join(col).strip() for col in data.columns.values]
    # Rename columns
    data.rename(columns={
    'Close_KO': 'Close',
    'Open_KO': 'Open',
    'High_KO': 'High',
    'Low_KO': 'Low',
    'Volume_KO': 'Volume'
    }, inplace=True)
   

    data.reset_index(inplace=True)
    data['MA_20'] = data['Close'].rolling(window=20).mean()
    data['MA_50'] = data['Close'].rolling(window=50).mean()
    data['Daily_Return'] = data['Close'].pct_change()
    data['Volatility'] = data['Daily_Return'].rolling(window=20).std()
    data.fillna(0, inplace=True)
    return data

data = load_data()

# ---------- Train Model ----------
features = ['Open', 'High', 'Low', 'Volume', 'MA_20', 'MA_50', 'Daily_Return', 'Volatility']
target = 'Close'

X = data[features]
y = data[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ---------- Fetch Live Data for Prediction ----------
live_data = yf.download('KO', period='1d', interval='1m')
live_data.reset_index(inplace=True)

live_data['MA_20'] = live_data['Close'].rolling(window=20).mean()
live_data['MA_50'] = live_data['Close'].rolling(window=50).mean()
live_data['Daily_Return'] = live_data['Close'].pct_change()
live_data['Volatility'] = live_data['Daily_Return'].rolling(window=20).std()
live_data.fillna(0, inplace=True)

latest = live_data[features].iloc[-1:].dropna()
predicted_price = model.predict(latest)[0] if not latest.empty else "Not available"

# ---------- Streamlit UI ----------
st.title("📈 Coca-Cola Stock Price Prediction")

st.subheader("Historic Price Chart + Moving Averages")
st.line_chart(data[['Close', 'MA_20', 'MA_50']])

st.subheader("📅 Latest Live Prediction")
if isinstance(predicted_price, float):
    st.success(f"Predicted Closing Price: **${predicted_price:.2f}**")
else:
    st.warning("Live prediction is not available yet (waiting for enough data)")
