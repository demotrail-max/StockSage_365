import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np
import streamlit as st

# Static exchange rates (update these as needed)
STATIC_EXCHANGE_RATES = {
    'USD': 1.0,   # US Dollar
    'EUR': 0.93,  # Euro
    'GBP': 0.80,  # British Pound
    'INR': 82.00, # Indian Rupee
    'JPY': 135.00,# Japanese Yen
    'AUD': 1.50,  # Australian Dollar
    'CAD': 1.35,  # Canadian Dollar
    'CHF': 0.91,  # Swiss Franc
    'CNY': 7.00,  # Chinese Yuan
    'MXN': 18.00, # Mexican Peso
    'BRL': 5.00,  # Brazilian Real
    'NZD': 1.60,  # New Zealand Dollar
    'SGD': 1.35,  # Singapore Dollar
    'KRW': 1350.00,# South Korean Won
    'ZAR': 19.00, # South African Rand
    'HKD': 7.85,  # Hong Kong Dollar
    'SEK': 10.20, # Swedish Krona
    'NOK': 9.15,  # Norwegian Krone
    'DKK': 6.90,  # Danish Krone
    'ILS': 3.60,  # Israeli New Shekel
    'TRY': 27.00, # Turkish Lira
    'RUB': 80.00, # Russian Ruble
    'SAR': 3.75,  # Saudi Riyal
    'MYR': 4.70,  # Malaysian Ringgit
    'PHP': 56.00, # Philippine Peso
    'TWD': 30.00, # New Taiwan Dollar
    'PLN': 4.00,  # Polish Zloty
    'CZK': 22.50, # Czech Koruna
    'HUF': 320.00,# Hungarian Forint
    'CLP': 800.00,# Chilean Peso
    'COP': 4100.00,# Colombian Peso
    'PEN': 3.70,  # Peruvian Nuevo Sol
    'ARS': 380.00,# Argentine Peso
    'VEF': 25.00, # Venezuelan Bolívar
    'DOP': 56.00, # Dominican Peso
    'BHD': 0.38,  # Bahraini Dinar
    'KWD': 0.31,  # Kuwaiti Dinar
    'OMR': 0.39,  # Omani Rial
    'QAR': 3.64,  # Qatari Riyal
    'JOD': 0.71,  # Jordanian Dinar
    'RSD': 117.00,# Serbian Dinar
    'MAD': 10.30, # Moroccan Dirham
    'TND': 3.05,  # Tunisian Dinar
}

# Example list of popular tickers (including some Indian company tickers)
POPULAR_TICKERS = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'BRK-B', 'JPM', 'V',
    'MA', 'WMT', 'DIS', 'HD', 'KO', 'PFE', 'MRK', 'BA', 'C', 'CSCO', 'NKE', 'UNH',
    'INTC', 'T', 'ORCL', 'IBM', 'ADBE', 'CVX', 'XOM', 'MCD', 'PEP', 'ABT', 'NFLX',
    'TATAMOTORS.BO', 'RELIANCE.BO', 'HDFCBANK.BO', 'INFY.BO', 'HDFC.BO', 'ICICIBANK.BO',
    'LT.BO', 'SBIN.BO', 'HINDUNILVR.BO', 'ITC.BO', 'KOTAKBANK.BO', 'BHARTIARTL.BO' , 'ZOMATO.BO' , 'UBER',
    'LYFT', 'BYND', 'PINS', 'SHOP', 'BABA', 'TCEHY', 'NIO', 'PLTR' , 'ABCAPITAL.BO' , 'INFY.BO' , 'WIPRO.BO'
]

def fetch_data_yahoo(ticker, start_date, end_date):
    """Fetch historical stock data from Yahoo Finance."""
    try:
        data = yf.download(ticker, start=start_date, end=end_date)
        if data.empty:
            raise ValueError(f"No data found for the ticker symbol '{ticker}'. It might be delisted or incorrect.")
        return data
    except Exception as e:
        st.error(f"Error fetching data from Yahoo Finance for ticker '{ticker}': {e}")
        return None

def preprocess_data(data, lookback_period):
    """Preprocess data by creating lag features."""
    data['Date'] = pd.to_datetime(data.index)
    data['Day'] = data['Date'].dt.day
    data['Month'] = data['Date'].dt.month
    data['Year'] = data['Date'].dt.year
    
    features = pd.DataFrame()
    target = data['Close']
    
    # Create lag features
    for i in range(1, lookback_period + 1):
        features[f'Lag_{i}'] = target.shift(i)

    features = features.dropna()
    target = target[lookback_period:]  # Align target with features

    return train_test_split(features, target, test_size=0.2, shuffle=False)

def train_model(X_train, y_train):
    """Train the linear regression model."""
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def predict_future(model, X_test, lookahead_days):
    """Predict future stock prices."""
    predictions = []
    last_known_data = X_test.iloc[-1:].copy()
    
    for _ in range(lookahead_days):
        pred = model.predict(last_known_data)[0]
        predictions.append(pred)
        
        # Shift features for the next prediction
        last_known_data = last_known_data.shift(-1, axis=1)
        last_known_data.iloc[0, -1] = pred
    
    return predictions

def calculate_profit_loss(predicted_prices, current_price):
    """Calculate profit or loss based on predictions."""
    profit_loss = []
    for price in predicted_prices:
        if price > current_price:
            profit_loss.append('Profit')
        elif price < current_price:
            profit_loss.append('Loss')
        else:
            profit_loss.append('No Change')
    
    # Provide investment advice
    if all(p > current_price for p in predicted_prices):
        advice = 'Strong Buy - All future prices are higher than the current price.'
    elif any(p > current_price for p in predicted_prices):
        advice = 'Buy with Caution - Some future prices are higher than the current price.'
    else:
        advice = 'Sell - All future prices are lower than the current price.'
    
    return profit_loss, advice

def fetch_exchange_rate(base_currency, target_currency):
    """Fetch the static conversion rate from base_currency to target_currency."""
    base_rate = STATIC_EXCHANGE_RATES.get(base_currency, None)
    target_rate = STATIC_EXCHANGE_RATES.get(target_currency, None)
    
    if base_rate and target_rate:
        return target_rate / base_rate
    else:
        st.error(f"Exchange rate for {target_currency} not found.")
        return None

def stock_price_prediction(ticker, start_date, end_date, lookback_period, lookahead_days, target_currency):
    """Combine all steps to predict stock prices."""
    data = fetch_data_yahoo(ticker, start_date, end_date)
    if data is None:
        return None, None, None
    
    X_train, X_test, y_train, y_test = preprocess_data(data, lookback_period)
    
    model = train_model(X_train, y_train)
    predictions = predict_future(model, X_test, lookahead_days)
    
    current_price = data['Close'].iloc[-1]
    
    # Fetch static exchange rate for the target currency
    conversion_rate = fetch_exchange_rate('USD', target_currency)
    if conversion_rate is None:
        return None, None, None
    
    # Convert predictions to both USD and target currency
    predictions_converted = [p * conversion_rate for p in predictions]
    
    future_dates = pd.date_range(start=X_test.index[-1] + pd.Timedelta(days=1), periods=lookahead_days)
    profit_loss, advice = calculate_profit_loss(predictions, current_price)
    
    result = pd.DataFrame({
        'Date': future_dates,
        'Predicted Price (USD)': predictions,
        'Predicted Price': predictions_converted,
        'Profit/Loss': profit_loss
    })
    
    return result, conversion_rate, advice

def plot_results(results, historical_data, conversion_rate, target_currency):
    """Visualize historical and predicted stock prices in the selected currency."""
    fig, ax = plt.subplots(figsize=(14, 8))

    # Plot Historical Prices in Target Currency
    ax.plot(historical_data.index, historical_data['Close'] * conversion_rate, label=f'Historical Price ({target_currency})', color='royalblue', linestyle='-', linewidth=2)
    
    # Plot Predicted Prices in Target Currency
    ax.plot(results['Date'], results['Predicted Price'], linestyle='--', color='orange', linewidth=2, alpha=0.5, label=f'Predicted Price ({target_currency})')
    
    # Separate predictions by profit and loss
    profit_dates = results[results['Profit/Loss'] == 'Profit']['Date']
    profit_prices = results[results['Profit/Loss'] == 'Profit']['Predicted Price']

    loss_dates = results[results['Profit/Loss'] == 'Loss']['Date']
    loss_prices = results[results['Profit/Loss'] == 'Loss']['Predicted Price']

    no_change_dates = results[results['Profit/Loss'] == 'No Change']['Date']
    no_change_prices = results[results['Profit/Loss'] == 'No Change']['Predicted Price']
    
    ax.plot(profit_dates, profit_prices, linestyle='-', color='green', label='Profit')
    ax.plot(loss_dates, loss_prices, linestyle='-', color='red', label='Loss')
    ax.plot(no_change_dates, no_change_prices, linestyle='--', color='gray', label='No Change')
    
    ax.set_xlabel('Date')
    ax.set_ylabel(f'Price ({target_currency})')
    ax.set_title(f'Stock Price Prediction in {target_currency}')
    ax.legend()
    ax.grid(True)

    st.pyplot(fig)

# Streamlit app
st.title('Stock Sage')

# Input widgets
ticker = st.text_input('Enter Ticker Symbol', 'AAPL').upper()
start_date = st.date_input('Start Date', pd.to_datetime('2023-01-01'))
end_date = st.date_input('End Date', pd.to_datetime('2023-12-31'))
target_currency = st.text_input('Enter Currency Code', 'USD').upper()
lookback_period = st.number_input('Lookback Period', min_value=1, max_value=365, value=5)
lookahead_days = st.number_input('Lookahead Days', min_value=1, max_value=100, value=10)

if st.button('Run Prediction'):
    if ticker not in POPULAR_TICKERS:
        st.error(f"Ticker symbol '{ticker}' is not in the list of available tickers. Please check the symbol.")
    else:
        results, conversion_rate, advice = stock_price_prediction(ticker, start_date, end_date, lookback_period, lookahead_days, target_currency)
        if results is not None:
            st.write("Predicted Stock Prices:")
            st.dataframe(results)

            historical_data = fetch_data_yahoo(ticker, start_date, end_date)
            if historical_data is not None:
                plot_results(results, historical_data, conversion_rate, target_currency)
            else:
                st.error("Failed to fetch historical data for plotting.")
            
            st.write("Investment Advice:")
            st.write(advice)
        else:
            st.error("Failed to generate predictions.")
