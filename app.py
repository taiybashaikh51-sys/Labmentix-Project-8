import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import joblib
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Tesla Stock Price Prediction",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stApp { background-color: #0B1220; color: #ffffff; }
    .main-header {
        background: linear-gradient(135deg, #1A2333, #0B1220);
        border: 1px solid rgba(239,68,68,0.3);
        border-radius: 16px;
        padding: 2rem;
        margin-bottom: 2rem;
        text-align: center;
    }
    .section-title {
        color: #EF4444;
        font-size: 1.3rem;
        font-weight: 700;
        margin: 1.5rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid rgba(239,68,68,0.3);
    }
    div[data-testid="stSidebar"] {
        background-color: #111827;
        border-right: 1px solid rgba(239,68,68,0.2);
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    df = pd.read_csv('TSLA.csv')
    df.columns = [col.strip() for col in df.columns]
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    # Technical indicators
    df['MA_20'] = df['Close'].rolling(window=20).mean()
    df['MA_50'] = df['Close'].rolling(window=50).mean()
    df['MA_200'] = df['Close'].rolling(window=200).mean()
    df['Returns'] = df['Close'].pct_change() * 100
    df['Volatility'] = df['Returns'].rolling(window=20).std()
    df['Price_Range'] = df['High'] - df['Low']
    df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
    # RSI
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    return df

@st.cache_resource
def load_model():
    try:
        model = joblib.load('xgboost_model.pkl')
        return model, 'XGBoost'
    except:
        try:
            model = joblib.load('xgboost_model.joblib')
            return model, 'XGBoost'
        except:
            return None, None

with st.spinner("Loading Tesla stock data..."):
    df = load_data()

model, model_name = load_model()

with st.sidebar:
    st.markdown("### ⚡ Tesla Stock")
    st.markdown("---")
    page = st.radio("📊 Select Page", [
        "🏠 Overview",
        "📈 Price Analysis",
        "🤖 ML Prediction",
        "📊 Technical Analysis",
        "📋 Data Explorer"
    ])
    st.markdown("---")
    st.metric("📅 Data Period", f"{df['Date'].dt.year.min()} - {df['Date'].dt.year.max()}")
    st.metric("📊 Total Records", f"{len(df):,}")
    if model_name:
        st.metric("🤖 Model", model_name)

st.markdown("""
<div class="main-header">
    <h1 style="color:#EF4444; margin:0; font-size:2rem;">⚡ Tesla Stock Price Prediction</h1>
    <p style="color:#94A3B8; margin:0.5rem 0 0 0;">XGBoost + Time Series Analysis | Financial Forecasting</p>
</div>
""", unsafe_allow_html=True)

col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("💰 Latest Close", f"${df['Close'].iloc[-1]:.2f}")
with col2:
    st.metric("📈 All-Time High", f"${df['High'].max():.2f}")
with col3:
    st.metric("📉 All-Time Low", f"${df['Low'].min():.2f}")
with col4:
    avg_return = df['Returns'].mean()
    st.metric("📊 Avg Daily Return", f"{avg_return:.2f}%")
with col5:
    volatility = df['Returns'].std()
    st.metric("⚡ Volatility", f"{volatility:.2f}%")

st.markdown("---")

if page == "🏠 Overview":
    st.markdown('<p class="section-title">📈 Stock Price History</p>', unsafe_allow_html=True)

    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df['Date'], open=df['Open'], high=df['High'],
        low=df['Low'], close=df['Close'], name='OHLC',
        increasing_line_color='#22C55E',
        decreasing_line_color='#EF4444'
    ))
    fig.update_layout(
        paper_bgcolor='#1A2333', plot_bgcolor='#1A2333',
        font=dict(color='#94A3B8'), height=450,
        xaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
        yaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
        xaxis_rangeslider_visible=True
    )
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=df['Date'], y=df['Close'],
            name='Close', line=dict(color='#EF4444', width=2)))
        fig2.add_trace(go.Scatter(x=df['Date'], y=df['MA_20'],
            name='MA 20', line=dict(color='#3B82F6', width=1.5, dash='dot')))
        fig2.add_trace(go.Scatter(x=df['Date'], y=df['MA_50'],
            name='MA 50', line=dict(color='#22C55E', width=1.5, dash='dot')))
        fig2.update_layout(paper_bgcolor='#1A2333', plot_bgcolor='#1A2333',
            font=dict(color='#94A3B8'), title='Close Price with Moving Averages',
            xaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
            yaxis=dict(gridcolor='rgba(255,255,255,0.05)'))
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        fig3 = go.Figure()
        fig3.add_trace(go.Bar(
            x=df['Date'], y=df['Returns'],
            marker_color=df['Returns'].apply(lambda x: '#22C55E' if x > 0 else '#EF4444'),
            name='Daily Returns'))
        fig3.update_layout(paper_bgcolor='#1A2333', plot_bgcolor='#1A2333',
            font=dict(color='#94A3B8'), title='Daily Returns (%)',
            xaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
            yaxis=dict(gridcolor='rgba(255,255,255,0.05)'))
        st.plotly_chart(fig3, use_container_width=True)

elif page == "📈 Price Analysis":
    st.markdown('<p class="section-title">📈 Price Deep Dive</p>', unsafe_allow_html=True)

    yearly = df.groupby(df['Date'].dt.year).agg({
        'Open': 'first', 'Close': 'last',
        'High': 'max', 'Low': 'min', 'Volume': 'mean'
    }).reset_index()
    yearly.columns = ['Year', 'Open', 'Close', 'High', 'Low', 'Avg Volume']

    fig = px.bar(yearly, x='Year', y=['High', 'Low'],
        barmode='group', title='Yearly High & Low Prices',
        color_discrete_map={'High': '#22C55E', 'Low': '#EF4444'})
    fig.update_layout(paper_bgcolor='#1A2333', plot_bgcolor='#1A2333',
        font=dict(color='#94A3B8'),
        xaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
        yaxis=dict(gridcolor='rgba(255,255,255,0.05)'))
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        fig2 = px.histogram(df.dropna(), x='Returns', nbins=50,
            title='Return Distribution',
            color_discrete_sequence=['#EF4444'])
        fig2.update_layout(paper_bgcolor='#1A2333', plot_bgcolor='#1A2333',
            font=dict(color='#94A3B8'),
            xaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
            yaxis=dict(gridcolor='rgba(255,255,255,0.05)'))
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        vol_fig = go.Figure()
        vol_fig.add_trace(go.Bar(x=df['Date'], y=df['Volume'],
            marker_color='rgba(239,68,68,0.5)', name='Volume'))
        vol_fig.add_trace(go.Scatter(x=df['Date'], y=df['Volume_MA'],
            line=dict(color='#FBB024', width=2), name='20-day MA'))
        vol_fig.update_layout(paper_bgcolor='#1A2333', plot_bgcolor='#1A2333',
            font=dict(color='#94A3B8'), title='Volume Analysis',
            xaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
            yaxis=dict(gridcolor='rgba(255,255,255,0.05)'))
        st.plotly_chart(vol_fig, use_container_width=True)

elif page == "🤖 ML Prediction":
    st.markdown('<p class="section-title">🤖 Stock Price Prediction</p>', unsafe_allow_html=True)

    df_ml = df.dropna().copy()
    df_ml['Month'] = df_ml['Date'].dt.month
    df_ml['Year_Num'] = df_ml['Date'].dt.year
    df_ml['Day'] = df_ml['Date'].dt.day
    df_ml['DayOfWeek'] = df_ml['Date'].dt.dayofweek

    features = ['Open', 'High', 'Low', 'Volume', 'MA_20', 'MA_50',
                'Returns', 'Volatility', 'RSI', 'Price_Range',
                'Month', 'Year_Num', 'Day', 'DayOfWeek']

    X = df_ml[features].fillna(0)
    y = df_ml['Close']

    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    if model is not None:
        try:
            y_pred = model.predict(X_test)
            st.success(f"✅ Using pre-trained {model_name} model!")
        except:
            rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            rf.fit(X_train, y_train)
            y_pred = rf.predict(X_test)
            model_name = 'Random Forest'
    else:
        with st.spinner("Training model..."):
            rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            rf.fit(X_train, y_train)
            y_pred = rf.predict(X_test)
            model_name = 'Random Forest'

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mse)

    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("📊 R² Score", f"{r2:.4f}")
    with col2: st.metric("📉 MAE", f"${mae:.2f}")
    with col3: st.metric("📉 RMSE", f"${rmse:.2f}")
    with col4: st.metric("✅ Accuracy", f"{r2*100:.1f}%")

    fig = go.Figure()
    dates_test = df_ml['Date'].iloc[split:]
    fig.add_trace(go.Scatter(x=dates_test, y=y_test,
        name='Actual', line=dict(color='#EF4444', width=2)))
    fig.add_trace(go.Scatter(x=dates_test, y=y_pred,
        name='Predicted', line=dict(color='#3B82F6', width=2, dash='dot')))
    fig.update_layout(paper_bgcolor='#1A2333', plot_bgcolor='#1A2333',
        font=dict(color='#94A3B8'), title='Actual vs Predicted Stock Price',
        xaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
        yaxis=dict(gridcolor='rgba(255,255,255,0.05)'))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<p class="section-title">🎯 Predict Future Price</p>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        open_p = st.number_input("📈 Open ($)", 0.0, 5000.0, float(df['Open'].iloc[-1]))
        high_p = st.number_input("📈 High ($)", 0.0, 5000.0, float(df['High'].iloc[-1]))
        low_p = st.number_input("📉 Low ($)", 0.0, 5000.0, float(df['Low'].iloc[-1]))
    with col2:
        volume = st.number_input("📊 Volume", 0, 500000000, int(df['Volume'].iloc[-1]))
        ma20 = st.number_input("MA 20", 0.0, 5000.0, float(df['MA_20'].dropna().iloc[-1]))
        ma50 = st.number_input("MA 50", 0.0, 5000.0, float(df['MA_50'].dropna().iloc[-1]))
    with col3:
        month = st.slider("Month", 1, 12, 6)
        year = st.slider("Year", 2020, 2030, 2025)
        day = st.slider("Day", 1, 31, 15)

    price_range = high_p - low_p
    input_data = [[open_p, high_p, low_p, volume, ma20, ma50,
                   0.5, 2.0, 50.0, price_range, month, year, day, 1]]

    if model is not None:
        try:
            pred_price = model.predict(input_data)[0]
        except:
            pred_price = (open_p + high_p + low_p) / 3
    else:
        pred_price = rf.predict(input_data)[0]

    st.markdown(f"""
    <div style="background:#1A2333; border:2px solid #EF4444; border-radius:16px;
    padding:2rem; text-align:center; margin-top:1rem;">
        <h2 style="color:#EF4444; margin:0;">⚡ Predicted Close Price: ${pred_price:.2f}</h2>
        <p style="color:#94A3B8; margin:0.5rem 0 0 0;">Based on {model_name} Model</p>
    </div>
    """, unsafe_allow_html=True)

elif page == "📊 Technical Analysis":
    st.markdown('<p class="section-title">📊 Technical Indicators</p>', unsafe_allow_html=True)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'],
        name='Close', line=dict(color='#EF4444', width=2)))
    fig.add_trace(go.Scatter(x=df['Date'], y=df['MA_20'],
        name='MA 20', line=dict(color='#3B82F6', width=1.5)))
    fig.add_trace(go.Scatter(x=df['Date'], y=df['MA_50'],
        name='MA 50', line=dict(color='#22C55E', width=1.5)))
    fig.add_trace(go.Scatter(x=df['Date'], y=df['MA_200'],
        name='MA 200', line=dict(color='#F97316', width=1.5)))
    fig.update_layout(paper_bgcolor='#1A2333', plot_bgcolor='#1A2333',
        font=dict(color='#94A3B8'), height=400, title='Moving Averages',
        xaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
        yaxis=dict(gridcolor='rgba(255,255,255,0.05)'))
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=df['Date'], y=df['RSI'],
            line=dict(color='#8B5CF6', width=2), name='RSI'))
        fig2.add_hline(y=70, line_dash='dash', line_color='#EF4444',
            annotation_text='Overbought (70)')
        fig2.add_hline(y=30, line_dash='dash', line_color='#22C55E',
            annotation_text='Oversold (30)')
        fig2.update_layout(paper_bgcolor='#1A2333', plot_bgcolor='#1A2333',
            font=dict(color='#94A3B8'), title='RSI Indicator',
            xaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
            yaxis=dict(gridcolor='rgba(255,255,255,0.05)', range=[0, 100]))
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=df['Date'], y=df['Volatility'],
            fill='tozeroy', line=dict(color='#F97316', width=2),
            fillcolor='rgba(249,115,22,0.1)', name='Volatility'))
        fig3.update_layout(paper_bgcolor='#1A2333', plot_bgcolor='#1A2333',
            font=dict(color='#94A3B8'), title='Rolling Volatility (20-day)',
            xaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
            yaxis=dict(gridcolor='rgba(255,255,255,0.05)'))
        st.plotly_chart(fig3, use_container_width=True)

elif page == "📋 Data Explorer":
    st.markdown('<p class="section-title">📋 Data Explorer</p>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        year_filter = st.multiselect("📅 Year",
            options=sorted(df['Date'].dt.year.unique()),
            default=sorted(df['Date'].dt.year.unique())[-3:])
    with col2:
        show_cols = st.multiselect("📊 Columns",
            options=['Date','Open','High','Low','Close','Volume',
                     'Returns','MA_20','MA_50','RSI','Volatility'],
            default=['Date','Open','High','Low','Close','Volume','Returns'])

    filtered = df[df['Date'].dt.year.isin(year_filter)][show_cols]
    st.dataframe(filtered, use_container_width=True, height=400)
    csv = filtered.to_csv(index=False)
    st.download_button("📥 Download Data", csv, "tesla_stock.csv", "text/csv")

st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#94A3B8; font-size:0.8rem; padding:1rem;">
    ⚡ Tesla Stock Prediction | Built by <strong style="color:#EF4444">Taiyba Shaikh</strong> | ML Project
</div>
""", unsafe_allow_html=True)