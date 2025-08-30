import streamlit as st
import ccxt
import pandas as pd
import numpy as np
import ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import time

# Page configuration
st.set_page_config(
    page_title="Crypto Trading Analysis Bot",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

class StreamlitTradingBot:
    def __init__(self, exchange_name: str = 'binance'):
        """Initialize the trading analysis bot for Streamlit"""
        self.exchange = getattr(ccxt, exchange_name)({
            'sandbox': False,
            'rateLimit': 1200,
            'enableRateLimit': True,
        })
    
    @st.cache_data(ttl=300)  # Cache for 5 minutes
    def get_multi_timeframe_data(_self, symbol: str) -> dict:
        """Fetch data across multiple timeframes with caching"""
        timeframes = {
            '1h': 168,   # 1 week of hourly data
            '4h': 168,   # 4 weeks of 4h data  
            '1d': 100    # 100 days of daily data
        }
        
        data = {}
        progress_bar = st.progress(0)
        
        for i, (tf, limit) in enumerate(timeframes.items()):
            try:
                with st.spinner(f'Fetching {tf} data...'):
                    ohlcv = _self.exchange.fetch_ohlcv(symbol, tf, limit=limit)
                    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    df.set_index('timestamp', inplace=True)
                    df = _self.calculate_technical_indicators(df)
                    data[tf] = df
                progress_bar.progress((i + 1) / len(timeframes))
            except Exception as e:
                st.error(f"Error fetching {tf} data: {e}")
                
        progress_bar.empty()
        return data
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate comprehensive technical indicators"""
        if len(df) < 20:
            return df
            
        # Moving Averages
        df['sma_20'] = df['close'].rolling(20).mean()
        df['sma_50'] = df['close'].rolling(50).mean() if len(df) >= 50 else df['close'].rolling(len(df)//2).mean()
        df['ema_12'] = df['close'].ewm(span=12).mean()
        df['ema_26'] = df['close'].ewm(span=26).mean()
        
        # MACD
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(20).mean()
        bb_std = df['close'].rolling(20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        
        # Stochastic
        low_14 = df['low'].rolling(14).min()
        high_14 = df['high'].rolling(14).max()
        df['stoch_k'] = 100 * (df['close'] - low_14) / (high_14 - low_14)
        df['stoch_d'] = df['stoch_k'].rolling(3).mean()
        
        # Volume
        df['volume_sma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        return df
    
    def detect_chart_patterns(self, df: pd.DataFrame) -> list:
        """Detect chart patterns"""
        patterns = []
        
        if len(df) < 20:
            return patterns
            
        # Simple pattern detection
        recent_highs = df['high'].tail(10).values
        recent_lows = df['low'].tail(10).values
        
        # Double Top (simplified)
        peaks = []
        for i in range(2, len(recent_highs) - 2):
            if recent_highs[i] > recent_highs[i-1] and recent_highs[i] > recent_highs[i+1]:
                peaks.append(recent_highs[i])
        
        if len(peaks) >= 2 and abs(peaks[-1] - peaks[-2]) / peaks[-1] < 0.02:
            patterns.append({
                'type': 'Double Top',
                'signal': 'Bearish',
                'strength': 'Medium'
            })
        
        return patterns
    
    def analyze_timeframe(self, df: pd.DataFrame, timeframe: str) -> dict:
        """Analyze individual timeframe"""
        if len(df) < 10:
            return {'error': f'Insufficient data for {timeframe}'}
            
        latest = df.iloc[-1]
        
        signals = []
        strength = 0
        
        # RSI Analysis
        rsi = latest['rsi']
        if not pd.isna(rsi):
            if rsi < 30:
                signals.append(f'🟢 RSI Oversold ({rsi:.1f})')
                strength += 1
            elif rsi > 70:
                signals.append(f'🔴 RSI Overbought ({rsi:.1f})')
                strength -= 1
            else:
                signals.append(f'🟡 RSI Neutral ({rsi:.1f})')
        
        # MACD Analysis
        if not pd.isna(latest['macd']) and not pd.isna(latest['macd_signal']):
            if latest['macd'] > latest['macd_signal']:
                signals.append('🟢 MACD Bullish')
                strength += 1
            else:
                signals.append('🔴 MACD Bearish')
                strength -= 1
        
        # Moving Average Analysis
        if not pd.isna(latest['sma_20']) and not pd.isna(latest['sma_50']):
            if latest['sma_20'] > latest['sma_50']:
                signals.append('🟢 SMA Bullish Trend')
                strength += 1
            else:
                signals.append('🔴 SMA Bearish Trend')
                strength -= 1
        
        # Final recommendation
        if strength >= 2:
            recommendation = 'BUY'
            color = '🟢'
        elif strength <= -2:
            recommendation = 'SELL'
            color = '🔴'
        else:
            recommendation = 'HOLD'
            color = '🟡'
        
        return {
            'timeframe': timeframe,
            'price': latest['close'],
            'recommendation': recommendation,
            'strength': strength,
            'signals': signals,
            'color': color,
            'rsi': rsi if not pd.isna(rsi) else 0,
            'volume_ratio': latest['volume_ratio'] if not pd.isna(latest['volume_ratio']) else 1
        }
    
    def create_price_chart(self, df: pd.DataFrame, title: str) -> go.Figure:
        """Create interactive price chart with indicators"""
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=('Price & Moving Averages', 'RSI', 'MACD'),
            vertical_spacing=0.1,
            row_heights=[0.6, 0.2, 0.2]
        )
        
        # Candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'],
                name='Price'
            ),
            row=1, col=1
        )
        
        # Moving averages
        if 'sma_20' in df.columns:
            fig.add_trace(
                go.Scatter(x=df.index, y=df['sma_20'], name='SMA 20', line=dict(color='orange')),
                row=1, col=1
            )
        
        if 'sma_50' in df.columns:
            fig.add_trace(
                go.Scatter(x=df.index, y=df['sma_50'], name='SMA 50', line=dict(color='red')),
                row=1, col=1
            )
        
        # Bollinger Bands
        if all(col in df.columns for col in ['bb_upper', 'bb_lower']):
            fig.add_trace(
                go.Scatter(x=df.index, y=df['bb_upper'], name='BB Upper', 
                          line=dict(color='gray', dash='dash'), opacity=0.5),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(x=df.index, y=df['bb_lower'], name='BB Lower',
                          line=dict(color='gray', dash='dash'), opacity=0.5),
                row=1, col=1
            )
        
        # RSI
        if 'rsi' in df.columns:
            fig.add_trace(
                go.Scatter(x=df.index, y=df['rsi'], name='RSI', line=dict(color='purple')),
                row=2, col=1
            )
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
        
        # MACD
        if all(col in df.columns for col in ['macd', 'macd_signal']):
            fig.add_trace(
                go.Scatter(x=df.index, y=df['macd'], name='MACD', line=dict(color='blue')),
                row=3, col=1
            )
            fig.add_trace(
                go.Scatter(x=df.index, y=df['macd_signal'], name='Signal', line=dict(color='red')),
                row=3, col=1
            )
        
        fig.update_layout(
            title=title,
            template='plotly_dark',
            height=800,
            showlegend=False
        )
        
        return fig

def main():
    # App header
    st.title("📈 Advanced Crypto Trading Analysis Bot")
    st.markdown("*Analyze cryptocurrency markets with advanced technical indicators and pattern recognition*")
    
    # Sidebar controls
    st.sidebar.header("🔧 Controls")
    
    # Popular trading pairs
    popular_pairs = [
        'BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'ADA/USDT', 'XRP/USDT',
        'DOT/USDT', 'LINK/USDT', 'LTC/USDT', 'BCH/USDT', 'UNI/USDT'
    ]
    
    # User inputs
    symbol = st.sidebar.selectbox(
        "📊 Select Trading Pair",
        popular_pairs,
        index=0
    )
    
    # Custom symbol input
    custom_symbol = st.sidebar.text_input("Or enter custom pair (e.g., DOGE/USDT)")
    if custom_symbol:
        symbol = custom_symbol.upper()
    
    # Analysis button
    analyze_button = st.sidebar.button("🚀 Analyze Market", type="primary")
    
    # Auto-refresh option
    auto_refresh = st.sidebar.checkbox("🔄 Auto-refresh (30s)")
    
    if auto_refresh:
        time.sleep(30)
        st.rerun()
    
    # Main analysis
    if analyze_button or 'analysis_data' not in st.session_state:
        with st.spinner(f'🔍 Analyzing {symbol}...'):
            bot = StreamlitTradingBot('binance')
            
            try:
                # Get multi-timeframe data
                data = bot.get_multi_timeframe_data(symbol)
                
                if not data:
                    st.error("❌ Failed to fetch market data. Please try again.")
                    return
                
                # Store in session state
                st.session_state.analysis_data = data
                st.session_state.symbol = symbol
                st.session_state.last_update = datetime.now()
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                return
    
    # Use cached data
    if 'analysis_data' not in st.session_state:
        st.info("👆 Click 'Analyze Market' to start analysis")
        return
    
    data = st.session_state.analysis_data
    symbol = st.session_state.symbol
    
    # Header with last update
    st.success(f"✅ Analysis complete for {symbol}")
    st.caption(f"Last updated: {st.session_state.last_update.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Quick metrics
    if '1h' in data and not data['1h'].empty:
        latest_price = data['1h']['close'].iloc[-1]
        price_change_24h = ((data['1h']['close'].iloc[-1] - data['1h']['close'].iloc[-24]) / data['1h']['close'].iloc[-24] * 100) if len(data['1h']) >= 24 else 0
        volume_24h = data['1h']['volume'].tail(24).sum() if len(data['1h']) >= 24 else 0
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("💰 Current Price", f"${latest_price:.4f}")
        
        with col2:
            st.metric("📈 24h Change", f"{price_change_24h:.2f}%", 
                     delta=f"{price_change_24h:.2f}%")
        
        with col3:
            st.metric("📊 24h Volume", f"{volume_24h:,.0f}")
        
        with col4:
            latest_rsi = data['1h']['rsi'].iloc[-1] if not pd.isna(data['1h']['rsi'].iloc[-1]) else 50
            rsi_color = "normal"
            if latest_rsi > 70:
                rsi_color = "off"
            elif latest_rsi < 30:
                rsi_color = "normal"
            st.metric("⚡ RSI", f"{latest_rsi:.1f}", delta=None)
    
    # Multi-timeframe analysis
    st.header("🕐 Multi-Timeframe Analysis")
    
    timeframe_cols = st.columns(3)
    bot = StreamlitTradingBot('binance')
    
    for i, (tf, df) in enumerate(data.items()):
        if df.empty:
            continue
            
        analysis = bot.analyze_timeframe(df, tf)
        
        with timeframe_cols[i]:
            # Recommendation card
            rec_color = "success" if analysis['recommendation'] == 'BUY' else "error" if analysis['recommendation'] == 'SELL' else "warning"
            
            with st.container():
                st.subheader(f"{tf.upper()} Timeframe")
                
                # Recommendation badge
                if analysis['recommendation'] == 'BUY':
                    st.success(f"🟢 {analysis['recommendation']}")
                elif analysis['recommendation'] == 'SELL':
                    st.error(f"🔴 {analysis['recommendation']}")
                else:
                    st.warning(f"🟡 {analysis['recommendation']}")
                
                st.metric("Strength Score", analysis['strength'])
                
                # Signals
                st.write("**Signals:**")
                for signal in analysis['signals']:
                    st.write(f"• {signal}")
    
    # Detailed Charts
    st.header("📊 Technical Analysis Charts")
    
    chart_timeframe = st.selectbox("Select timeframe for detailed chart:", 
                                  list(data.keys()), 
                                  index=0)
    
    if chart_timeframe in data and not data[chart_timeframe].empty:
        chart_df = data[chart_timeframe]
        fig = bot.create_price_chart(chart_df, f"{symbol} - {chart_timeframe.upper()} Chart")
        st.plotly_chart(fig, use_container_width=True)
    
    # Pattern Analysis
    st.header("🔍 Pattern Recognition")
    
    patterns_detected = []
    for tf, df in data.items():
        if not df.empty:
            patterns = bot.detect_chart_patterns(df)
            for pattern in patterns:
                pattern['timeframe'] = tf
                patterns_detected.append(pattern)
    
    if patterns_detected:
        for pattern in patterns_detected:
            signal_color = "success" if pattern['signal'] == 'Bullish' else "error"
            
            with st.container():
                col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
                
                with col1:
                    st.write(f"**{pattern['type']}**")
                
                with col2:
                    if pattern['signal'] == 'Bullish':
                        st.success(pattern['signal'])
                    else:
                        st.error(pattern['signal'])
                
                with col3:
                    st.write(f"*{pattern['strength']}*")
                
                with col4:
                    st.write(f"*{pattern['timeframe']}*")
    else:
        st.info("🔍 No significant patterns detected in current analysis window")
    
    # Market Regime
    st.header("🌡️ Market Regime Analysis")
    
    if '1d' in data and not data['1d'].empty:
        daily_df = data['1d']
        
        # Calculate regime metrics
        current_price = daily_df['close'].iloc[-1]
        sma_20 = daily_df['sma_20'].iloc[-1] if not pd.isna(daily_df['sma_20'].iloc[-1]) else current_price
        sma_50 = daily_df['sma_50'].iloc[-1] if not pd.isna(daily_df['sma_50'].iloc[-1]) else current_price
        
        # Determine regime
        if current_price > sma_20 > sma_50:
            regime = "🟢 Strong Bull Market"
            regime_color = "success"
        elif current_price > sma_20:
            regime = "🟢 Mild Bull Market"
            regime_color = "success"
        elif current_price < sma_20 < sma_50:
            regime = "🔴 Strong Bear Market"
            regime_color = "error"
        elif current_price < sma_20:
            regime = "🔴 Mild Bear Market"
            regime_color = "error"
        else:
            regime = "🟡 Sideways Market"
            regime_color = "warning"
        
        col1, col2 = st.columns(2)
        
        with col1:
            if regime_color == "success":
                st.success(f"**Market Regime:** {regime}")
            elif regime_color == "error":
                st.error(f"**Market Regime:** {regime}")
            else:
                st.warning(f"**Market Regime:** {regime}")
        
        with col2:
            volatility = daily_df['close'].tail(20).std() / daily_df['close'].tail(20).mean() * 100
            st.metric("Volatility (20d)", f"{volatility:.2f}%")
    
    # Trading Insights
    st.header("💡 Trading Insights")
    
    insights = []
    
    # Multi-timeframe confluence
    recommendations = [bot.analyze_timeframe(df, tf)['recommendation'] for tf, df in data.items() if not df.empty]
    
    if recommendations.count('BUY') >= 2:
        insights.append("🟢 **Multi-timeframe BUY confluence detected** - Strong bullish signal")
    elif recommendations.count('SELL') >= 2:
        insights.append("🔴 **Multi-timeframe SELL confluence detected** - Strong bearish signal")
    elif len(set(recommendations)) == len(recommendations):
        insights.append("🟡 **Mixed signals across timeframes** - Wait for clearer direction")
    
    # Volume insights
    if '1h' in data and not data['1h'].empty:
        latest_volume_ratio = data['1h']['volume_ratio'].iloc[-1]
        if not pd.isna(latest_volume_ratio):
            if latest_volume_ratio > 2:
                insights.append("🔊 **Exceptional volume** - Strong institutional interest")
            elif latest_volume_ratio < 0.3:
                insights.append("🔇 **Low volume** - Lack of conviction in current move")
    
    if insights:
        for insight in insights:
            st.markdown(insight)
    else:
        st.info("📊 Market showing normal trading conditions")
    
    # Footer
    st.markdown("---")
    st.markdown("*Disclaimer: This analysis is for educational purposes only. Always do your own research before making trading decisions.*")

# Sidebar info
with st.sidebar:
    st.markdown("---")
    st.markdown("### 📖 How to Use")
    st.markdown("""
    1. **Select** a trading pair
    2. **Click** 'Analyze Market'
    3. **Review** multi-timeframe signals
    4. **Check** pattern recognition
    5. **Consider** market regime
    
    **Green** = Bullish signals
    **Red** = Bearish signals  
    **Yellow** = Neutral/Hold
    """)
    
    st.markdown("### ⚙️ Features")
    st.markdown("""
    - 🕐 Multi-timeframe analysis
    - 📊 Technical indicators
    - 🔍 Pattern recognition
    - 🌡️ Market regime detection
    - 📈 Interactive charts
    """)

if __name__ == "__main__":
    main()
