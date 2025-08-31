import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import time
import json

# Page configuration
st.set_page_config(
    page_title="Crypto Trading Analysis Bot",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

class CoinGeckoTradingBot:
    def __init__(self):
        """Initialize the trading analysis bot using CoinGecko API"""
        self.base_url = "https://api.coingecko.com/api/v3"
        
    @st.cache_data(ttl=3600) # Cache for 1 hour
    def get_all_coins_list(_self):
        """Get a comprehensive list of all available coins with their IDs and symbols for search."""
        try:
            response = requests.get(f"{_self.base_url}/coins/list")
            response.raise_for_status() # Raise an exception for HTTP errors
            coins = response.json()
            return {coin['name']: {'id': coin['id'], 'symbol': coin['symbol'].upper()} for coin in coins}
        except requests.exceptions.RequestException as e:
            st.error(f"Error fetching full coin list: {e}")
            return {}
        except Exception as e:
            st.error(f"An unexpected error occurred while fetching coin list: {e}")
            return {}

    def get_coin_list(self):
        """Get list of available coins (used for popular coins dropdown initially)"""
        try:
            response = requests.get(f"{self.base_url}/coins/markets", 
                                  params={'vs_currency': 'usd', 'order': 'market_cap_desc', 'per_page': 100})
            coins = response.json()
            return {coin['name']: coin['id'] for coin in coins}
        except Exception as e:
            st.error(f"Error fetching coin list: {e}")
            return {}
    
    @st.cache_data(ttl=300)
    def get_price_data(_self, coin_id: str, days: int = 30) -> pd.DataFrame:
        """Fetch OHLC data from CoinGecko"""
        try:
            # Get OHLC data
            response = requests.get(f"{_self.base_url}/coins/{coin_id}/ohlc", 
                                  params={'vs_currency': 'usd', 'days': days})
            data = response.json()
            
            if not data:
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Get volume data separately
            volume_response = requests.get(f"{_self.base_url}/coins/{coin_id}/market_chart", 
                                        params={'vs_currency': 'usd', 'days': days})
            volume_data = volume_response.json()
            
            if 'total_volumes' in volume_data:
                volume_df = pd.DataFrame(volume_data['total_volumes'], columns=['timestamp', 'volume'])
                volume_df['timestamp'] = pd.to_datetime(volume_df['timestamp'], unit='ms')
                volume_df.set_index('timestamp', inplace=True)
                
                # Merge volume data
                df = df.merge(volume_df, left_index=True, right_index=True, how='left')
            else:
                df['volume'] = 0
            
            return df
            
        except Exception as e:
            st.error(f"Error fetching price data: {e}")
            return pd.DataFrame()
    
    def get_current_price_info(self, coin_id: str) -> dict:
        """Get current price and market data"""
        try:
            response = requests.get(f"{self.base_url}/coins/{coin_id}")
            data = response.json()
            
            market_data = data.get('market_data', {})
            
            return {
                'current_price': market_data.get('current_price', {}).get('usd', 0),
                'price_change_24h': market_data.get('price_change_percentage_24h', 0),
                'market_cap': market_data.get('market_cap', {}).get('usd', 0),
                'volume_24h': market_data.get('total_volume', {}).get('usd', 0),
                'high_24h': market_data.get('high_24h', {}).get('usd', 0),
                'low_24h': market_data.get('low_24h', {}).get('usd', 0),
                'market_cap_rank': data.get('market_cap_rank', 0),
                'name': data.get('name', ''),
                'symbol': data.get('symbol', '').upper()
            }
        except Exception as e:
            st.error(f"Error fetching current price: {e}")
            return {}
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators"""
        if len(df) < 20:
            return df
            
        # Moving Averages
        df['sma_7'] = df['close'].rolling(7).mean()
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
        df['bb_width'] = ((df['bb_upper'] - df['bb_lower']) / df['bb_middle']) * 100
        
        # Stochastic
        low_14 = df['low'].rolling(14).min()
        high_14 = df['high'].rolling(14).max()
        df['stoch_k'] = 100 * (df['close'] - low_14) / (high_14 - low_14)
        df['stoch_d'] = df['stoch_k'].rolling(3).mean()
        
        # Volume indicators
        if 'volume' in df.columns and df['volume'].sum() > 0:
            df['volume_sma'] = df['volume'].rolling(20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma']
        else:
            df['volume_sma'] = 0
            df['volume_ratio'] = 1
        
        # Volatility
        df['volatility'] = df['close'].rolling(20).std() / df['close'].rolling(20).mean() * 100
        
        return df
    
    def detect_chart_patterns(self, df: pd.DataFrame) -> list:
        """Detect chart patterns"""
        patterns = []
        
        if len(df) < 20:
            return patterns
        
        # Recent price action
        recent = df.tail(20)
        
        # Double Top Pattern
        highs = recent['high'].values
        peaks = []
        for i in range(2, len(highs) - 2):
            if highs[i] > highs[i-1] and highs[i] > highs[i+1] and highs[i] > highs[i-2] and highs[i] > highs[i+2]:
                peaks.append((i, highs[i]))
        
        if len(peaks) >= 2:
            last_two_peaks = peaks[-2:]
            if abs(last_two_peaks[0][1] - last_two_peaks[1][1]) / last_two_peaks[0][1] < 0.03:
                patterns.append({
                    'type': 'Double Top',
                    'signal': 'Bearish',
                    'strength': 'Medium',
                    'description': 'Two similar highs detected - potential reversal'
                })
        
        # Breakout Detection
        recent_high = recent['high'].max()
        recent_low = recent['low'].min()
        current_price = df['close'].iloc[-1]
        
        if current_price > recent_high * 0.99:
            patterns.append({
                'type': 'Breakout Above Resistance',
                'signal': 'Bullish',
                'strength': 'Strong',
                'description': 'Price breaking above recent resistance'
            })
        elif current_price < recent_low * 1.01:
            patterns.append({
                'type': 'Breakdown Below Support',
                'signal': 'Bearish', 
                'strength': 'Strong',
                'description': 'Price breaking below recent support'
            })
        
        return patterns
    
    def analyze_signals(self, df: pd.DataFrame) -> dict:
        """Comprehensive signal analysis"""
        if len(df) < 20:
            return {'error': 'Insufficient data'}
            
        latest = df.iloc[-1]
        signals = []
        score = 0
        
        # RSI Analysis
        rsi = latest['rsi']
        if not pd.isna(rsi):
            if rsi < 30:
                signals.append({'indicator': 'RSI', 'signal': 'OVERSOLD', 'strength': 'Strong', 'bullish': True})
                score += 2
            elif rsi < 40:
                signals.append({'indicator': 'RSI', 'signal': 'Weak', 'strength': 'Medium', 'bullish': True})
                score += 1
            elif rsi > 70:
                signals.append({'indicator': 'RSI', 'signal': 'OVERBOUGHT', 'strength': 'Strong', 'bullish': False})
                score -= 2
            elif rsi > 60:
                signals.append({'indicator': 'RSI', 'signal': 'Strong', 'strength': 'Medium', 'bullish': False})
                score -= 1
            else:
                signals.append({'indicator': 'RSI', 'signal': 'Neutral', 'strength': 'Weak', 'bullish': None})
        
        # MACD Analysis
        if not pd.isna(latest['macd']) and not pd.isna(latest['macd_signal']):
            if latest['macd'] > latest['macd_signal']:
                if latest['macd_histogram'] > 0:
                    signals.append({'indicator': 'MACD', 'signal': 'Bullish Momentum', 'strength': 'Strong', 'bullish': True})
                    score += 2
                else:
                    signals.append({'indicator': 'MACD', 'signal': 'Bullish', 'strength': 'Medium', 'bullish': True})
                    score += 1
            else:
                if latest['macd_histogram'] < 0:
                    signals.append({'indicator': 'MACD', 'signal': 'Bearish Momentum', 'strength': 'Strong', 'bullish': False})
                    score -= 2
                else:
                    signals.append({'indicator': 'MACD', 'signal': 'Bearish', 'strength': 'Medium', 'bullish': False})
                    score -= 1
        
        # Moving Average Trend
        if not pd.isna(latest['sma_7']) and not pd.isna(latest['sma_20']):
            if latest['close'] > latest['sma_7'] > latest['sma_20']:
                signals.append({'indicator': 'MA Trend', 'signal': 'Strong Uptrend', 'strength': 'Strong', 'bullish': True})
                score += 2
            elif latest['close'] > latest['sma_20']:
                signals.append({'indicator': 'MA Trend', 'signal': 'Uptrend', 'strength': 'Medium', 'bullish': True})
                score += 1
            elif latest['close'] < latest['sma_7'] < latest['sma_20']:
                signals.append({'indicator': 'MA Trend', 'signal': 'Strong Downtrend', 'strength': 'Strong', 'bullish': False})
                score -= 2
            elif latest['close'] < latest['sma_20']:
                signals.append({'indicator': 'MA Trend', 'signal': 'Downtrend', 'strength': 'Medium', 'bullish': False})
                score -= 1
        
        # Bollinger Bands
        if all(col in df.columns for col in ['bb_upper', 'bb_lower', 'bb_middle']):
            if latest['close'] > latest['bb_upper']:
                signals.append({'indicator': 'Bollinger', 'signal': 'Overbought', 'strength': 'Medium', 'bullish': False})
                score -= 1
            elif latest['close'] < latest['bb_lower']:
                signals.append({'indicator': 'Bollinger', 'signal': 'Oversold', 'strength': 'Medium', 'bullish': True})
                score += 1
        
        # Volume confirmation
        if latest['volume_ratio'] > 1.5:
            signals.append({'indicator': 'Volume', 'signal': 'High Volume Confirmation', 'strength': 'Medium', 'bullish': None})
        elif latest['volume_ratio'] < 0.5:
            signals.append({'indicator': 'Volume', 'signal': 'Low Volume Warning', 'strength': 'Weak', 'bullish': None})
        
        # Final recommendation
        if score >= 4:
            recommendation = 'STRONG BUY'
            confidence = min(95, 70 + abs(score) * 5)
        elif score >= 2:
            recommendation = 'BUY'
            confidence = min(85, 60 + abs(score) * 5)
        elif score <= -4:
            recommendation = 'STRONG SELL'
            confidence = min(95, 70 + abs(score) * 5)
        elif score <= -2:
            recommendation = 'SELL'
            confidence = min(85, 60 + abs(score) * 5)
        else:
            recommendation = 'HOLD'
            confidence = 50
        
        return {
            'signals': signals,
            'score': score,
            'recommendation': recommendation,
            'confidence': confidence,
            'rsi': latest['rsi'] if not pd.isna(latest['rsi']) else 50,
            'macd_signal': 'Bullish' if latest['macd'] > latest['macd_signal'] else 'Bearish'
        }
    
    def create_price_chart(self, df: pd.DataFrame, coin_name: str) -> go.Figure:
        """Create comprehensive price chart"""
        fig = make_subplots(
            rows=4, cols=1,
            subplot_titles=(
                f'{coin_name} Price & Moving Averages', 
                'RSI (Relative Strength Index)', 
                'MACD',
                'Volume'
            ),
            vertical_spacing=0.08,
            row_heights=[0.5, 0.15, 0.15, 0.2]
        )
        
        # Price chart with moving averages
        fig.add_trace(
            go.Scatter(x=df.index, y=df['close'], name='Price', 
                      line=dict(color='white', width=2)),
            row=1, col=1
        )
        
        if 'sma_7' in df.columns:
            fig.add_trace(
                go.Scatter(x=df.index, y=df['sma_7'], name='SMA 7', 
                          line=dict(color='orange', width=1)),
                row=1, col=1
            )
        
        if 'sma_20' in df.columns:
            fig.add_trace(
                go.Scatter(x=df.index, y=df['sma_20'], name='SMA 20', 
                          line=dict(color='blue', width=1)),
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
                go.Scatter(x=df.index, y=df['rsi'], name='RSI', 
                          line=dict(color='purple')),
                row=2, col=1
            )
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
            fig.add_hline(y=50, line_dash="dot", line_color="gray", row=2, col=1)
        
        # MACD
        if all(col in df.columns for col in ['macd', 'macd_signal', 'macd_histogram']):
            fig.add_trace(
                go.Scatter(x=df.index, y=df['macd'], name='MACD', 
                          line=dict(color='blue')),
                row=3, col=1
            )
            fig.add_trace(
                go.Scatter(x=df.index, y=df['macd_signal'], name='Signal', 
                          line=dict(color='red')),
                row=3, col=1
            )
            # MACD Histogram
            colors = ['green' if val > 0 else 'red' for val in df['macd_histogram']]
            fig.add_trace(
                go.Bar(x=df.index, y=df['macd_histogram'], name='Histogram',
                      marker_color=colors, opacity=0.6),
                row=3, col=1
            )
        
        # Volume
        if 'volume' in df.columns:
            fig.add_trace(
                go.Bar(x=df.index, y=df['volume'], name='Volume',
                      marker_color='rgba(0,100,200,0.6)'),
                row=4, col=1
            )
        
        fig.update_layout(
            template='plotly_dark',
            height=800,
            showlegend=False,
            title_x=0.5
        )
        
        fig.update_xaxes(rangeslider_visible=False)
        
        return fig

def main():
    # App header
    st.title("📈 Advanced Crypto Trading Analysis")
    st.markdown("*Professional cryptocurrency market analysis powered by CoinGecko*")
    
    # Initialize bot
    bot = CoinGeckoTradingBot()
    
    # Sidebar controls
    st.sidebar.header("🔧 Analysis Controls")
    
    # Get all coins for search
    all_coins_data = bot.get_all_coins_list()
    all_coin_names = sorted(list(all_coins_data.keys()))

    # Search feature
    search_query = st.sidebar.text_input("🔍 Search for a Cryptocurrency", "").strip()
    
    # Coin selection logic
    selected_coin_name = None
    coin_id = None

    if search_query:
        # Filter coins based on search query
        filtered_coins = [
            name for name in all_coin_names 
            if search_query.lower() in name.lower() or 
               search_query.lower() == all_coins_data[name]['symbol'].lower()
        ]
        
        if filtered_coins:
            selected_coin_name = st.sidebar.selectbox(
                "Select from search results",
                filtered_coins,
                index=0
            )
            coin_id = all_coins_data[selected_coin_name]['id']
        else:
            st.sidebar.info("No coins found matching your search.")
            st.session_state.current_analysis = None # Clear previous analysis if no coin is found
    else:
        # Get popular coins if no search query
        with st.spinner("Loading popular coins..."):
            popular_coin_options = bot.get_coin_list()
        
        if popular_coin_options:
            selected_coin_name = st.sidebar.selectbox(
                "📊 Select a Popular Cryptocurrency",
                list(popular_coin_options.keys()),
                index=0
            )
            coin_id = popular_coin_options[selected_coin_name]
        else:
            st.sidebar.error("Could not load popular coins. Please try searching.")
            st.session_state.current_analysis = None


    # Analysis period
    analysis_days = st.sidebar.selectbox(
        "📅 Analysis Period",
        [7, 14, 30, 90],
        index=2
    )
    
    # Analysis button
    analyze_button = st.sidebar.button("🚀 Analyze Now", type="primary", disabled=(coin_id is None))
    
    # Auto-refresh
    auto_refresh = st.sidebar.checkbox("🔄 Auto-refresh (60s)")
    
    if auto_refresh:
        time.sleep(60)
        st.rerun()
    
    # Main analysis
    # Trigger analysis if button is clicked OR if a coin is newly selected from search/dropdown
    # and no analysis is currently displayed OR if a different coin is selected.
    trigger_analysis = False
    if analyze_button:
        trigger_analysis = True
    elif 'current_analysis' not in st.session_state and coin_id:
        trigger_analysis = True
    elif 'current_analysis' in st.session_state and coin_id and st.session_state.current_analysis.get('coin_id') != coin_id:
        # If the coin ID has changed, re-analyze
        trigger_analysis = True

    if trigger_analysis and coin_id: # Ensure coin_id is not None
        with st.spinner(f'🔍 Analyzing {selected_coin_name}...'):
            try:
                # Get current price info
                price_info = bot.get_current_price_info(coin_id)
                
                if not price_info:
                    st.error("❌ Failed to fetch coin data. Please try again or search for a different coin.")
                    st.session_state.current_analysis = None
                    return
                
                # Get historical data
                df = bot.get_price_data(coin_id, days=analysis_days)
                
                if df.empty:
                    st.error("❌ No historical data available for this coin for the selected period.")
                    st.session_state.current_analysis = None
                    return
                
                # Calculate indicators
                df = bot.calculate_technical_indicators(df)
                
                # Analyze signals
                analysis = bot.analyze_signals(df)
                
                # Store in session
                st.session_state.current_analysis = {
                    'price_info': price_info,
                    'df': df,
                    'analysis': analysis,
                    'coin_name': selected_coin_name,
                    'coin_id': coin_id, # Store coin_id to check for changes
                    'last_update': datetime.now()
                }
                
            except Exception as e:
                st.error(f"❌ Analysis failed: {str(e)}")
                st.session_state.current_analysis = None
                return
    
    # Display results
    if 'current_analysis' not in st.session_state or st.session_state.current_analysis is None:
        st.info("👆 Search for a coin or select from popular options, then click 'Analyze Now' to begin analysis.")
        return
    
    data = st.session_state.current_analysis
    price_info = data['price_info']
    df = data['df']
    analysis = data['analysis']
    coin_name = data['coin_name']
    
    # Success message
    st.success(f"✅ Analysis complete for {coin_name} ({price_info['symbol']})")
    st.caption(f"Last updated: {data['last_update'].strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Quick metrics row
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            "💰 Current Price", 
            f"${price_info['current_price']:,.4f}",
            delta=f"{price_info['price_change_24h']:.2f}%"
        )
    
    with col2:
        st.metric("📊 Market Cap Rank", f"#{price_info['market_cap_rank']}")
    
    with col3:
        st.metric("📈 24h High", f"${price_info['high_24h']:,.4f}")
    
    with col4:
        st.metric("📉 24h Low", f"${price_info['low_24h']:,.4f}")
    
    with col5:
        st.metric("💨 24h Volume", f"${price_info['volume_24h']:,.0f}")
    
    # Main recommendation
    rec = analysis['recommendation']
    conf = analysis['confidence']
    
    if rec in ['BUY', 'STRONG BUY']:
        st.success(f"🟢 **{rec}** (Confidence: {conf}%)")
    elif rec in ['SELL', 'STRONG SELL']:
        st.error(f"🔴 **{rec}** (Confidence: {conf}%)")
    else:
        st.warning(f"🟡 **{rec}** (Confidence: {conf}%)")
    
    # Technical Analysis Chart
    st.header("📊 Technical Analysis Chart")
    
    fig = bot.create_price_chart(df, coin_name)
    st.plotly_chart(fig, use_container_width=True)
    
    # Detailed Analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Signal Analysis")
        
        for signal in analysis['signals']:
            if signal['bullish'] is True:
                st.success(f"🟢 **{signal['indicator']}**: {signal['signal']} ({signal['strength']})")
            elif signal['bullish'] is False:
                st.error(f"🔴 **{signal['indicator']}**: {signal['signal']} ({signal['strength']})")
            else:
                st.info(f"🟡 **{signal['indicator']}**: {signal['signal']} ({signal['strength']})")
    
    with col2:
        st.subheader("🔍 Pattern Recognition")
        
        patterns = bot.detect_chart_patterns(df)
        
        if patterns:
            for pattern in patterns:
                if pattern['signal'] == 'Bullish':
                    st.success(f"🟢 **{pattern['type']}** - {pattern['description']}")
                else:
                    st.error(f"🔴 **{pattern['type']}** - {pattern['description']}")
        else:
            st.info("🔍 No significant patterns detected")
    
    # Market Insights
    st.header("💡 Trading Insights")
    
    insights = []
    
    # RSI insights
    rsi = analysis['rsi']
    if rsi < 30:
        insights.append("🟢 **RSI suggests oversold conditions** - Potential buying opportunity")
    elif rsi > 70:
        insights.append("🔴 **RSI indicates overbought conditions** - Consider taking profits")
    
    # Volatility insights
    current_volatility = df['volatility'].iloc[-1] if not pd.isna(df['volatility'].iloc[-1]) else 0
    avg_volatility = df['volatility'].mean()
    
    if current_volatility > avg_volatility * 1.5:
        insights.append("⚠️ **High volatility detected** - Expect larger price swings")
    elif current_volatility < avg_volatility * 0.5:
        insights.append("😴 **Low volatility period** - Possible breakout incoming")
    
    # Trend insights
    if analysis['score'] >= 3:
        insights.append("🚀 **Multiple bullish signals converging** - Strong upward momentum")
    elif analysis['score'] <= -3:
        insights.append("📉 **Multiple bearish signals present** - Caution advised")
    
    if insights:
        for insight in insights:
            st.markdown(insight)
    else:
        st.info("📊 Market showing balanced conditions - Monitor for clearer signals")
    
    # Technical Indicators Table
    st.header("📋 Current Indicator Values")
    
    latest = df.iloc[-1]
    
    indicators_data = {
        'Indicator': ['RSI (14)', 'MACD', 'SMA 7', 'SMA 20', 'BB Position', 'Stochastic %K'],
        'Value': [
            f"{latest['rsi']:.1f}" if not pd.isna(latest['rsi']) else 'N/A',
            f"{latest['macd']:.6f}" if not pd.isna(latest['macd']) else 'N/A',
            f"${latest['sma_7']:.4f}" if not pd.isna(latest['sma_7']) else 'N/A',
            f"${latest['sma_20']:.4f}" if not pd.isna(latest['sma_20']) else 'N/A',
            f"{((latest['close'] - latest['bb_lower']) / (latest['bb_upper'] - latest['bb_lower']) * 100):.1f}%" if all(not pd.isna(latest[col]) for col in ['bb_upper', 'bb_lower']) else 'N/A',
            f"{latest['stoch_k']:.1f}" if not pd.isna(latest['stoch_k']) else 'N/A'
        ],
        'Signal': [
            'Oversold' if not pd.isna(latest['rsi']) and latest['rsi'] < 30 else ('Overbought' if not pd.isna(latest['rsi']) and latest['rsi'] > 70 else 'Neutral'),
            'Bullish' if not pd.isna(latest['macd']) and not pd.isna(latest['macd_signal']) and latest['macd'] > latest['macd_signal'] else 'Bearish',
            'Above' if not pd.isna(latest['close']) and not pd.isna(latest['sma_7']) and latest['close'] > latest['sma_7'] else 'Below',
            'Above' if not pd.isna(latest['close']) and not pd.isna(latest['sma_20']) and latest['close'] > latest['sma_20'] else 'Below', 
            'Upper' if not pd.isna(latest['close']) and not pd.isna(latest['bb_upper']) and latest['close'] > latest['bb_upper'] else ('Lower' if not pd.isna(latest['close']) and not pd.isna(latest['bb_lower']) and latest['close'] < latest['bb_lower'] else 'Middle'),
            'Oversold' if not pd.isna(latest['stoch_k']) and latest['stoch_k'] < 20 else ('Overbought' if not pd.isna(latest['stoch_k']) and latest['stoch_k'] > 80 else 'Neutral')
        ]
    }
    
    indicators_df = pd.DataFrame(indicators_data)
    st.dataframe(indicators_df, use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown("### 📘 How to Read the Analysis")
    st.markdown("""
    **🟢 BUY signals:** Multiple bullish indicators align  
    **🔴 SELL signals:** Multiple bearish indicators align  
    **🟡 HOLD signals:** Mixed or neutral indicators  
    
    **RSI:** < 30 = Oversold (buy opportunity), > 70 = Overbought (sell opportunity)  
    **MACD:** Above signal line = Bullish momentum  
    **Moving Averages:** Price above = Uptrend, below = Downtrend  
    """)
    
    st.markdown("*⚠️ Disclaimer: This analysis is for educational purposes only. Always conduct your own research before making trading decisions.*")

# Sidebar info
with st.sidebar:
    st.markdown("---")
    st.markdown("### 📖 About This Bot")
    st.markdown("""
    **Data Source:** CoinGecko API  
    **Update Frequency:** Real-time  
    **Analysis Type:** Technical Analysis  
    **Indicators Used:** RSI, MACD, Moving Averages, Bollinger Bands, Volume
    """)
    
    st.markdown("### 🎯 Features")
    st.markdown("""
    - ✅ Real-time price data
    - ✅ Multi-indicator analysis  
    - ✅ Pattern recognition
    - ✅ Interactive charts
    - ✅ Clear buy/sell signals
    - ✅ No geo-restrictions
    """)

if __name__ == "__main__":
    main()
