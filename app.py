import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import time
import json
from typing import Dict, List, Optional

# Page configuration
st.set_page_config(
    page_title="Crypto Trading Analysis Bot",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

class OptimizedCoinGeckoBot:
    def __init__(self):
        """Initialize the optimized trading analysis bot using CoinGecko API"""
        self.base_url = "https://api.coingecko.com/api/v3"
        self.request_delay = 2.5  # Delay between requests to avoid rate limits
        self.last_request_time = 0
        
    def _rate_limit_delay(self):
        """Ensure minimum delay between API requests"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.request_delay:
            time.sleep(self.request_delay - time_since_last)
        
        self.last_request_time = time.time()
    
    def _make_request(self, url: str, params: dict = None, retries: int = 3) -> Optional[dict]:
        """Make API request with retry logic and rate limiting"""
        for attempt in range(retries):
            try:
                self._rate_limit_delay()
                response = requests.get(url, params=params, timeout=10)
                response.raise_for_status()
                return response.json()
            except requests.exceptions.RequestException as e:
                if attempt < retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    st.warning(f"Request failed, retrying in {wait_time} seconds... (attempt {attempt + 1})")
                    time.sleep(wait_time)
                else:
                    st.error(f"API request failed after {retries} attempts: {e}")
                    return None
        return None
    
    @st.cache_data(ttl=900)  # Cache for 15 minutes
    def get_comprehensive_coin_data(_self, coin_id: str, days: int = 30) -> tuple:
        """Fetch all required data for a coin in optimized batch calls"""
        try:
            # First, try to get data from the market_chart endpoint which includes price and volume
            market_chart_data = _self._make_request(
                f"{_self.base_url}/coins/{coin_id}/market_chart",
                params={'vs_currency': 'usd', 'days': days}
            )
            
            if not market_chart_data:
                return pd.DataFrame(), {}
            
            # Get OHLC data separately (this is the only additional call needed)
            ohlc_data = _self._make_request(
                f"{_self.base_url}/coins/{coin_id}/ohlc",
                params={'vs_currency': 'usd', 'days': days}
            )
            
            # Get current detailed info
            current_info = _self._make_request(f"{_self.base_url}/coins/{coin_id}")
            
            # Process OHLC data
            df = pd.DataFrame()
            if ohlc_data:
                df = pd.DataFrame(ohlc_data, columns=['timestamp', 'open', 'high', 'low', 'close'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
            
            # Add volume data from market_chart
            if 'total_volumes' in market_chart_data and not df.empty:
                volume_df = pd.DataFrame(market_chart_data['total_volumes'], columns=['timestamp', 'volume'])
                volume_df['timestamp'] = pd.to_datetime(volume_df['timestamp'], unit='ms')
                volume_df.set_index('timestamp', inplace=True)
                
                # Merge volume data
                df = df.merge(volume_df, left_index=True, right_index=True, how='left')
            elif not df.empty:
                df['volume'] = 0
            
            # Process current price info
            price_info = {}
            if current_info:
                market_data = current_info.get('market_data', {})
                price_info = {
                    'current_price': market_data.get('current_price', {}).get('usd', 0),
                    'price_change_24h': market_data.get('price_change_percentage_24h', 0),
                    'price_change_7d': market_data.get('price_change_percentage_7d', 0),
                    'market_cap': market_data.get('market_cap', {}).get('usd', 0),
                    'volume_24h': market_data.get('total_volume', {}).get('usd', 0),
                    'high_24h': market_data.get('high_24h', {}).get('usd', 0),
                    'low_24h': market_data.get('low_24h', {}).get('usd', 0),
                    'market_cap_rank': current_info.get('market_cap_rank', 0),
                    'name': current_info.get('name', ''),
                    'symbol': current_info.get('symbol', '').upper(),
                    'circulating_supply': market_data.get('circulating_supply', 0),
                    'total_supply': market_data.get('total_supply', 0),
                    'ath': market_data.get('ath', {}).get('usd', 0),
                    'atl': market_data.get('atl', {}).get('usd', 0)
                }
            
            return df, price_info
            
        except Exception as e:
            st.error(f"Error fetching comprehensive coin data: {e}")
            return pd.DataFrame(), {}
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators with improved error handling"""
        if len(df) < 20:
            return df
        
        try:
            # Moving Averages
            df['sma_7'] = df['close'].rolling(7, min_periods=7).mean()
            df['sma_20'] = df['close'].rolling(20, min_periods=20).mean()
            df['sma_50'] = df['close'].rolling(50, min_periods=min(50, len(df)//2)).mean()
            df['ema_12'] = df['close'].ewm(span=12, min_periods=12).mean()
            df['ema_26'] = df['close'].ewm(span=26, min_periods=26).mean()
            
            # MACD
            df['macd'] = df['ema_12'] - df['ema_26']
            df['macd_signal'] = df['macd'].ewm(span=9, min_periods=9).mean()
            df['macd_histogram'] = df['macd'] - df['macd_signal']
            
            # RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # Bollinger Bands
            df['bb_middle'] = df['close'].rolling(20, min_periods=20).mean()
            bb_std = df['close'].rolling(20, min_periods=20).std()
            df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
            df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
            df['bb_width'] = ((df['bb_upper'] - df['bb_lower']) / df['bb_middle']) * 100
            
            # Stochastic
            low_14 = df['low'].rolling(14, min_periods=14).min()
            high_14 = df['high'].rolling(14, min_periods=14).max()
            df['stoch_k'] = 100 * (df['close'] - low_14) / (high_14 - low_14)
            df['stoch_d'] = df['stoch_k'].rolling(3, min_periods=3).mean()
            
            # Volume indicators
            if 'volume' in df.columns and df['volume'].sum() > 0:
                df['volume_sma'] = df['volume'].rolling(20, min_periods=20).mean()
                df['volume_ratio'] = df['volume'] / df['volume_sma']
            else:
                df['volume_sma'] = 0
                df['volume_ratio'] = 1
            
            # Volatility
            df['volatility'] = df['close'].rolling(20, min_periods=20).std() / df['close'].rolling(20, min_periods=20).mean() * 100
            
        except Exception as e:
            st.warning(f"Error calculating some technical indicators: {e}")
        
        return df
    
    def detect_chart_patterns(self, df: pd.DataFrame) -> list:
        """Detect chart patterns with improved logic"""
        patterns = []
        
        if len(df) < 20:
            return patterns
        
        try:
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
            
            # Support/Resistance Breakout
            recent_high = recent['high'].max()
            recent_low = recent['low'].min()
            current_price = df['close'].iloc[-1]
            price_range = recent_high - recent_low
            
            if current_price > recent_high * 0.995:  # Within 0.5% of breaking resistance
                patterns.append({
                    'type': 'Resistance Breakout',
                    'signal': 'Bullish',
                    'strength': 'Strong',
                    'description': f'Price breaking above resistance at ${recent_high:.4f}'
                })
            elif current_price < recent_low * 1.005:  # Within 0.5% of breaking support
                patterns.append({
                    'type': 'Support Breakdown',
                    'signal': 'Bearish', 
                    'strength': 'Strong',
                    'description': f'Price breaking below support at ${recent_low:.4f}'
                })
            
            # Ascending Triangle
            if len(peaks) >= 2 and recent_low < recent['low'].tail(10).max():
                patterns.append({
                    'type': 'Ascending Triangle',
                    'signal': 'Bullish',
                    'strength': 'Medium',
                    'description': 'Higher lows with resistance level - bullish pattern'
                })
            
        except Exception as e:
            st.warning(f"Error in pattern detection: {e}")
        
        return patterns
    
    def analyze_signals(self, df: pd.DataFrame) -> dict:
        """Enhanced signal analysis with better scoring"""
        if len(df) < 20:
            return {'error': 'Insufficient data for analysis'}
            
        latest = df.iloc[-1]
        signals = []
        score = 0
        
        try:
            # RSI Analysis with multiple timeframes
            rsi = latest['rsi']
            if not pd.isna(rsi):
                if rsi < 25:
                    signals.append({'indicator': 'RSI', 'signal': 'EXTREMELY OVERSOLD', 'strength': 'Very Strong', 'bullish': True})
                    score += 3
                elif rsi < 30:
                    signals.append({'indicator': 'RSI', 'signal': 'OVERSOLD', 'strength': 'Strong', 'bullish': True})
                    score += 2
                elif rsi < 40:
                    signals.append({'indicator': 'RSI', 'signal': 'Weak', 'strength': 'Medium', 'bullish': True})
                    score += 1
                elif rsi > 75:
                    signals.append({'indicator': 'RSI', 'signal': 'EXTREMELY OVERBOUGHT', 'strength': 'Very Strong', 'bullish': False})
                    score -= 3
                elif rsi > 70:
                    signals.append({'indicator': 'RSI', 'signal': 'OVERBOUGHT', 'strength': 'Strong', 'bullish': False})
                    score -= 2
                elif rsi > 60:
                    signals.append({'indicator': 'RSI', 'signal': 'Strong', 'strength': 'Medium', 'bullish': False})
                    score -= 1
                else:
                    signals.append({'indicator': 'RSI', 'signal': 'Neutral', 'strength': 'Weak', 'bullish': None})
            
            # Enhanced MACD Analysis
            if not pd.isna(latest['macd']) and not pd.isna(latest['macd_signal']):
                macd_diff = latest['macd'] - latest['macd_signal']
                histogram = latest['macd_histogram']
                
                if latest['macd'] > latest['macd_signal']:
                    if histogram > 0 and len(df) > 2 and df['macd_histogram'].iloc[-2] < histogram:
                        signals.append({'indicator': 'MACD', 'signal': 'Strong Bullish Momentum', 'strength': 'Very Strong', 'bullish': True})
                        score += 3
                    elif histogram > 0:
                        signals.append({'indicator': 'MACD', 'signal': 'Bullish Momentum', 'strength': 'Strong', 'bullish': True})
                        score += 2
                    else:
                        signals.append({'indicator': 'MACD', 'signal': 'Bullish', 'strength': 'Medium', 'bullish': True})
                        score += 1
                else:
                    if histogram < 0 and len(df) > 2 and df['macd_histogram'].iloc[-2] > histogram:
                        signals.append({'indicator': 'MACD', 'signal': 'Strong Bearish Momentum', 'strength': 'Very Strong', 'bullish': False})
                        score -= 3
                    elif histogram < 0:
                        signals.append({'indicator': 'MACD', 'signal': 'Bearish Momentum', 'strength': 'Strong', 'bullish': False})
                        score -= 2
                    else:
                        signals.append({'indicator': 'MACD', 'signal': 'Bearish', 'strength': 'Medium', 'bullish': False})
                        score -= 1
            
            # Enhanced Moving Average Analysis
            if not pd.isna(latest['sma_7']) and not pd.isna(latest['sma_20']) and not pd.isna(latest['sma_50']):
                price = latest['close']
                sma7, sma20, sma50 = latest['sma_7'], latest['sma_20'], latest['sma_50']
                
                if price > sma7 > sma20 > sma50:
                    signals.append({'indicator': 'MA Alignment', 'signal': 'Perfect Bull Alignment', 'strength': 'Very Strong', 'bullish': True})
                    score += 3
                elif price > sma7 > sma20:
                    signals.append({'indicator': 'MA Trend', 'signal': 'Strong Uptrend', 'strength': 'Strong', 'bullish': True})
                    score += 2
                elif price > sma20:
                    signals.append({'indicator': 'MA Trend', 'signal': 'Uptrend', 'strength': 'Medium', 'bullish': True})
                    score += 1
                elif price < sma7 < sma20 < sma50:
                    signals.append({'indicator': 'MA Alignment', 'signal': 'Perfect Bear Alignment', 'strength': 'Very Strong', 'bullish': False})
                    score -= 3
                elif price < sma7 < sma20:
                    signals.append({'indicator': 'MA Trend', 'signal': 'Strong Downtrend', 'strength': 'Strong', 'bullish': False})
                    score -= 2
                elif price < sma20:
                    signals.append({'indicator': 'MA Trend', 'signal': 'Downtrend', 'strength': 'Medium', 'bullish': False})
                    score -= 1
            
            # Bollinger Bands with squeeze detection
            if all(col in df.columns for col in ['bb_upper', 'bb_lower', 'bb_middle', 'bb_width']):
                bb_position = (latest['close'] - latest['bb_lower']) / (latest['bb_upper'] - latest['bb_lower'])
                bb_width = latest['bb_width']
                
                if bb_width < 10:  # Bollinger Band squeeze
                    signals.append({'indicator': 'BB Squeeze', 'signal': 'Volatility Squeeze', 'strength': 'Medium', 'bullish': None})
                
                if latest['close'] > latest['bb_upper']:
                    signals.append({'indicator': 'Bollinger', 'signal': 'Strong Breakout', 'strength': 'Strong', 'bullish': True})
                    score += 1
                elif latest['close'] < latest['bb_lower']:
                    signals.append({'indicator': 'Bollinger', 'signal': 'Oversold', 'strength': 'Medium', 'bullish': True})
                    score += 1
                elif bb_position > 0.8:
                    signals.append({'indicator': 'Bollinger', 'signal': 'Near Upper Band', 'strength': 'Weak', 'bullish': False})
                elif bb_position < 0.2:
                    signals.append({'indicator': 'Bollinger', 'signal': 'Near Lower Band', 'strength': 'Weak', 'bullish': True})
            
            # Enhanced Volume Analysis
            if 'volume_ratio' in df.columns and not pd.isna(latest['volume_ratio']):
                vol_ratio = latest['volume_ratio']
                if vol_ratio > 2.5:
                    signals.append({'indicator': 'Volume', 'signal': 'Exceptional Volume Spike', 'strength': 'Very Strong', 'bullish': None})
                    score += 1  # High volume generally confirms trends
                elif vol_ratio > 1.5:
                    signals.append({'indicator': 'Volume', 'signal': 'High Volume Confirmation', 'strength': 'Strong', 'bullish': None})
                elif vol_ratio < 0.3:
                    signals.append({'indicator': 'Volume', 'signal': 'Very Low Volume', 'strength': 'Medium', 'bullish': None})
                    score -= 1  # Low volume weakens signals
            
            # Stochastic Analysis
            if not pd.isna(latest['stoch_k']) and not pd.isna(latest['stoch_d']):
                stoch_k, stoch_d = latest['stoch_k'], latest['stoch_d']
                
                if stoch_k < 20 and stoch_d < 20:
                    signals.append({'indicator': 'Stochastic', 'signal': 'Oversold', 'strength': 'Medium', 'bullish': True})
                    score += 1
                elif stoch_k > 80 and stoch_d > 80:
                    signals.append({'indicator': 'Stochastic', 'signal': 'Overbought', 'strength': 'Medium', 'bullish': False})
                    score -= 1
                elif stoch_k > stoch_d and stoch_k < 80:
                    signals.append({'indicator': 'Stochastic', 'signal': 'Bullish Cross', 'strength': 'Weak', 'bullish': True})
            
            # Volatility Analysis
            if 'volatility' in df.columns and not pd.isna(latest['volatility']):
                current_vol = latest['volatility']
                avg_vol = df['volatility'].mean()
                
                if current_vol > avg_vol * 2:
                    signals.append({'indicator': 'Volatility', 'signal': 'High Volatility Warning', 'strength': 'Medium', 'bullish': None})
                elif current_vol < avg_vol * 0.5:
                    signals.append({'indicator': 'Volatility', 'signal': 'Low Volatility - Breakout Coming', 'strength': 'Medium', 'bullish': None})
            
            # Final recommendation with improved confidence calculation
            confidence_base = 50
            confidence_modifier = min(abs(score) * 8, 45)  # Max 45% modifier
            
            if score >= 5:
                recommendation = 'STRONG BUY'
                confidence = confidence_base + confidence_modifier
            elif score >= 3:
                recommendation = 'BUY'
                confidence = confidence_base + confidence_modifier * 0.8
            elif score >= 1:
                recommendation = 'WEAK BUY'
                confidence = confidence_base + confidence_modifier * 0.6
            elif score <= -5:
                recommendation = 'STRONG SELL'
                confidence = confidence_base + confidence_modifier
            elif score <= -3:
                recommendation = 'SELL'
                confidence = confidence_base + confidence_modifier * 0.8
            elif score <= -1:
                recommendation = 'WEAK SELL'
                confidence = confidence_base + confidence_modifier * 0.6
            else:
                recommendation = 'HOLD'
                confidence = confidence_base
            
            return {
                'signals': signals,
                'score': score,
                'recommendation': recommendation,
                'confidence': min(confidence, 95),  # Cap at 95%
                'rsi': latest['rsi'] if not pd.isna(latest['rsi']) else 50,
                'macd_signal': 'Bullish' if not pd.isna(latest['macd']) and not pd.isna(latest['macd_signal']) and latest['macd'] > latest['macd_signal'] else 'Bearish'
            }
            
        except Exception as e:
            st.error(f"Error in signal analysis: {e}")
            return {'error': 'Analysis failed'}
    
    def create_price_chart(self, df: pd.DataFrame, coin_name: str) -> go.Figure:
        """Create comprehensive price chart with enhanced visuals"""
        fig = make_subplots(
            rows=4, cols=1,
            subplot_titles=(
                f'{coin_name} Price & Technical Analysis', 
                'RSI (14) - Momentum Oscillator', 
                'MACD - Trend Following',
                'Volume & Volatility'
            ),
            vertical_spacing=0.06,
            row_heights=[0.5, 0.15, 0.15, 0.2]
        )
        
        # Candlestick chart (if we have OHLC data)
        if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
            fig.add_trace(
                go.Candlestick(
                    x=df.index,
                    open=df['open'],
                    high=df['high'],
                    low=df['low'],
                    close=df['close'],
                    name='Price',
                    increasing_line_color='#00ff88',
                    decreasing_line_color='#ff4444'
                ),
                row=1, col=1
            )
        else:
            # Fallback to line chart
            fig.add_trace(
                go.Scatter(x=df.index, y=df['close'], name='Price', 
                          line=dict(color='white', width=2)),
                row=1, col=1
            )
        
        # Moving averages
        if 'sma_7' in df.columns:
            fig.add_trace(
                go.Scatter(x=df.index, y=df['sma_7'], name='SMA 7', 
                          line=dict(color='orange', width=1.5)),
                row=1, col=1
            )
        
        if 'sma_20' in df.columns:
            fig.add_trace(
                go.Scatter(x=df.index, y=df['sma_20'], name='SMA 20', 
                          line=dict(color='#4da6ff', width=1.5)),
                row=1, col=1
            )
        
        if 'sma_50' in df.columns:
            fig.add_trace(
                go.Scatter(x=df.index, y=df['sma_50'], name='SMA 50', 
                          line=dict(color='purple', width=1.5)),
                row=1, col=1
            )
        
        # Bollinger Bands with fill
        if all(col in df.columns for col in ['bb_upper', 'bb_lower']):
            fig.add_trace(
                go.Scatter(x=df.index, y=df['bb_upper'], name='BB Upper',
                          line=dict(color='gray', dash='dash', width=1), 
                          fill=None, opacity=0.3),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(x=df.index, y=df['bb_lower'], name='BB Lower',
                          line=dict(color='gray', dash='dash', width=1),
                          fill='tonexty', fillcolor='rgba(128,128,128,0.1)', opacity=0.3),
                row=1, col=1
            )
        
        # RSI with colored zones
        if 'rsi' in df.columns:
            fig.add_trace(
                go.Scatter(x=df.index, y=df['rsi'], name='RSI', 
                          line=dict(color='purple', width=2)),
                row=2, col=1
            )
            
            # RSI zones
            fig.add_hrect(y0=70, y1=100, fillcolor="red", opacity=0.2, row=2, col=1)
            fig.add_hrect(y0=0, y1=30, fillcolor="green", opacity=0.2, row=2, col=1)
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
            fig.add_hline(y=50, line_dash="dot", line_color="gray", row=2, col=1)
        
        # MACD with improved visualization
        if all(col in df.columns for col in ['macd', 'macd_signal', 'macd_histogram']):
            fig.add_trace(
                go.Scatter(x=df.index, y=df['macd'], name='MACD', 
                          line=dict(color='blue', width=2)),
                row=3, col=1
            )
            fig.add_trace(
                go.Scatter(x=df.index, y=df['macd_signal'], name='Signal', 
                          line=dict(color='red', width=2)),
                row=3, col=1
            )
            
            # MACD Histogram with dynamic colors
            colors = ['#00ff88' if val > 0 else '#ff4444' for val in df['macd_histogram']]
            fig.add_trace(
                go.Bar(x=df.index, y=df['macd_histogram'], name='Histogram',
                      marker_color=colors, opacity=0.7),
                row=3, col=1
            )
            fig.add_hline(y=0, line_color="gray", row=3, col=1)
        
        # Volume with volatility overlay
        if 'volume' in df.columns:
            fig.add_trace(
                go.Bar(x=df.index, y=df['volume'], name='Volume',
                      marker_color='rgba(0,150,255,0.6)'),
                row=4, col=1
            )
            
            # Add volatility as a secondary y-axis line on volume chart
            if 'volatility' in df.columns:
                fig.add_trace(
                    go.Scatter(x=df.index, y=df['volatility'], name='Volatility %',
                              line=dict(color='yellow', width=1.5), yaxis='y2'),
                    row=4, col=1
                )
        
        # Update layout with better styling
        fig.update_layout(
            template='plotly_dark',
            height=900,
            showlegend=True,
            title_x=0.5,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        # Update y-axis for volatility
        fig.update_yaxes(title_text="Volatility %", secondary_y=True, row=4, col=1)
        fig.update_xaxes(rangeslider_visible=False)
        
        return fig

# Predefined coins list - no API calls needed for coin selection
PREDEFINED_COINS = {
    'Bitcoin': 'bitcoin',
    'Ethereum': 'ethereum',
    'Tether': 'tether',
    'BNB': 'binancecoin',
    'Solana': 'solana',
    'USDC': 'usd-coin',
    'XRP': 'ripple',
    'Lido Staked Ether': 'staked-ether',
    'Toncoin': 'the-open-network',
    'Dogecoin': 'dogecoin',
    'Cardano': 'cardano',
    'TRON': 'tron',
    'Avalanche': 'avalanche-2',
    'Wrapped Bitcoin': 'wrapped-bitcoin',
    'Shiba Inu': 'shiba-inu',
    'Chainlink': 'chainlink',
    'Bitcoin Cash': 'bitcoin-cash',
    'Polkadot': 'polkadot',
    'Polygon': 'matic-network',
    'Litecoin': 'litecoin',
    'Near Protocol': 'near',
    'Uniswap': 'uniswap',
    'Internet Computer': 'internet-computer',
    'Pepe': 'pepe',
    'Dai': 'dai',
    'Ethereum Classic': 'ethereum-classic',
    'Aptos': 'aptos',
    'Fetch.ai': 'fetch-ai',
    'Monero': 'monero',
    'Stellar': 'stellar',
    'Arbitrum': 'arbitrum',
    'VeChain': 'vechain',
    'Filecoin': 'filecoin',
    'Cosmos': 'cosmos',
    'Hedera': 'hedera-hashgraph',
    'Cronos': 'crypto-com-chain',
    'OKB': 'okb',
    'Algorand': 'algorand',
    'Optimism': 'optimism',
    'Fantom': 'fantom',
    'Theta Network': 'theta-token',
    'The Graph': 'the-graph',
    'Aave': 'aave',
    'Quant': 'quant-network',
    'Flow': 'flow',
    'MultiversX': 'elrond-erd-2',
    'ImmutableX': 'immutable-x',
    'Tezos': 'tezos',
    'Sandbox': 'the-sandbox',
    'Axie Infinity': 'axie-infinity',
    'Bitcoin SV': 'bitcoin-cash-sv',
    'Chiliz': 'chiliz',
    'Decentraland': 'decentraland',
    'EOS': 'eos',
    'Klaytn': 'klay-token',
    'IOTA': 'iota',
    'Maker': 'maker',
    'Compound': 'compound-governance-token',
    'Zcash': 'zcash',
    'Synthetix': 'havven',
    'Neo': 'neo',
    'Dash': 'dash',
    'UNUS SED LEO': 'leo-token',
    'Curve DAO Token': 'curve-dao-token',
    'Amp': 'amp-token',
    'Pancakeswap': 'pancakeswap-token',
    'Yearn Finance': 'yearn-finance',
    'Sushi': 'sushi',
    '1inch': '1inch',
    'Basic Attention Token': 'basic-attention-token',
    'Enjin Coin': 'enjincoin',
    'Loopring': 'loopring',
    'Zilliqa': 'zilliqa',
    'OMG Network': 'omisego',
    'Qtum': 'qtum',
    'Waves': 'waves',
    'Gala': 'gala',
    'Harmony': 'harmony',
    'Celo': 'celo',
    'Ren': 'republic-protocol',
    'Storj': 'storj',
    'Ocean Protocol': 'ocean-protocol',
    'Numeraire': 'numeraire',
    'Bancor': 'bancor',
    'SKALE Network': 'skale',
    'Livepeer': 'livepeer',
    'Origin Protocol': 'origin-protocol',
    'Balancer': 'balancer',
    'API3': 'api3',
    'Rally': 'rally-2',
    'Keep Network': 'keep-network',
    'NuCypher': 'nucypher',
    'Ankr': 'ankr',
    'Civic': 'civic',
    'District0x': 'district0x',
    'Request': 'request-network',
    'Power Ledger': 'power-ledger',
    'Metal': 'metal',
    'Status': 'status',
    'Golem': 'golem',
    'Aragon': 'aragon',
    'Gnosis': 'gnosis',
    'Augur': 'augur-v2',
    'iExec RLC': 'iexec-rlc',
    'Polymath': 'polymath-network',
    'Substratum': 'substratum',
    'Render Token': 'render-token',
    'Jupiter': 'jupiter-exchange-solana',
    'Pyth Network': 'pyth-network',
    'Jito': 'jito-governance-token',
    'Bonk': 'bonk',
    'Manta Network': 'manta-network',
    'JasmyCoin': 'jasmycoin',
    'Sei': 'sei-network',
    'Blur': 'blur',
    'Meme': 'memecoin',
    'Popcat': 'popcat',
    'Worldcoin': 'worldcoin-wld',
    'Sui': 'sui',
    'Mantle': 'mantle',
    'Celestia': 'celestia',
    'Injective': 'injective-protocol',
    'Brett': 'based-brett',
    'Ondo': 'ondo-finance',
    'Wormhole': 'wormhole',
    'Starknet': 'starknet',
    'Immutable': 'immutable-x',
    'NotCoin': 'notcoin',
    'THORChain': 'thorchain',
    'Floki': 'floki',
    'Kaspa': 'kaspa',
    'dYdX': 'dydx-chain',
    'Ethena': 'ethena',
    'Arweave': 'arweave',
    'LayerZero': 'layerzero'
}

def main():
    # App header with enhanced info
    st.title("Advanced Crypto Trading Analysis")
    st.markdown("*Professional cryptocurrency market analysis powered by CoinGecko API*")
    
    # Initialize bot
    bot = OptimizedCoinGeckoBot()
    
    # Show API status
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("API Source", "CoinGecko")
    with col2:
        st.metric("Rate Limit", "10-30/min")
    with col3:
        st.metric("Free Tier", "Full OHLC Data")
    
    # Sidebar controls
    st.sidebar.header("Analysis Controls")
    
    # Use predefined coins - no API calls needed
    all_coin_names = sorted(list(PREDEFINED_COINS.keys()))
    
    # Local search through predefined coins
    search_query = st.sidebar.text_input(
        "Search Cryptocurrencies", 
        placeholder="Type coin name (e.g., Bitcoin, Ethereum)",
        value=""
    ).strip()
    
    # Filter predefined coins locally
    if search_query:
        search_lower = search_query.lower()
        filtered_coins = [name for name in all_coin_names if search_lower in name.lower()]
        
        if filtered_coins:
            selected_coin_name = st.sidebar.selectbox(
                f"Select from {len(filtered_coins)} matches",
                filtered_coins,
                index=0
            )
        else:
            st.sidebar.warning("No coins found in predefined list.")
            selected_coin_name = None
    else:
        selected_coin_name = st.sidebar.selectbox(
            "Select Cryptocurrency",
            all_coin_names,
            index=0
        )

    coin_id = PREDEFINED_COINS.get(selected_coin_name) if selected_coin_name else None

    # Enhanced analysis controls
    st.sidebar.markdown("---")
    analysis_days = st.sidebar.selectbox(
        "Analysis Period",
        [7, 14, 30, 90, 180],
        index=2,
        help="Longer periods provide more reliable signals but slower loading"
    )
    
    # Analysis button
    analyze_button = st.sidebar.button(
        "Analyze Now", 
        type="primary", 
        disabled=(coin_id is None),
        help="Start comprehensive technical analysis"
    )
    
    # Auto-refresh with better UX
    auto_refresh = st.sidebar.checkbox("Auto-refresh (60s)")
    if auto_refresh:
        st.sidebar.info("Auto-refresh enabled - data will update every minute")
    
    # Main analysis logic with improved state management
    trigger_analysis = False
    
    # Check if we should trigger analysis
    if analyze_button:
        trigger_analysis = True
    elif ('current_analysis' not in st.session_state and coin_id and 
          st.sidebar.button("Quick Analysis", help="Fast analysis with cached data")):
        trigger_analysis = True
    elif ('current_analysis' in st.session_state and coin_id and 
          st.session_state.current_analysis.get('coin_id') != coin_id):
        trigger_analysis = True
    
    # Auto-refresh logic
    if (auto_refresh and 'current_analysis' in st.session_state and 
        datetime.now() - st.session_state.current_analysis.get('last_update', datetime.now()) > timedelta(minutes=1)):
        trigger_analysis = True
        time.sleep(1)  # Brief pause for auto-refresh

    if trigger_analysis and coin_id:
        # Progressive loading with status updates
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        with st.spinner(f'Analyzing {selected_coin_name}...'):
            try:
                # Step 1: Get comprehensive data (optimized batch call)
                status_text.text("Fetching market data...")
                progress_bar.progress(25)
                
                df, price_info = bot.get_comprehensive_coin_data(coin_id, days=analysis_days)
                
                if not price_info or df.empty:
                    st.error("Failed to fetch coin data. Please try again or search for a different coin.")
                    progress_bar.empty()
                    status_text.empty()
                    return
                
                # Step 2: Calculate technical indicators
                status_text.text("Calculating technical indicators...")
                progress_bar.progress(60)
                
                df = bot.calculate_technical_indicators(df)
                
                # Step 3: Analyze signals
                status_text.text("Analyzing market signals...")
                progress_bar.progress(80)
                
                analysis = bot.analyze_signals(df)
                
                if 'error' in analysis:
                    st.error(f"Analysis failed: {analysis['error']}")
                    progress_bar.empty()
                    status_text.empty()
                    return
                
                # Step 4: Store results
                status_text.text("Analysis complete!")
                progress_bar.progress(100)
                
                st.session_state.current_analysis = {
                    'price_info': price_info,
                    'df': df,
                    'analysis': analysis,
                    'coin_name': selected_coin_name,
                    'coin_id': coin_id,
                    'analysis_period': analysis_days,
                    'last_update': datetime.now()
                }
                
                # Clear progress indicators
                time.sleep(0.5)
                progress_bar.empty()
                status_text.empty()
                
            except Exception as e:
                st.error(f"Analysis failed: {str(e)}")
                progress_bar.empty()
                status_text.empty()
                return
    
    # Display results
    if 'current_analysis' not in st.session_state or st.session_state.current_analysis is None:
        st.info("Search for a coin or select from the dropdown, then click 'Analyze Now' to begin comprehensive analysis.")
        
        # Show some predefined coins as quick options
        st.subheader("Popular Cryptocurrencies")
        popular_selection = ['Bitcoin', 'Ethereum', 'Solana', 'Cardano', 'Polygon']
        cols = st.columns(5)
        for i, coin in enumerate(popular_selection):
            with cols[i]:
                if st.button(f"{coin}", key=f"popular_{i}"):
                    st.session_state.selected_coin_name = coin
                    st.session_state.selected_coin_id = PREDEFINED_COINS[coin]
                    st.rerun()
        return
    
    # Extract analysis data
    data = st.session_state.current_analysis
    price_info = data['price_info']
    df = data['df']
    analysis = data['analysis']
    coin_name = data['coin_name']
    
    # Success message with timestamp
    st.success(f"Analysis complete for {coin_name} ({price_info['symbol']})")
    col1, col2 = st.columns([3, 1])
    with col1:
        st.caption(f"Last updated: {data['last_update'].strftime('%Y-%m-%d %H:%M:%S')} | Period: {data['analysis_period']} days")
    with col2:
        if st.button("Refresh Analysis"):
            st.cache_data.clear()
            st.rerun()
    
    # Enhanced metrics row
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    with col1:
        st.metric(
            "Current Price", 
            f"${price_info['current_price']:,.6f}" if price_info['current_price'] < 1 else f"${price_info['current_price']:,.2f}",
            delta=f"{price_info['price_change_24h']:.2f}%"
        )
    
    with col2:
        st.metric(
            "Rank", 
            f"#{price_info['market_cap_rank']}",
            help="Market cap ranking"
        )
    
    with col3:
        st.metric(
            "24h High", 
            f"${price_info['high_24h']:,.6f}" if price_info['high_24h'] < 1 else f"${price_info['high_24h']:,.2f}"
        )
    
    with col4:
        st.metric(
            "24h Low", 
            f"${price_info['low_24h']:,.6f}" if price_info['low_24h'] < 1 else f"${price_info['low_24h']:,.2f}"
        )
    
    with col5:
        st.metric(
            "Volume", 
            f"${price_info['volume_24h']:,.0f}"
        )
    
    with col6:
        if 'price_change_7d' in price_info:
            st.metric(
                "7d Change",
                f"{price_info['price_change_7d']:.2f}%"
            )
    
    # Enhanced recommendation display
    rec = analysis['recommendation']
    conf = analysis['confidence']
    score = analysis['score']
    
    # Color-coded recommendation with score
    if rec in ['BUY', 'STRONG BUY', 'WEAK BUY']:
        st.success(f"**{rec}** (Confidence: {conf}% | Score: +{score})")
    elif rec in ['SELL', 'STRONG SELL', 'WEAK SELL']:
        st.error(f"**{rec}** (Confidence: {conf}% | Score: {score})")
    else:
        st.warning(f"**{rec}** (Confidence: {conf}% | Score: {score})")
    
    # Add risk assessment
    if 'volatility' in df.columns:
        avg_volatility = df['volatility'].mean()
        if avg_volatility > 15:
            st.warning("**High Risk Asset** - Significant price volatility detected")
        elif avg_volatility < 5:
            st.info("**Lower Risk Asset** - Relatively stable price movements")
    
    # Technical Analysis Chart
    st.header("Technical Analysis Chart")
    
    fig = bot.create_price_chart(df, coin_name)
    st.plotly_chart(fig, use_container_width=True)
    
    # Enhanced Analysis Layout
    col1, col2 = st.columns([1.2, 0.8])
    
    with col1:
        st.subheader("Signal Analysis")
        
        # Group signals by strength for better readability
        strong_signals = [s for s in analysis['signals'] if s['strength'] in ['Very Strong', 'Strong']]
        medium_signals = [s for s in analysis['signals'] if s['strength'] == 'Medium']
        weak_signals = [s for s in analysis['signals'] if s['strength'] == 'Weak']
        
        if strong_signals:
            st.markdown("**Strong Signals:**")
            for signal in strong_signals:
                if signal['bullish'] is True:
                    st.success(f"**{signal['indicator']}**: {signal['signal']}")
                elif signal['bullish'] is False:
                    st.error(f"**{signal['indicator']}**: {signal['signal']}")
                else:
                    st.info(f"**{signal['indicator']}**: {signal['signal']}")
        
        if medium_signals:
            st.markdown("**Medium Signals:**")
            for signal in medium_signals:
                if signal['bullish'] is True:
                    st.success(f"{signal['indicator']}: {signal['signal']}")
                elif signal['bullish'] is False:
                    st.error(f"{signal['indicator']}: {signal['signal']}")
                else:
                    st.info(f"{signal['indicator']}: {signal['signal']}")
        
        if weak_signals:
            with st.expander("Weak Signals"):
                for signal in weak_signals:
                    if signal['bullish'] is True:
                        st.success(f"{signal['indicator']}: {signal['signal']}")
                    elif signal['bullish'] is False:
                        st.error(f"{signal['indicator']}: {signal['signal']}")
                    else:
                        st.info(f"{signal['indicator']}: {signal['signal']}")
    
    with col2:
        st.subheader("Pattern Recognition")
        
        patterns = bot.detect_chart_patterns(df)
        
        if patterns:
            for pattern in patterns:
                if pattern['signal'] == 'Bullish':
                    st.success(f"**{pattern['type']}**")
                    st.caption(pattern['description'])
                else:
                    st.error(f"**{pattern['type']}**")
                    st.caption(pattern['description'])
        else:
            st.info("No significant patterns detected")
        
        # Add additional market info
        if price_info.get('ath') and price_info.get('atl'):
            st.markdown("### Price Levels")
            current = price_info['current_price']
            ath = price_info['ath']
            atl = price_info['atl']
            
            ath_distance = ((ath - current) / current) * 100
            atl_distance = ((current - atl) / atl) * 100
            
            st.metric("All-Time High", f"${ath:,.2f}", f"{ath_distance:.1f}% away")
            st.metric("All-Time Low", f"${atl:.6f}" if atl < 1 else f"${atl:,.2f}", f"+{atl_distance:.1f}% above")
    
    # Enhanced Market Insights
    st.header("Trading Insights & Recommendations")
    
    insights = []
    
    # RSI insights with multiple levels
    rsi = analysis['rsi']
    if rsi < 25:
        insights.append("**Extremely Oversold** - Strong buying opportunity, but wait for confirmation")
    elif rsi < 30:
        insights.append("**RSI Oversold** - Potential buying opportunity")
    elif rsi > 75:
        insights.append("**Extremely Overbought** - High risk, consider taking profits")
    elif rsi > 70:
        insights.append("**RSI Overbought** - Consider reducing position size")
    
    # Volume insights
    if 'volume_ratio' in df.columns and not pd.isna(df['volume_ratio'].iloc[-1]):
        vol_ratio = df['volume_ratio'].iloc[-1]
        if vol_ratio > 2.5:
            insights.append("**Exceptional volume spike** - Major market interest, trend likely to continue")
        elif vol_ratio > 1.5:
            insights.append("**High volume confirmation** - Current trend has strong support")
        elif vol_ratio < 0.5:
            insights.append("**Low volume warning** - Weak trend, be cautious of reversals")
    
    # Volatility insights
    if 'volatility' in df.columns:
        current_volatility = df['volatility'].iloc[-1] if not pd.isna(df['volatility'].iloc[-1]) else 0
        avg_volatility = df['volatility'].mean()
        
        if current_volatility > avg_volatility * 2:
            insights.append("**Extreme volatility** - Use smaller position sizes and tighter stops")
        elif current_volatility > avg_volatility * 1.5:
            insights.append("**High volatility** - Expect larger price swings")
        elif current_volatility < avg_volatility * 0.5:
            insights.append("**Low volatility** - Possible breakout incoming, watch for volume spike")
    
    # Trend strength insights
    if analysis['score'] >= 5:
        insights.append("**Very strong bullish convergence** - Multiple timeframes align bullishly")
    elif analysis['score'] >= 3:
        insights.append("**Bullish signals dominating** - Upward momentum building")
    elif analysis['score'] <= -5:
        insights.append("**Very strong bearish convergence** - Multiple indicators suggest decline")
    elif analysis['score'] <= -3:
        insights.append("**Bearish signals present** - Downward pressure building")
    
    # MACD insights
    if analysis.get('macd_signal') == 'Bullish' and not pd.isna(df['macd_histogram'].iloc[-1]):
        if df['macd_histogram'].iloc[-1] > 0:
            insights.append("**MACD bullish momentum** - Trend acceleration likely")
    
    # Display insights
    if insights:
        for i, insight in enumerate(insights):
            st.markdown(insight)
            if i < len(insights) - 1:
                st.markdown("---")
    else:
        st.info("Market showing balanced conditions - Monitor for clearer directional signals")
    
    # Enhanced Technical Indicators Table
    st.header("Current Technical Indicators")
    
    latest = df.iloc[-1]
    
    # Create two columns for better organization
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Momentum Indicators**")
        momentum_data = {
            'Indicator': ['RSI (14)', 'Stochastic %K', 'Stochastic %D', 'MACD'],
            'Value': [
                f"{latest['rsi']:.1f}" if not pd.isna(latest['rsi']) else 'N/A',
                f"{latest['stoch_k']:.1f}" if not pd.isna(latest['stoch_k']) else 'N/A',
                f"{latest['stoch_d']:.1f}" if not pd.isna(latest['stoch_d']) else 'N/A',
                f"{latest['macd']:.6f}" if not pd.isna(latest['macd']) else 'N/A'
            ],
            'Signal': [
                'Oversold' if not pd.isna(latest['rsi']) and latest['rsi'] < 30 else ('Overbought' if not pd.isna(latest['rsi']) and latest['rsi'] > 70 else 'Neutral'),
                'Oversold' if not pd.isna(latest['stoch_k']) and latest['stoch_k'] < 20 else ('Overbought' if not pd.isna(latest['stoch_k']) and latest['stoch_k'] > 80 else 'Neutral'),
                'Oversold' if not pd.isna(latest['stoch_d']) and latest['stoch_d'] < 20 else ('Overbought' if not pd.isna(latest['stoch_d']) and latest['stoch_d'] > 80 else 'Neutral'),
                'Bullish' if not pd.isna(latest['macd']) and not pd.isna(latest['macd_signal']) and latest['macd'] > latest['macd_signal'] else 'Bearish'
            ]
        }
        st.dataframe(pd.DataFrame(momentum_data), use_container_width=True)
    
    with col2:
        st.markdown("**Trend Indicators**")
        trend_data = {
            'Indicator': ['SMA 7', 'SMA 20', 'SMA 50', 'BB Position'],
            'Value': [
                f"${latest['sma_7']:.4f}" if not pd.isna(latest['sma_7']) else 'N/A',
                f"${latest['sma_20']:.4f}" if not pd.isna(latest['sma_20']) else 'N/A',
                f"${latest['sma_50']:.4f}" if not pd.isna(latest['sma_50']) else 'N/A',
                f"{((latest['close'] - latest['bb_lower']) / (latest['bb_upper'] - latest['bb_lower']) * 100):.1f}%" if all(not pd.isna(latest[col]) for col in ['bb_upper', 'bb_lower']) else 'N/A'
            ],
            'Signal': [
                'Above' if not pd.isna(latest['close']) and not pd.isna(latest['sma_7']) and latest['close'] > latest['sma_7'] else 'Below',
                'Above' if not pd.isna(latest['close']) and not pd.isna(latest['sma_20']) and latest['close'] > latest['sma_20'] else 'Below',
                'Above' if not pd.isna(latest['close']) and not pd.isna(latest['sma_50']) and latest['close'] > latest['sma_50'] else 'Below',
                'Upper' if not pd.isna(latest['close']) and not pd.isna(latest['bb_upper']) and latest['close'] > latest['bb_upper'] else ('Lower' if not pd.isna(latest['close']) and not pd.isna(latest['bb_lower']) and latest['close'] < latest['bb_lower'] else 'Middle')
            ]
        }
        st.dataframe(pd.DataFrame(trend_data), use_container_width=True)
    
    # Enhanced footer with performance tips
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### How to Read the Analysis")
        st.markdown("""
        **Signal Strength:**
        - **Very Strong/Strong:** High confidence signals
        - **Medium:** Moderate confidence, confirm with other indicators  
        - **Weak:** Low confidence, use as supporting evidence only
        
        **Key Indicators:**
        - **RSI:** < 30 = Oversold, > 70 = Overbought
        - **MACD:** Above signal = Bullish momentum
        - **Moving Averages:** Price position indicates trend direction
        """)
    
    with col2:
        st.markdown("### Performance Info")
        st.markdown(f"""
        **Data Freshness:** {(datetime.now() - data['last_update']).seconds // 60} minutes ago  
        **Analysis Period:** {data['analysis_period']} days  
        **API Calls:** Optimized for rate limits  
        **Cache Status:** {len(df)} data points loaded  
        **Coins Available:** {len(PREDEFINED_COINS)} predefined coins  
        """)
        
        if auto_refresh:
            next_refresh = 60 - (datetime.now() - data['last_update']).seconds
            if next_refresh > 0:
                st.info(f"Auto-refresh in {next_refresh} seconds")
    
    st.markdown("*Disclaimer: This analysis is for educational purposes only. Always conduct your own research and consider multiple sources before making trading decisions.*")

# Enhanced Sidebar Info
with st.sidebar:
    st.markdown("---")
    st.markdown("### About This Bot")
    st.markdown("""
    **Data Source:** CoinGecko API (Free Tier)  
    **Optimization:** Predefined coins + caching  
    **Update Frequency:** 15 minutes  
    **Analysis:** 8+ Technical Indicators  
    **Patterns:** Support/Resistance + Trends  
    """)
    
    st.markdown("### Performance Features")
    st.markdown("""
    - **Predefined coin list** (no API calls for search)
    - **Smart caching** (15min data cache)
    - **Rate limit protection** (2.5s delays)
    - **Retry logic** (exponential backoff)
    - **Progressive loading** (real-time status)
    - **Local search** (instant filtering)
    """)
    
    # API Status indicator
    st.markdown("### API Status")
    if 'current_analysis' in st.session_state:
        st.success("Connected")
        st.caption("Real-time data flowing")
    else:
        st.info("Ready")
        st.caption("Select a coin to begin")

if __name__ == "__main__":
    main()
