import os
import sys
import logging
import time
import pandas as pd
from typing import Optional
from iqoptionapi.stable_api import IQ_Option
from flask import Flask
from threading import Thread

app = Flask('')

@app.route('/')
def home():
    return "Server is running!"

def run():
    app.run(host='0.0.0.0',port=8080)

def server_on():
    t = Thread(target=run)
    t.start()

def get_env_data():

    # ดึงทุก environment variables ที่มีอยู่ในระบบ
    env_data = dict(os.environ)
    
    # แปลงค่า LOG_ เป็น boolean ถ้ามี
    if 'LOG_' in env_data:
        env_data['LOG_'] = True if env_data['LOG_'].lower() == 'true' else False

    return env_data

class IQOptionTrader:
    """A class to manage trading operations with IQ Option"""
    
    def __init__(self, SELECT_ASSET="EURUSD"):
        self._setup_logging(SELECT_ASSET)
        self.config = self._load_config()
        self.config.select_asset = SELECT_ASSET
        self.amount: int = self.config.start_bet
        self.mm: int = 0
        self.direction: str = "call"
        self.account: Optional[IQ_Option] = None
        self._initialize_connection()

        # Candle size (1 minute)
        self.candle_size: int = 60

        self.TempLose = 0
        self.MaxLose = 0
        self.gameWin = 0
        self.gameLose = 0

    def _setup_logging(self, asset_name: str) -> None:
        """Configure logging settings with file output per asset"""
        log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        # Clear previous handlers to prevent duplicate logs
        if self.logger.hasHandlers():
            self.logger.handlers.clear()

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(log_formatter)
        self.logger.addHandler(console_handler)

        # File handler
        log_file_name = f"{asset_name}.log".replace("/", "_").replace(":", "_")
        file_handler = logging.FileHandler(f"log/{log_file_name}")
        file_handler.setFormatter(log_formatter)
        self.logger.addHandler(file_handler)

    def _load_config(self) -> dict:
        """Load and validate configuration from environment"""
        try:
            env_data = get_env_data()
            required_fields = {
                'API_USERNAME': str,
                'API_PASSWORD': str,
                'START_BET': int,
                'MAX_MARTINGEL': int,
                # 'SELECT_ASSET': str,
                'EXPIRATION': int,
                'API_BALANCE': str
            }
            
            config = {}
            for field, field_type in required_fields.items():
                if field not in env_data:
                    raise ValueError(f"Missing required config field: {field}")
                config[field.lower()] = field_type(env_data[field])
                
            return type('Config', (), config)()
        except (ValueError, TypeError) as e:
            self.logger.error(f"Configuration error: {str(e)}")

    def _initialize_connection(self) -> None:
        """Initialize and verify connection to IQ Option"""
        try:
            self.account = IQ_Option(self.config.api_username, self.config.api_password)
            connected, reason = self.account.connect()
            if not connected:
                self.logger.info(f"Connection failed: {reason}")
            
            self.logger.info("Successfully connected to IQ Option")
            self._switch_to_practice_account()
            
        except Exception as e:
            self.logger.error(f"Connection initialization failed: {str(e)}")

    def _switch_to_practice_account(self) -> None:
        """Switch to practice account and log balance"""
        try:
            self.account.change_balance(self.config.api_balance)
            balance = self.account.get_balance()
            self.logger.info(f"Switched to practice account. Current balance: {balance} USD")
        except Exception as e:
            self.logger.error(f"Failed to switch account: {str(e)}")

    def _get_price_data(self) -> pd.DataFrame:
        """Fetch historical price data for analysis and return it as a DataFrame"""
        try:
            # Get enough candles for the slow MA (26) + signal (9)
            candles = self.account.get_candles(
                self.config.select_asset,
                self.candle_size,
                200,
                time.time()
            )
            
            # Convert the candles into a DataFrame
            price_data = pd.DataFrame([
                {
                    'open': candle['open'],
                    'close': candle['close'],
                    'low': candle['min'],
                    'high': candle['max']
                }
                for candle in candles
            ])
            
            # Ensure that the data is sorted by time in ascending order
            price_data['id'] = pd.to_datetime([candle['id'] for candle in candles], unit='s')
            price_data.set_index('id', inplace=True)

            return price_data

        except Exception as e:
            self.logger.error(f"Failed to fetch price data: {str(e)}")
            return pd.DataFrame()  # Return empty DataFrame if there's an error
        
    def _analyze_market(self) -> Optional[str]:
        """Analyze market conditions using Stochastic RSI strategy"""
        try:
            price_data = self._get_price_data()

            if price_data.empty or len(price_data) < 2:
                self.logger.error(f"{self.config.select_asset} Price data is empty or too short.")
                return None

            if not all(col in price_data.columns for col in ['low', 'high']):
                self.logger.error(f"{self.config.select_asset} Missing required columns: 'low', 'high'")
                return None

            # Calculate Bollinger Bands
            price_data['SMA'] = price_data['close'].rolling(window=20).mean()
            price_data['STD'] = price_data['close'].rolling(window=20).std()
            price_data['Upper_BB'] = price_data['SMA'] + 2 * price_data['STD']
            price_data['Lower_BB'] = price_data['SMA'] - 2 * price_data['STD']

            # Calculate EMA
            price_data['ema_cross'] = price_data['close'].ewm(span=12, adjust=False).mean()
            price_data['ema_trend'] = price_data['close'].ewm(span=50, adjust=False).mean()
            price_data['ema_base'] = price_data['close'].ewm(span=200, adjust=False).mean()

            price_data = price_data.dropna()
            if price_data.empty or len(price_data) < 2:
                self.logger.error(f"{self.config.select_asset} Not enough valid data after indicators.")
                return None

            latest_row = price_data.iloc[-1]
            second_last_row = price_data.iloc[-2]
            third_last_row = price_data.iloc[-2]

            trendUp = second_last_row['ema_base'] < second_last_row['ema_trend'] < second_last_row['ema_cross']
            trendDn = second_last_row['ema_base'] > second_last_row['ema_trend'] > second_last_row['ema_cross']
            priceIn = second_last_row['low'] > second_last_row['Lower_BB'] and second_last_row['high'] < second_last_row['Upper_BB']

            crossOver = third_last_row['ema_cross'] < third_last_row['ema_trend'] and second_last_row['ema_cross'] > second_last_row['ema_trend']
            crossUnder = third_last_row['ema_cross'] > third_last_row['ema_trend'] and second_last_row['ema_cross'] < second_last_row['ema_trend']

            Signal_buy = trendUp and priceIn and crossOver
            Signal_sell = trendDn and priceIn and crossUnder

            # print(f"{self.config.select_asset} Signal: Buy={Signal_buy}, Sell={Signal_sell}")
            # self.logger.info(f"{self.config.select_asset} Signal: Buy={Signal_buy}, Sell={Signal_sell}")

            if latest_row['close'] > latest_row['open']:
                return 'call'
            elif latest_row['close'] < latest_row['open']:
                return 'put'
            # if Signal_buy:
            #     return 'call'
            # elif Signal_sell:
            #     return 'put'
            else:
                return None

        except Exception as e:
            self.logger.error(f"{self.config.select_asset} Market analysis failed: {str(e)}")
            return None

    def create_order(self) -> None:
        """Create and manage a trading order after market analysis"""
        try:
            # print(f"Account: {self.config.api_balance} {self.account.get_balance()}$ {self.config.select_asset}")
            direction = self._analyze_market()
            if not direction:
                # self.logger.info("No clear trading signal, skipping order")
                return
            
            self.logger.info(
                f"Account: {self.config.api_balance} {self.account.get_balance()}$ {self.config.select_asset} "
                f"Win:{self.gameWin}, Lose:{self.gameLose}, MaxLose:{self.MaxLose}"
            )

            self.direction = direction
            status, position_id = self.account.buy(
                self.amount,
                self.config.select_asset,
                self.direction,
                self.config.expiration
            )

            if not status:
                self.logger.error(f"Order creation failed. Status: {status}")
                return

            self.logger.info(
                f"Order opened: {self.direction} {self.config.select_asset} "
                f"ID: {position_id}, TF: {self.config.expiration}, Volume: {self.amount}"
            )

            self._handle_order_result(self.account.check_win_v3(position_id))

        except Exception as e:
            self.logger.error(f"Order processing error: {str(e)}")

    def _handle_order_result(self, win) -> None:
        """Process the result of a trade"""
        
        if win < 0:
            self.logger.info(f"Loss: {win}$")
            self.amount += round((abs(win) + (abs(win) * .15)))
            self.mm += 1
            self.gameLose += 1
            self.TempLose += win
            if self.TempLose < self.MaxLose:
                self.MaxLose = self.TempLose
            
            if self.mm > self.config.max_martingel:
                self.logger.info("Max Martingale reached, resetting")
                self.amount = self.config.start_bet
                self.mm = 0

            # self.create_order2(self.mm)
        else:
            self.logger.info(f"Win: {win}$")
            self.amount = self.config.start_bet
            self.mm = 0
            self.gameWin += 1
            self.TempLose = 0

    def run(self) -> None:
        """Main trading loop with error handling"""
        try:
            while True:
                self.create_order()
                time.sleep(1)
        except KeyboardInterrupt:
            self.logger.info("Trading stopped by user")
        except Exception as e:
            self.logger.error(f"Trading loop error: {str(e)}")
            raise

if __name__ == "__main__":
    server_on()
    defTrade = "GBPUSD-OTC"
    print("Trade:",defTrade)
    trader = IQOptionTrader(defTrade)
    trader.run()
    input()

