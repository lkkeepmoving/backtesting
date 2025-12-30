"""
TradeStation API Client for Backtesting
Handles fetching historical daily bar data
"""

import requests
import pandas as pd
import logging
import os
import json
import webbrowser
import urllib.parse
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

logger = logging.getLogger(__name__)


class CallbackHandler(BaseHTTPRequestHandler):
    """HTTP request handler for TradeStation OAuth2 callback."""

    def do_GET(self):
        """Process OAuth callback and extract authorization code."""
        parsed_url = urlparse(self.path)
        query_params = parse_qs(parsed_url.query)

        if 'code' in query_params:
            state = query_params.get('state', [''])[0]
            if state == 'tradestation_auth':
                self.server.auth_code = query_params['code'][0]
            else:
                logger.warning(f"OAuth state mismatch: {state}")
                self.server.auth_code = None
                self.server.auth_error = "Invalid state parameter"

            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()

            success_html = """
            <html>
            <head><title>TradeStation API Authorization</title></head>
            <body>
                <h1>Authorization Successful!</h1>
                <p>You can now close this browser window and return to the terminal.</p>
            </body>
            </html>
            """
            self.wfile.write(success_html.encode())
        else:
            error = query_params.get('error', ['Unknown error'])[0]
            self.server.auth_code = None
            self.server.auth_error = error

            self.send_response(400)
            self.send_header('Content-type', 'text/html')
            self.end_headers()

            error_html = f"""
            <html>
            <head><title>TradeStation API Authorization Error</title></head>
            <body>
                <h1>Authorization Failed</h1>
                <p>Error: {error}</p>
            </body>
            </html>
            """
            self.wfile.write(error_html.encode())

    def log_message(self, format, *args):
        """Suppress HTTP server log messages"""
        pass


class TradeStationClient:
    """Client for interacting with TradeStation API"""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize TradeStation client with automatic token management

        Args:
            config: TradeStation configuration with keys:
                   - api_key
                   - api_secret
                   - base_url (e.g., "https://api.tradestation.com/v3")
        """
        self.api_key = config['api_key']
        self.api_secret = config['api_secret']
        self.base_url = config['base_url']
        self.callback_url = "http://localhost:3000"
        self.auth_base_url = "https://signin.tradestation.com"

        self.access_token = None
        self.refresh_token = None
        self.token_expires_at = None
        self.authenticated = False

        # Try to authenticate with existing tokens first
        if not self._try_authenticate():
            logger.info("No valid tokens found. Interactive authentication required.")
            logger.info("Opening browser for TradeStation authorization...")

            if not self._authenticate_interactive():
                raise ValueError(
                    "Failed to authenticate with TradeStation. "
                    "Please check your API credentials and try again."
                )

        logger.info("TradeStation client initialized and authenticated successfully")

    def _try_authenticate(self) -> bool:
        """Try to authenticate silently using existing tokens."""
        try:
            token_file = os.path.join(os.getcwd(), '.tradestation_tokens')
            if os.path.exists(token_file):
                with open(token_file, 'r') as f:
                    token_data = json.load(f)

                self.access_token = token_data.get('access_token')
                self.refresh_token = token_data.get('refresh_token')
                self.token_expires_at = token_data.get('expires_at')

                if self.access_token and self._is_token_valid():
                    self.authenticated = True
                    logger.info("Successfully authenticated with existing tokens")
                    return True
                elif self.refresh_token:
                    if self._refresh_access_token():
                        self.authenticated = True
                        logger.info("Successfully refreshed API token")
                        return True

                logger.warning("Existing tokens are invalid or expired")
            else:
                logger.info("No existing tokens found - authentication required")

            return False

        except Exception as e:
            logger.error(f"Authentication initialization error: {str(e)}")
            return False

    def _authenticate_interactive(self) -> bool:
        """Authenticate with TradeStation API using OAuth 2.0 interactive flow"""
        if not self.api_key or not self.api_secret:
            logger.error("TradeStation API credentials not configured")
            return False

        try:
            logger.info("Starting TradeStation OAuth authentication flow...")

            server = HTTPServer(('localhost', 3000), CallbackHandler)
            server.auth_code = None
            server.auth_error = None
            server.timeout = 1

            auth_url = self._get_authorization_url()
            logger.info(f"Opening browser to: {auth_url}")

            try:
                webbrowser.open(auth_url)
                logger.info("Browser opened successfully")
            except Exception as e:
                logger.warning(f"Could not open browser: {e}")
                logger.info(f"Please manually visit: {auth_url}")

            logger.info("Waiting for authorization...")

            max_attempts = 120  # 2 minutes timeout
            attempts = 0

            while attempts < max_attempts and server.auth_code is None and server.auth_error is None:
                server.handle_request()
                attempts += 1

            if server.auth_code:
                logger.info("Authorization code received!")

                if self._exchange_code_for_tokens(server.auth_code):
                    self.authenticated = True
                    logger.info("Authentication completed successfully!")
                    self._save_tokens()
                    return True
                else:
                    logger.error("Failed to exchange authorization code for tokens")
                    return False

            elif server.auth_error:
                logger.error(f"Authorization failed: {server.auth_error}")
                return False
            else:
                logger.error("Timeout waiting for authorization")
                return False

        except Exception as e:
            logger.error(f"Authentication error: {str(e)}")
            return False
        finally:
            try:
                server.server_close()
            except:
                pass

    def _get_authorization_url(self) -> str:
        """Generate OAuth authorization URL"""
        params = {
            'response_type': 'code',
            'client_id': self.api_key,
            'audience': 'https://api.tradestation.com',
            'redirect_uri': self.callback_url,
            'scope': 'openid offline_access MarketData ReadAccount Trade',
            'state': 'tradestation_auth'
        }
        query_string = urllib.parse.urlencode(params)
        return f"{self.auth_base_url}/authorize?{query_string}"

    def _exchange_code_for_tokens(self, auth_code: str) -> bool:
        """Exchange authorization code for access and refresh tokens"""
        try:
            headers = {'Content-Type': 'application/x-www-form-urlencoded'}

            data = {
                'grant_type': 'authorization_code',
                'client_id': self.api_key,
                'client_secret': self.api_secret,
                'code': auth_code,
                'redirect_uri': self.callback_url
            }

            response = requests.post(
                f"{self.auth_base_url}/oauth/token",
                headers=headers,
                data=data,
                timeout=30
            )

            if response.status_code == 200:
                token_data = response.json()
                self.access_token = token_data.get('access_token')
                self.refresh_token = token_data.get('refresh_token')

                expires_in = token_data.get('expires_in', 3600)
                self.token_expires_at = (datetime.now() + timedelta(seconds=expires_in)).isoformat()

                return True
            else:
                logger.error(f"Token exchange failed: {response.status_code} - {response.text}")
                return False

        except Exception as e:
            logger.error(f"Error exchanging code for tokens: {e}")
            return False

    def _is_token_valid(self) -> bool:
        """Check if current access token is still valid"""
        if not self.access_token or not self.token_expires_at:
            return False

        try:
            expires_at = datetime.fromisoformat(self.token_expires_at)
            return datetime.now() < (expires_at - timedelta(minutes=5))
        except (ValueError, TypeError):
            return False

    def _refresh_access_token(self) -> bool:
        """Refresh the access token using refresh token"""
        if not self.refresh_token:
            return False

        try:
            headers = {'Content-Type': 'application/x-www-form-urlencoded'}

            data = {
                'grant_type': 'refresh_token',
                'client_id': self.api_key,
                'client_secret': self.api_secret,
                'refresh_token': self.refresh_token
            }

            response = requests.post(
                f"{self.auth_base_url}/oauth/token",
                headers=headers,
                data=data,
                timeout=30
            )

            if response.status_code == 200:
                token_data = response.json()
                self.access_token = token_data.get('access_token')

                expires_in = token_data.get('expires_in', 3600)
                self.token_expires_at = (datetime.now() + timedelta(seconds=expires_in)).isoformat()

                self._save_tokens()
                return True
            else:
                logger.error(f"Token refresh failed: {response.status_code}")
                return False

        except Exception as e:
            logger.error(f"Error refreshing token: {e}")
            return False

    def _save_tokens(self):
        """Save tokens to local file"""
        try:
            token_data = {
                'access_token': self.access_token,
                'refresh_token': self.refresh_token,
                'expires_at': self.token_expires_at,
                'timestamp': datetime.now().isoformat()
            }

            token_file = os.path.join(os.getcwd(), '.tradestation_tokens')
            with open(token_file, 'w') as f:
                json.dump(token_data, f)

            logger.info("Tokens saved successfully")

        except Exception as e:
            logger.warning(f"Could not save tokens: {e}")

    def _ensure_authenticated(self) -> bool:
        """Ensure client is authenticated with valid token"""
        if not self.authenticated:
            logger.error("Client is not authenticated")
            return False

        if not self._is_token_valid() and self.refresh_token:
            logger.info("Access token expired, refreshing...")
            if not self._refresh_access_token():
                logger.error("Failed to refresh access token")
                return False
            logger.debug("Token refreshed successfully")

        return True

    def _get_headers(self) -> Dict[str, str]:
        """Get headers for API requests"""
        return {
            'Authorization': f'Bearer {self.access_token}',
            'Content-Type': 'application/json'
        }

    def get_bars_by_bar_count(self, symbol: str, bar_count: int) -> Optional[pd.DataFrame]:
        """
        Retrieve historical daily OHLC data by number of bars

        Args:
            symbol: Stock symbol (e.g., "AAPL", "MSFT")
            bar_count: Number of bars (trading days) to retrieve

        Returns:
            DataFrame with columns: Date, Open, High, Low, Close, Volume
            Sorted oldest to newest
            Returns None on error
        """
        if not self._ensure_authenticated():
            return None

        try:
            logger.debug(f"Fetching {bar_count} bars for {symbol}")

            url = f"{self.base_url}/marketdata/barcharts/{symbol}"
            params = {
                'interval': '1',
                'unit': 'Daily',
                'barsback': str(bar_count)
            }

            response = requests.get(url, headers=self._get_headers(), params=params, timeout=30)

            if response.status_code != 200:
                logger.error(f"API error fetching bars for {symbol}: {response.status_code}")
                logger.error(f"Response: {response.text}")
                return None

            data = response.json()
            bars = data.get('Bars', [])

            if not bars:
                logger.warning(f"No bar data returned for {symbol}")
                return None

            df = pd.DataFrame(bars)

            # Rename columns to match our desired format
            df.rename(columns={
                'TimeStamp': 'Date',
                'Open': 'Open',
                'High': 'High',
                'Low': 'Low',
                'Close': 'Close',
                'Volume': 'Volume'
            }, inplace=True)

            # Convert date to YYYY-MM-DD format
            df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')

            # Convert OHLC to numeric types
            df['Open'] = pd.to_numeric(df['Open'], errors='coerce')
            df['High'] = pd.to_numeric(df['High'], errors='coerce')
            df['Low'] = pd.to_numeric(df['Low'], errors='coerce')
            df['Close'] = pd.to_numeric(df['Close'], errors='coerce')

            # Handle volume (may not exist for indices)
            if 'Volume' in df.columns:
                df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce')
            else:
                df['Volume'] = 0

            # Select and order columns
            df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]

            # Sort by date (newest to oldest)
            df = df.sort_values('Date', ascending=False).reset_index(drop=True)

            logger.debug(f"Successfully fetched {len(df)} bars for {symbol}")
            logger.debug(f"Date range: {df['Date'].iloc[0]} to {df['Date'].iloc[-1]}")

            return df

        except Exception as e:
            logger.error(f"Error fetching bars for {symbol}: {e}", exc_info=True)
            return None
