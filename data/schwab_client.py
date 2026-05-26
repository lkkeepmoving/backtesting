"""
Charles Schwab API Client for Backtesting
Handles fetching historical daily OHLCV data via the Market Data API.

Authentication: OAuth 2.0 Authorization Code flow.
Since Schwab requires an HTTPS redirect URI, the interactive flow prints the
authorization URL, opens the browser, and asks the user to paste back the
full redirect URL containing the authorization code.
"""

import requests
import pandas as pd
import logging
import os
import json
import webbrowser
import urllib.parse
from typing import Optional, Dict, Any
from datetime import datetime, timedelta, timezone

logger = logging.getLogger(__name__)

SCHWAB_AUTH_BASE_URL = "https://api.schwabapi.com/v1/oauth"
SCHWAB_MARKET_DATA_BASE_URL = "https://api.schwabapi.com/marketdata/v1"
TOKEN_FILE = os.path.join(os.path.dirname(__file__), ".schwab_tokens")


class SchwabClient:
    """Client for interacting with the Charles Schwab Market Data API."""

    def __init__(self, client_id: str, client_secret: str, redirect_uri: str):
        """
        Initialize the Schwab client with automatic token management.

        Args:
            client_id:     App key from developer.schwab.com
            client_secret: App secret from developer.schwab.com
            redirect_uri:  Redirect URI registered in your Schwab app
                           (e.g. "https://127.0.0.1")
        """
        self.client_id = client_id
        self.client_secret = client_secret
        self.redirect_uri = redirect_uri

        self.access_token: Optional[str] = None
        self.refresh_token: Optional[str] = None
        self.token_expires_at: Optional[str] = None
        self.authenticated = False

        if not self._try_authenticate():
            logger.info("No valid tokens found. Starting interactive OAuth flow...")
            if not self._authenticate_interactive():
                raise ValueError(
                    "Failed to authenticate with Schwab. "
                    "Check your client_id, client_secret, and redirect_uri."
                )

        logger.info("Schwab client initialized and authenticated successfully.")

    # ------------------------------------------------------------------
    # Authentication
    # ------------------------------------------------------------------

    def _try_authenticate(self) -> bool:
        """Attempt silent authentication using cached tokens."""
        if not os.path.exists(TOKEN_FILE):
            logger.info("No cached tokens found.")
            return False

        try:
            with open(TOKEN_FILE, "r") as f:
                token_data = json.load(f)

            self.access_token = token_data.get("access_token")
            self.refresh_token = token_data.get("refresh_token")
            self.token_expires_at = token_data.get("expires_at")

            if self.access_token and self._is_token_valid():
                self.authenticated = True
                logger.info("Authenticated with cached access token.")
                return True

            if self.refresh_token:
                if self._refresh_access_token():
                    self.authenticated = True
                    logger.info("Authenticated via token refresh.")
                    return True

            logger.warning("Cached tokens are expired or invalid.")
            return False

        except Exception as e:
            logger.error(f"Error loading cached tokens: {e}")
            return False

    def _authenticate_interactive(self) -> bool:
        """
        OAuth 2.0 Authorization Code flow.

        Schwab requires an HTTPS redirect URI, so we cannot spin up a plain
        HTTP localhost server to capture the callback.  Instead we:
          1. Print (and open) the authorization URL.
          2. Ask the user to paste back the full redirect URL after authorizing.
          3. Extract the code from that URL and exchange it for tokens.
        """
        auth_url = self._get_authorization_url()

        print("\n" + "=" * 60)
        print("Schwab OAuth Authorization Required")
        print("=" * 60)
        print("Opening the following URL in your browser:\n")
        print(f"  {auth_url}\n")
        print("If the browser does not open, copy the URL above and")
        print("paste it into your browser manually.\n")
        print("After you authorize the app, your browser will be")
        print("redirected to your registered redirect URI.  The URL")
        print("will contain a 'code=...' query parameter.\n")
        print("Paste the FULL redirect URL here and press Enter:")
        print("=" * 60)

        try:
            webbrowser.open(auth_url)
        except Exception:
            pass  # Already told the user to open it manually

        try:
            redirect_url = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            logger.error("Interactive input cancelled.")
            return False

        auth_code = self._extract_code_from_url(redirect_url)
        if not auth_code:
            logger.error("Could not extract authorization code from the provided URL.")
            return False

        if self._exchange_code_for_tokens(auth_code):
            self.authenticated = True
            self._save_tokens()
            logger.info("Interactive authentication completed successfully.")
            return True

        return False

    def _get_authorization_url(self) -> str:
        """Build the Schwab OAuth authorization URL."""
        params = {
            "response_type": "code",
            "client_id": self.client_id,
            "redirect_uri": self.redirect_uri,
            "scope": "readonly",
            "state": "schwab_auth",
        }
        return f"{SCHWAB_AUTH_BASE_URL}/authorize?{urllib.parse.urlencode(params)}"

    @staticmethod
    def _extract_code_from_url(url: str) -> Optional[str]:
        """Extract the authorization code from a redirect URL."""
        try:
            parsed = urllib.parse.urlparse(url)
            params = urllib.parse.parse_qs(parsed.query)
            codes = params.get("code", [])
            return codes[0] if codes else None
        except Exception:
            return None

    def _exchange_code_for_tokens(self, auth_code: str) -> bool:
        """Exchange an authorization code for access + refresh tokens."""
        try:
            response = requests.post(
                f"{SCHWAB_AUTH_BASE_URL}/token",
                auth=(self.client_id, self.client_secret),
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                data={
                    "grant_type": "authorization_code",
                    "code": auth_code,
                    "redirect_uri": self.redirect_uri,
                },
                timeout=30,
            )

            if response.status_code == 200:
                self._store_token_response(response.json())
                return True

            logger.error(f"Token exchange failed: {response.status_code} – {response.text}")
            return False

        except Exception as e:
            logger.error(f"Error exchanging code for tokens: {e}")
            return False

    def _refresh_access_token(self) -> bool:
        """Use the refresh token to obtain a new access token."""
        if not self.refresh_token:
            return False

        try:
            response = requests.post(
                f"{SCHWAB_AUTH_BASE_URL}/token",
                auth=(self.client_id, self.client_secret),
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                data={
                    "grant_type": "refresh_token",
                    "refresh_token": self.refresh_token,
                },
                timeout=30,
            )

            if response.status_code == 200:
                self._store_token_response(response.json())
                self._save_tokens()
                return True

            logger.error(f"Token refresh failed: {response.status_code} – {response.text}")
            return False

        except Exception as e:
            logger.error(f"Error refreshing token: {e}")
            return False

    def _store_token_response(self, token_data: Dict[str, Any]):
        """Parse and store tokens from a token endpoint response."""
        self.access_token = token_data.get("access_token")
        self.refresh_token = token_data.get("refresh_token", self.refresh_token)
        expires_in = token_data.get("expires_in", 1800)
        self.token_expires_at = (datetime.now() + timedelta(seconds=expires_in)).isoformat()

    def _is_token_valid(self) -> bool:
        """Return True if the access token exists and won't expire within 5 minutes."""
        if not self.access_token or not self.token_expires_at:
            return False
        try:
            expires_at = datetime.fromisoformat(self.token_expires_at)
            return datetime.now() < (expires_at - timedelta(minutes=5))
        except (ValueError, TypeError):
            return False

    def _save_tokens(self):
        """Persist tokens to disk."""
        try:
            with open(TOKEN_FILE, "w") as f:
                json.dump(
                    {
                        "access_token": self.access_token,
                        "refresh_token": self.refresh_token,
                        "expires_at": self.token_expires_at,
                        "saved_at": datetime.now().isoformat(),
                    },
                    f,
                )
            logger.info("Tokens saved to disk.")
        except Exception as e:
            logger.warning(f"Could not save tokens: {e}")

    def _ensure_authenticated(self) -> bool:
        """Ensure the client holds a valid access token, refreshing if needed."""
        if not self.authenticated:
            logger.error("Client is not authenticated.")
            return False

        if not self._is_token_valid():
            logger.info("Access token expired – refreshing...")
            if not self._refresh_access_token():
                logger.error("Failed to refresh access token.")
                return False

        return True

    def _get_headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.access_token}",
            "Accept": "application/json",
        }

    # ------------------------------------------------------------------
    # Market Data
    # ------------------------------------------------------------------

    def get_price_history(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
    ) -> Optional[pd.DataFrame]:
        """
        Fetch daily OHLCV data for a symbol over a date range.

        Args:
            symbol:     Ticker symbol (e.g. "TSM", "AAPL")
            start_date: Inclusive start date in "YYYY-MM-DD" format
            end_date:   Inclusive end date in "YYYY-MM-DD" format

        Returns:
            DataFrame with columns: Date, Open, High, Low, Close, Volume
            Sorted newest-to-oldest (consistent with the rest of this repo).
            Returns None on error.
        """
        if not self._ensure_authenticated():
            return None

        try:
            start_ms = self._date_to_epoch_ms(start_date)
            end_ms = self._date_to_epoch_ms(end_date, end_of_day=True)

            params = {
                "symbol": symbol,
                "periodType": "month",
                "frequencyType": "daily",
                "frequency": 1,
                "startDate": start_ms,
                "endDate": end_ms,
                "needExtendedHoursData": "false",
            }

            logger.debug(f"Fetching price history for {symbol} from {start_date} to {end_date}")

            response = requests.get(
                f"{SCHWAB_MARKET_DATA_BASE_URL}/pricehistory",
                headers=self._get_headers(),
                params=params,
                timeout=30,
            )

            if response.status_code != 200:
                logger.error(f"API error for {symbol}: {response.status_code} – {response.text}")
                return None

            data = response.json()

            if data.get("empty", True):
                logger.warning(f"No data returned for {symbol} in the requested date range.")
                return None

            candles = data.get("candles", [])
            if not candles:
                logger.warning(f"Empty candles list for {symbol}.")
                return None

            df = pd.DataFrame(candles)

            # datetime field is Unix milliseconds
            df["Date"] = pd.to_datetime(df["datetime"], unit="ms").dt.strftime("%Y-%m-%d")

            df["Open"] = pd.to_numeric(df["open"], errors="coerce")
            df["High"] = pd.to_numeric(df["high"], errors="coerce")
            df["Low"] = pd.to_numeric(df["low"], errors="coerce")
            df["Close"] = pd.to_numeric(df["close"], errors="coerce")
            df["Volume"] = pd.to_numeric(df.get("volume", 0), errors="coerce").fillna(0).astype(int)

            df = df[["Date", "Open", "High", "Low", "Close", "Volume"]]

            # Newest-to-oldest, consistent with the rest of the repo
            df = df.sort_values("Date", ascending=False).reset_index(drop=True)

            logger.debug(f"Fetched {len(df)} bars for {symbol} ({df['Date'].iloc[-1]} – {df['Date'].iloc[0]})")
            return df

        except Exception as e:
            logger.error(f"Error fetching price history for {symbol}: {e}", exc_info=True)
            return None

    def get_bars_by_bar_count(self, symbol: str, bar_count: int) -> Optional[pd.DataFrame]:
        """
        Fetch the most recent N daily bars for a symbol.

        Args:
            symbol:    Ticker symbol
            bar_count: Number of trading days to retrieve

        Returns:
            DataFrame with columns: Date, Open, High, Low, Close, Volume
            Sorted newest-to-oldest.
            Returns None on error.
        """
        if not self._ensure_authenticated():
            return None

        try:
            # Schwab valid periods: month=[1,2,3,6], year=[1,2,3,5,10,15,20]
            if bar_count <= 21:
                period_type, period = "month", 1
            elif bar_count <= 42:
                period_type, period = "month", 2
            elif bar_count <= 63:
                period_type, period = "month", 3
            elif bar_count <= 126:
                period_type, period = "month", 6
            elif bar_count <= 252:
                period_type, period = "year", 1
            elif bar_count <= 504:
                period_type, period = "year", 2
            else:
                period_type, period = "year", 3

            params = {
                "symbol": symbol,
                "periodType": period_type,
                "period": period,
                "frequencyType": "daily",
                "frequency": 1,
                "needExtendedHoursData": "false",
            }

            logger.debug(f"Fetching {bar_count} bars for {symbol}")

            response = requests.get(
                f"{SCHWAB_MARKET_DATA_BASE_URL}/pricehistory",
                headers=self._get_headers(),
                params=params,
                timeout=30,
            )

            if response.status_code != 200:
                logger.error(f"API error for {symbol}: {response.status_code} – {response.text}")
                return None

            data = response.json()

            if data.get("empty", True):
                logger.warning(f"No data returned for {symbol}.")
                return None

            candles = data.get("candles", [])
            if not candles:
                logger.warning(f"Empty candles list for {symbol}.")
                return None

            df = pd.DataFrame(candles)

            df["Date"] = pd.to_datetime(df["datetime"], unit="ms").dt.strftime("%Y-%m-%d")
            df["Open"] = pd.to_numeric(df["open"], errors="coerce")
            df["High"] = pd.to_numeric(df["high"], errors="coerce")
            df["Low"] = pd.to_numeric(df["low"], errors="coerce")
            df["Close"] = pd.to_numeric(df["close"], errors="coerce")
            df["Volume"] = pd.to_numeric(df.get("volume", 0), errors="coerce").fillna(0).astype(int)

            df = df[["Date", "Open", "High", "Low", "Close", "Volume"]]
            df = df.sort_values("Date", ascending=False).reset_index(drop=True)

            logger.debug(f"Fetched {len(df)} bars for {symbol}")
            return df

        except Exception as e:
            logger.error(f"Error fetching bars for {symbol}: {e}", exc_info=True)
            return None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _date_to_epoch_ms(date_str: str, end_of_day: bool = False) -> int:
        """
        Convert a "YYYY-MM-DD" string to Unix milliseconds (UTC).

        Args:
            date_str:   Date string in "YYYY-MM-DD" format
            end_of_day: If True, use 23:59:59 instead of 00:00:00
        """
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        if end_of_day:
            dt = dt.replace(hour=23, minute=59, second=59)
        dt_utc = dt.replace(tzinfo=timezone.utc)
        return int(dt_utc.timestamp() * 1000)
