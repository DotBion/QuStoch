import random

import numpy as np
import yfinance as yf

from pprint import pprint

# Used when the Treasury bill series cannot be fetched.
DEFAULT_RISK_FREE_RATE = 0.02

s_and_p_500 = ['MMM','AOS','ABT','ABBV','ACN','ATVI','AYI','ADBE','AAP','AMD','AES','AET','AMG','AFL','A','APD','AKAM','ALK','ALB','ARE','ALXN','ALGN','ALLE','AGN','ADS','LNT','ALL','GOOGL','GOOG','MO','AMZN','AEE','AAL','AEP','AXP','AIG','AMT','AWK','AMP','ABC','AME','AMGN','APH','APC','ADI','ANDV','ANSS','ANTM','AON','APA','AIV','AAPL','AMAT','APTV','ADM','ARNC','AJG','AIZ','T','ADSK','ADP','AZO','AVB','AVY','BHGE','BLL','BAC','BAX','BBT','BDX','BRK.B','BBY','BIIB','BLK','HRB','BA','BWA','BXP','BSX','BHF','BMY','AVGO','BF.B','CHRW','CA','COG','CDNS','CPB','COF','CAH','KMX','CCL','CAT','CBOE','CBG','CBS','CELG','CNC','CNP','CTL','CERN','CF','SCHW','CHTR','CHK','CVX','CMG','CB','CHD','CI','XEC','CINF','CTAS','CSCO','C','CFG','CTXS','CME','CMS','KO','CTSH','CL','CMCSA','CMA','CAG','CXO','COP','ED','STZ','GLW','COST','COTY','CCI','CSRA','CSX','CMI','CVS','DHI','DHR','DRI','DVA','DE','DAL','XRAY','DVN','DLR','DFS','DISCA','DISCK','DISH','DG','DLTR','D','DOV','DWDP','DPS','DTE','DUK','DRE','DXC','ETFC','EMN','ETN','EBAY','ECL','EIX','EW','EA','EMR','ETR','EVHC','EOG','EQT','EFX','EQIX','EQR','ESS','EL','RE','ES','EXC','EXPE','EXPD','ESRX','EXR','XOM','FFIV','FB','FAST','FRT','FDX','FIS','FITB','FE','FISV','FLIR','FLS','FLR','FMC','FL','F','FTV','FBHS','BEN','FCX','GPS','GRMN','IT','GD','GE','GGP','GIS','GM','GPC','GILD','GPN','GS','GT','GWW','HAL','HBI','HOG','HRS','HIG','HAS','HCA','HCP','HP','HSIC','HES','HPE','HLT','HOLX','HD','HON','HRL','HST','HPQ','HUM','HBAN','HII','IDXX','INFO','ITW','ILMN','INCY','IR','INTC','ICE','IBM','IP','IPG','IFF','INTU','ISRG','IVZ','IQV','IRM','JBHT','JEC','SJM','JNJ','JCI','JPM','JNPR','KSU','K','KEY','KMB','KIM','KMI','KLAC','KSS','KHC','KR','LB','LLL','LH','LRCX','LEG','LEN','LUK','LLY','LNC','LKQ','LMT','L','LOW','LYB','MTB','MAC','M','MRO','MPC','MAR','MMC','MLM','MAS','MA','MAT','MKC','MCD','MCK','MDT','MRK','MET','MTD','MGM','KORS','MCHP','MU','MSFT','MAA','MHK','TAP','MDLZ','MON','MNST','MCO','MS','MSI','MYL','NDAQ','NOV','NAVI','NTAP','NFLX','NWL','NFX','NEM','NWSA','NWS','NEE','NLSN','NKE','NI','NBL','JWN','NSC','NTRS','NOC','NCLH','NRG','NUE','NVDA','ORLY','OXY','OMC','OKE','ORCL','PCAR','PKG','PH','PDCO','PAYX','PYPL','PNR','PBCT','PEP','PKI','PRGO','PFE','PCG','PM','PSX','PNW','PXD','PNC','RL','PPG','PPL','PX','PCLN','PFG','PG','PGR','PLD','PRU','PEG','PSA','PHM','PVH','QRVO','QCOM','PWR','DGX','RRC','RJF','RTN','O','RHT','REG','REGN','RF','RSG','RMD','RHI','ROK','COL','ROP','ROST','RCL','SPGI','CRM','SBAC','SCG','SLB','SNI','STX','SEE','SRE','SHW','SIG','SPG','SWKS','SLG','SNA','SO','LUV','SWK','SBUX','STT','SRCL','SYK','STI','SYMC','SYF','SNPS','SYY','TROW','TPR','TGT','TEL','FTI','TXN','TXT','BK','CLX','COO','HSY','MOS','TRV','DIS','TMO','TIF','TWX','TJX','TMK','TSS','TSCO','TDG','TRIP','FOXA','FOX','TSN','USB','UDR','ULTA','UAA','UA','UNP','UAL','UNH','UPS','URI','UTX','UHS','UNM','VFC','VLO','VAR','VTR','VRSN','VRSK','VZ','VRTX','VIAB','V','VNO','VMC','WMT','WBA','WM','WAT','WEC','WFC','HCN','WDC','WU','WRK','WY','WHR','WMB','WLTW','WYN','WYNN','XEL','XRX','XLNX','XL','XYL','YUM','ZBH','ZION','ZTS']

def _last_scalar(obj):
    """Reduce whatever pandas hands back to the last finite float.

    yfinance returns a Series for some calls and a MultiIndex-column DataFrame for others
    depending on version and on whether one or many tickers were requested, so unpack
    generically rather than chaining positional ``.iloc`` calls.
    """
    values = np.asarray(obj, dtype=float).ravel()
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("no finite values in series")
    return float(values[-1])


def get_stock_daily_data(symbol: str):
    """
    Fetches daily historical data for a given symbol using yfinance.
    Retrieves approximately one year of data.
    
    Returns:
      - A DataFrame with date-indexed daily data containing columns like Close and Volume.
    """
    ticker = yf.Ticker(symbol)
    # Download ~1 year of historical data with daily resolution.
    df = ticker.history(period="1y", interval="1d")
    
    if df.empty:
        print(f"Error: No data found for {symbol} in the specified date range.")
        return None
    
    # Format the index to a string date format (YYYY-MM-DD)
    df.index = df.index.strftime('%Y-%m-%d')
    return df

def compute_historical_volatility(df, days=30, annualize=True):
    """
    Computes historical volatility based on the standard deviation of logarithmic returns.
    
    Parameters:
      - df: DataFrame containing at least the 'Close' column.
      - days: Number of most recent trading days to include.
      - annualize: If True, annualizes the volatility (assuming 252 trading days per year).
      
    Returns:
      - Annualized historical volatility (as a decimal) or None if there is an error.
    """
    # Use the last 'days' of closing prices.
    close_prices = np.asarray(df['Close'][-days:], dtype=float).ravel()

    if len(close_prices) < 2:
        print("Error: Not enough data to compute volatility.")
        return None

    # Calculate log returns: ln(P_t / P_(t-1))
    log_returns = np.diff(np.log(close_prices))
    # ddof=1: these returns are a sample, not the whole population.
    vol = float(np.std(log_returns, ddof=1))

    if annualize:
        vol = vol * np.sqrt(252)

    return vol

def risk_free_return():
    """
    Treasury bill yield is more stable estimator of risk_free_return.

    Returns the 3-month U.S. Treasury Bill yield (^IRX) as a decimal, falling back to
    DEFAULT_RISK_FREE_RATE when the series is unavailable.
    """
    try:
        tbill = yf.download("^IRX", period="1mo", progress=False)
        if tbill is None or tbill.empty:
            return DEFAULT_RISK_FREE_RATE
        return _last_scalar(tbill["Close"]) / 100  # the series is quoted in percent
    except Exception as exc:
        print(f"Could not fetch risk-free rate ({exc}); using default.")
        return DEFAULT_RISK_FREE_RATE

def get_stock_info(symbol):
    """
    Retrieves stock data for the specified symbol:
      - Downloads approximately one year of daily data using yfinance.
      - Extracts the latest day's closing price and volume.
      - Computes the annualized historical volatility based on the last 30 trading days.
      - Fetches the current risk-free rate from the 3-month Treasury bill.

    Returns:
        A dictionary with 'latest_date', 'latest_price', 'latest_volume',
        'historical_volatility' and 'risk_free_return'.

    Raises:
        ValueError: if the ticker has no data or too little history.
    """

    df = get_stock_daily_data(symbol)
    if df is None:
        raise ValueError(f"no market data available for {symbol!r}")

    # Get the latest day's data for display.
    latest_date = df.index[-1]
    latest_price = _last_scalar(df['Close'])
    latest_volume = _last_scalar(df['Volume'])

    # Compute historical volatility based on the last 30 trading days.
    hist_vol = compute_historical_volatility(df, days=30, annualize=True)
    if hist_vol is None:
        raise ValueError(f"not enough history to compute volatility for {symbol!r}")

    return {
        "latest_date": latest_date,
        "latest_price": latest_price,
        "latest_volume": latest_volume,
        "historical_volatility": hist_vol,
        "risk_free_return": float(risk_free_return()),
    }

# # Example usage:
if __name__ == "__main__":
    symbol = random.choice(s_and_p_500)

    try:
        stock_info = get_stock_info(symbol)
    except ValueError as exc:
        print(f"Could not load {symbol}: {exc}")
    else:
        print("\nReturned Stock Information for", symbol)
        pprint(stock_info)
