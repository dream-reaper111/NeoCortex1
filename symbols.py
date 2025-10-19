# -*- coding: utf-8 -*-
# symbols.py
def load_symbol_lists():
    # Keep these small & curated for now; you can load CSVs later.
    return {
        "etf_core": ["SPY","QQQ","IWM","DIA","TLT","HYG","GLD","SLV","USO"],
        "futures_reps": ["ES=F","NQ=F","YM=F","RTY=F","CL=F","GC=F","SI=F","ZN=F"],
        "forex_majors": ["EURUSD=X","USDJPY=X","GBPUSD=X","AUDUSD=X","USDCAD=X","USDCHF=X"],
        "crypto_yf": ["BTC-USD","ETH-USD","SOL-USD","ADA-USD","DOGE-USD"],
        "largecap_sample": ["AAPL","MSFT","NVDA","AMZN","META","GOOGL","TSLA","AVGO","LLY","JPM"]
    }
