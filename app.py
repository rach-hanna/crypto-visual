import math, time, json, requests, pandas as pd, numpy as np, plotly.graph_objects as go, plotly.express as px
from datetime import datetime, timezone
from plotly.io import to_html
import plotly.io as pio

# aesthetic
pio.templates.default = "plotly_dark"
pio.templates["plotly_dark"]["layout"]["font"] = {"family": "Aptos, sans-serif", "size": 13}

# config
BASE = "https://api.binance.com"
SYMBOL = "BTCUSDT"
INTERVAL = "1m"
LIMIT = 300

# binance
def binance_get(path, params=None):
    r = requests.get(BASE + path, params=params, timeout=15)
    r.raise_for_status()
    return r.json()

def get_klines(symbol=SYMBOL, interval=INTERVAL, limit=LIMIT):
    raw = binance_get("/api/v3/klines", {"symbol": symbol, "interval": interval, "limit": limit})
    cols = ["open_time","open","high","low","close","volume","close_time","quote_vol","trades","taker_base","taker_quote","ignore"]
    df = pd.DataFrame(raw, columns=cols)
    for c in ["open","high","low","close","volume","quote_vol","taker_base","taker_quote"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True).dt.tz_convert("UTC")
    df.set_index("time", inplace=True)
    return df[["open","high","low","close","volume"]]

def get_orderbook(symbol=SYMBOL, depth=50):
    raw = binance_get("/api/v3/depth", {"symbol": symbol, "limit": depth})
    bids = pd.DataFrame(raw["bids"], columns=["price","qty"]).astype(float)
    asks = pd.DataFrame(raw["asks"], columns=["price","qty"]).astype(float)
    bids["side"] = "bid"; asks["side"] = "ask"
    ob = pd.concat([bids, asks], ignore_index=True)
    ob["notional"] = ob["price"] * ob["qty"]
    return ob.sort_values(["side","price"], ascending=[True, True])

# metrics
def liquidity_metrics(ob):
    best_bid = ob[ob["side"]=="bid"]["price"].max()
    best_ask = ob[ob["side"]=="ask"]["price"].min()
    spread = best_ask - best_bid
    mid = (best_ask + best_bid) / 2
    rel_spread_bps = (spread / mid) * 1e4
    depth_10bp = ob[(ob["side"]=="bid") & (ob["price"] >= mid*(1-0.001))]["notional"].sum() + \
                 ob[(ob["side"]=="ask") & (ob["price"] <= mid*(1+0.001))]["notional"].sum()
    return {"mid_price": float(mid), "spread_abs": float(spread),
            "spread_bps": float(rel_spread_bps), "depth_±10bp_notional": float(depth_10bp)}

def realized_vol(df, window=30):
    r = np.log(df["close"]).diff()
    out = df.copy()
    out["rv"] = (r.rolling(window).std() * np.sqrt(60)).fillna(0)
    return out

# build
def build_dashboard(symbol=SYMBOL):
    kl = get_klines(symbol)
    ob = get_orderbook(symbol)
    mets = liquidity_metrics(ob)
    klv = realized_vol(kl)

    # animation
    frames = [
        go.Frame(
            data=[go.Candlestick(x=kl.index[:k], open=kl["open"][:k], high=kl["high"][:k],
                                 low=kl["low"][:k], close=kl["close"][:k])],
            name=str(k)
        ) for k in range(20, len(kl), 5)
    ]
    fig_price = go.Figure(
        data=[go.Candlestick(x=kl.index, open=kl["open"], high=kl["high"], low=kl["low"], close=kl["close"])],
        frames=frames
    )
    fig_price.update_layout(
        title=f"{symbol}: animated price (binance, {INTERVAL})",
        xaxis_title="time (UTC)",
        yaxis_title="price (USDT)",
        height=450,
        margin=dict(l=40, r=40, t=60, b=40),
        updatemenus=[{
            "type": "buttons", "x": 0.05, "y": 1.15, "showactive": False,
            "buttons": [
                {"label": "▶ play", "method": "animate",
                 "args": [None, {"frame": {"duration": 80, "redraw": True}, "fromcurrent": True}]},
                {"label": "⏸ pause", "method": "animate",
                 "args": [[None], {"frame": {"duration": 0}, "mode": "immediate"}]}
            ]
        }]
    )

    # volume
    fig_vol = px.bar(kl.reset_index(), x="time", y="volume", title="volume")
    fig_vol.update_layout(height=260, margin=dict(l=40, r=40, t=50, b=40))

    # realised volatility
    fig_rv = px.line(klv.reset_index(), x="time", y="rv", title="realised volatility (≈ hourly)")
    fig_rv.update_layout(height=260, yaxis_title="σ", margin=dict(l=40, r=40, t=50, b=40))

    # order book
    ob_plot = ob.copy()
    ob_plot["cum_qty"] = ob_plot.groupby("side")["qty"].cumsum()
    fig_ob = px.scatter(ob_plot, x="price", y="cum_qty", color="side",
                        title=f"order book depth: top {ob_plot.groupby('side').size().min()} levels")
    fig_ob.update_layout(height=360, xaxis_title="price", yaxis_title="cumulative qty",
                         margin=dict(l=40, r=40, t=50, b=40))

    # metrics
    metrics_md = f"""
    <h3>liquidity snapshot</h3>
    <ul>
      <li>mid price: <b>{mets['mid_price']:.2f}</b></li>
      <li>spread: <b>{mets['spread_abs']:.2f}</b> ({mets['spread_bps']:.2f} bps)</li>
      <li>depth ±10bp (notional): <b>{mets['depth_±10bp_notional']:.2f}</b></li>
    </ul>
    """

    # HTML aesthetic
    html_parts = [
        f"<h1>crypto visualiser: {symbol}</h1>",
        f"<p style='opacity:.8'>generated at {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S %Z')}</p>",
        metrics_md,
        f"<div class='block'>{to_html(fig_price, include_plotlyjs='cdn', full_html=False)}</div>",
        f"<div class='block'>{to_html(fig_vol, include_plotlyjs=False, full_html=False)}</div>",
        f"<div class='block'>{to_html(fig_rv, include_plotlyjs=False, full_html=False)}</div>",
        f"<div class='block'>{to_html(fig_ob, include_plotlyjs=False, full_html=False)}</div>",
        "<p style='font-size:12px;opacity:.7'>data from binance public rest api</p>"
    ]

    html = (
        "<title>crypto liquidity dashboard</title>"
"<!DOCTYPE html><html lang='en'><head>"
        "<meta charset='utf-8'><meta http-equiv='Content-Language' content='en'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        "<link href='https://fonts.cdnfonts.com/css/aptos' rel='stylesheet'>"
        "<style>"
        "body{background:#0f1116;color:#fff;font-family:'Aptos',sans-serif;margin:40px}"
        ".container{max-width:1100px;margin:auto}"
        ".block{margin:24px 0}"
        "</style></head><body>"
        "<div class='container'>" + "".join(html_parts) + "</div>"
        "</body></html>"
    )

    open("dashboard.html", "w", encoding="utf-8").write(html)
    print("dashboard generated")

# run
if __name__ == "__main__":
    build_dashboard()
