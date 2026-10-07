import sys, json, os, datetime as dt, urllib.request
import numpy as np, pandas as pd, yfinance as yf
R = json.load(open("regole.json"))["soglie"]
def wil(s, n): return s.ewm(alpha=1/n, adjust=False, min_periods=n).mean()
def nn(v): return v is None or pd.isna(v)
def f(x, d=4): return None if nn(x) else round(float(x), d)
def vote(v, lo, hi): return None if nn(v) else "Compra" if v >= hi else "Vendi" if v <= lo else "Neutrale"
def sgn(v): return None if nn(v) else "Compra" if v > 0 else "Vendi" if v < 0 else "Neutrale"
def label(v, lo, hi): return None if nn(v) else "Ipervenduto" if v < lo else "Ipercomprato" if v > hi else None
def agg(votes):
    v = [x for x in votes if x]
    if not v: return None, None
    s = (v.count("Compra") - v.count("Vendi")) / len(v) * 100; lo, hi = R["aggregato"]
    return ("Compra Adesso" if s >= hi else "Compra" if s >= lo else "Vendi Adesso" if s <= -hi else "Vendi" if s <= -lo else "Neutrale"), round(s, 1)
def calc(d):
    if len(d) < 30: return None
    h, l, c = d.High, d.Low, d.Close; pc = c.shift(1); up = c.diff()
    rsi = 100 - 100 / (1 + wil(up.clip(lower=0), 14) / wil((-up).clip(lower=0), 14))
    k = (100 * (c - l.rolling(9).min()) / (h.rolling(9).max() - l.rolling(9).min())).rolling(6).mean()
    srsi = 100 * (rsi - rsi.rolling(14).min()) / (rsi.rolling(14).max() - rsi.rolling(14).min())
    mac = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    sig = mac.ewm(span=9, adjust=False).mean(); diff = mac - sig
    s = np.sign(diff.iloc[-40:]).replace(0, np.nan).ffill().dropna(); ch = s[s != s.shift(1)].index[1:]; cross = None
    if len(ch):
        n = len(s) - 1 - s.index.get_loc(ch[-1])
        cross = {"direzione": "rialzista" if s.loc[ch[-1]] > 0 else "ribassista", "barre_fa": int(n), "recente": bool(n <= R["macd_recente"])}
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1); atr = wil(tr, 14)
    pdm = h.diff().where((h.diff() > -l.diff()) & (h.diff() > 0), 0.0); mdm = (-l.diff()).where((-l.diff() > h.diff()) & (-l.diff() > 0), 0.0)
    pdi = 100 * wil(pdm, 14) / atr; mdi = 100 * wil(mdm, 14) / atr; adx = wil(100 * (pdi - mdi).abs() / (pdi + mdi), 14)
    tp = (h + l + c) / 3; cci = (tp - tp.rolling(14).mean()) / (0.015 * tp.rolling(14).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True))
    roc = 100 * (c / c.shift(12) - 1)
    lo_ = pd.concat([l, pc], axis=1).min(axis=1); bp = c - lo_; t2 = pd.concat([h, pc], axis=1).max(axis=1) - lo_
    av = lambda n: bp.rolling(n).sum() / t2.rolling(n).sum(); uo = 100 * (4 * av(7) + 2 * av(14) + av(28)) / 7
    wr = -100 * (h.rolling(14).max() - c) / (h.rolling(14).max() - l.rolling(14).min())
    e13 = c.ewm(span=13, adjust=False).mean(); bull = h - e13; bear = l - e13
    el = "Neutrale"
    if e13.iloc[-1] > e13.iloc[-2] and bear.iloc[-1] < 0 and bear.iloc[-1] > bear.iloc[-2]: el = "Compra"
    elif e13.iloc[-1] < e13.iloc[-2] and bull.iloc[-1] > 0 and bull.iloc[-1] < bull.iloc[-2]: el = "Vendi"
    L = lambda x: x.iloc[-1]; ls = L(c)
    adx_v = None if nn(L(adx)) else ("Neutrale" if L(adx) <= R["adx"] else "Compra" if L(pdi) > L(mdi) else "Vendi")
    ind = [
     {"nome": "RSI(14)", "valore": f(L(rsi), 2), "azione": vote(L(rsi), *R["rsi"]), "nota": label(L(rsi), 30, 70)},
     {"nome": "STOCH(9,6)", "valore": f(L(k), 2), "azione": vote(L(k), *R["stoch"]), "nota": label(L(k), 20, 80)},
     {"nome": "STOCHRSI(14)", "valore": f(L(srsi), 2), "azione": vote(L(srsi), *R["stoch"]), "nota": label(L(srsi), 20, 80)},
     {"nome": "MACD(12,26,9)", "valore": f(L(mac)), "azione": sgn(L(diff)), "nota": None},
     {"nome": "ADX(14)", "valore": f(L(adx), 2), "azione": adx_v, "nota": None},
     {"nome": "CCI(14)", "valore": f(L(cci), 2), "azione": sgn(L(cci)), "nota": label(L(cci), -100, 100)},
     {"nome": "ROC(12)", "valore": f(L(roc), 3), "azione": sgn(L(roc)), "nota": None},
     {"nome": "Ultimate Oscillator", "valore": f(L(uo), 2), "azione": None if nn(L(uo)) else sgn(L(uo) - 50), "nota": label(L(uo), 30, 70)},
     {"nome": "Bull/Bear Power(13)", "valore": f(L(bull) + L(bear)), "azione": el, "nota": None},
     {"nome": "Williams %R(14)", "valore": f(L(wr), 2), "azione": None, "nota": label(L(wr), -80, -20)},
     {"nome": "ATR(14)", "valore": f(L(atr)), "azione": None, "nota": None}]
    ma = []
    for n in (5, 10, 20, 50, 100, 200):
        sm = c.rolling(n).mean().iloc[-1]; em = c.ewm(span=n, adjust=False, min_periods=n).mean().iloc[-1]
        ma.append({"periodo": n, "sma": f(sm, 3), "v_sma": None if nn(sm) else "Compra" if ls > sm else "Vendi",
                   "ema": f(em, 3), "v_ema": None if nn(em) else "Compra" if ls > em else "Vendi"})
    vi = [x["azione"] for x in ind]; vm = [m[q] for m in ma for q in ("v_sma", "v_ema")]
    (ri, si), (rm, sm_), (rt, st) = agg(vi), agg(vm), agg(vi + vm)
    return {"barre": len(d), "ultima": str(d.index[-1].date()), "riassunto": rt, "saldo": st, "ind_riassunto": ri, "ind_saldo": si,
            "ma_riassunto": rm, "ma_saldo": sm_, "indicatori": ind, "medie": ma,
            "macd": {"macd": f(L(mac)), "segnale": f(L(sig)), "istogramma": f(L(diff)), "incrocio": cross}}
def lv(p, h, l, c, r):  # [S3,S2,S1,P,R1,R2,R3] forma classica
    return [l - 2 * (h - p), p - r, 2 * p - h, p, 2 * p - l, p + r, h + 2 * (p - l)]
def pivots(d):
    o, h, l, c = d.Open.iloc[-1], d.High.iloc[-1], d.Low.iloc[-1], d.Close.iloc[-1]; p = (h + l + c) / 3; r = h - l; w = (h + l + 2 * c) / 4
    x = (h + 2 * l + c) if c < o else (2 * h + l + c) if c > o else (h + l + 2 * c)
    out = {"Classico": lv(p, h, l, c, r), "Fibonacci": [p - r, p - .618 * r, p - .382 * r, p, p + .382 * r, p + .618 * r, p + r],
           "Camarilla": [c - r * 1.1 / 4, c - r * 1.1 / 6, c - r * 1.1 / 12, p, c + r * 1.1 / 12, c + r * 1.1 / 6, c + r * 1.1 / 4],
           "Woodie": lv(w, h, l, c, r), "DeMark": [None, None, x / 2 - h, x / 4, x / 2 - l, None, None]}
    return {k: [f(v, 3) for v in vals] for k, vals in out.items()}
def main(ts):
    try: U = {i["ticker_yf"]: i for i in json.load(urllib.request.urlopen("https://raw.githubusercontent.com/Giorgiogoldoni/core/main/data/tickers_universe.json", timeout=30))["instruments"]}
    except Exception: U = {}
    os.makedirs("data/tech", exist_ok=True); a = {"Open": "first", "High": "max", "Low": "min", "Close": "last"}
    for t in ts:
        d = yf.Ticker(t).history(period="10y", interval="1d", auto_adjust=False).dropna(subset=["Close"])
        if len(d) < 60: print("SKIP", t, len(d)); continue
        d.index = d.index.tz_localize(None); last, prev, y = d.iloc[-1], d.iloc[-2], d.iloc[-252:]
        out = {"ticker": t, "isin": U.get(t, {}).get("isin"), "nome": U.get(t, {}).get("name", t), "valuta": "EUR",
               "borsa": "Borsa Italiana" if t.endswith(".MI") else "Xetra", "generato": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
               "prezzo": f(last.Close, 3), "var": f(last.Close - prev.Close, 3), "var_pct": f((last.Close / prev.Close - 1) * 100, 2),
               "min_gg": f(last.Low, 3), "max_gg": f(last.High, 3), "min_52": f(y.Low.min(), 3), "max_52": f(y.High.max(), 3),
               "tf": {"D": calc(d), "W": calc(d.resample("W-FRI").agg(a).dropna()), "M": calc(d.resample("ME").agg(a).dropna())}, "pivot": pivots(d)}
        json.dump(out, open(f"data/tech/{t.replace('.', '_')}.json", "w"), ensure_ascii=False, allow_nan=False); print("OK", t)
if __name__ == "__main__": main([x.strip() for x in (sys.argv[1] if len(sys.argv) > 1 else "ISPA.DE").split(",") if x.strip()])
