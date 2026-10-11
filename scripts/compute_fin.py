"""Scheda analisi per ticker (data/fin/TICKER.json). Metodo PROPRIO: non e' quello di Fineco/Wavenure.
Riusa da compute_tech: universo, segnale core, pivot, ATR-friendly helper. Uso: python scripts/compute_fin.py [ticker,..]  |  --merge"""
import sys, os, json, glob, time, datetime as dt
import numpy as np, pandas as pd, yfinance as yf
sys.path.insert(0, os.path.dirname(__file__)); import compute_tech as T
OUT = "data/fin"; PARTS = "data/fin_parts"; VOL_MIN_EUR = 50000   # sotto questo controvalore medio 20gg i volumi non sono significativi
f = T.f
def nn(x): return x is None or (isinstance(x, float) and np.isnan(x))

def trend(c, p, lag):
    sm = c.rolling(p).mean()
    if nn(sm.iloc[-1]) or len(sm) <= p + lag or nn(sm.iloc[-1 - lag]): return None
    up = c.iloc[-1] > sm.iloc[-1] and sm.iloc[-1] > sm.iloc[-1 - lag]; dn = c.iloc[-1] < sm.iloc[-1] and sm.iloc[-1] < sm.iloc[-1 - lag]
    return "Su" if up else "Giu" if dn else "Laterale"

def stats(d):
    c = d.Close; r = c.pct_change(); v = d.Volume if "Volume" in d else None
    var = [f((c.iloc[-1] / c.iloc[-1 - k] - 1) * 100, 3) if len(c) > k else None for k in (1, 5, 21)]
    vol = [f(r.iloc[-n:].std(ddof=1) * np.sqrt(252) * 100, 3) if len(r) > n else None for n in (5, 10, 21)]
    out = {"var": var, "vol": vol, "vm": [None, None, None], "vm_ok": False}
    if v is not None and v.notna().any():
        m5, m21 = v.iloc[-5:].mean(), v.iloc[-21:].mean(); tu = (v * c).iloc[-20:].mean()
        out["vm"] = [f(m5, 0), f(m21, 0), f(m5 / m21, 3) if m21 else None]; out["vm_ok"] = bool(tu >= VOL_MIN_EUR)
    return out

def range5(c):  # intervallo 5 gg da dev. std. giornaliera a 63 barre; calibrato sui dati del repo (copertura ~70% e ~89%)
    sd = c.pct_change().iloc[-63:].std(ddof=1)
    if nn(sd): return None
    h = sd * np.sqrt(5); p = c.iloc[-1]
    return {"p68": [f(p * (1 - h), 3), f(p * (1 + h), 3)], "p90": [f(p * (1 - 1.645 * h), 3), f(p * (1 + 1.645 * h), 3)]}

def stoch35(d):  # stocastico (3,5): %K su 3 barre, media semplice a 5. Riproduce i valori Fineco su 5 ETF (scarto <=0,1)
    k = 100 * (d.Close - d.Low.rolling(3).min()) / (d.High.rolling(3).max() - d.Low.rolling(3).min()); return f(k.rolling(5).mean().iloc[-1], 2)

def atr(d, n=14):
    pc = d.Close.shift(1); tr = pd.concat([d.High - d.Low, (d.High - pc).abs(), (d.Low - pc).abs()], axis=1).max(axis=1); return T.wil(tr, n).iloc[-1]

def swings(x, w=2):
    hi, lo = [], []
    for i in range(w, len(x) - w):
        seg = x.iloc[i - w:i + w + 1]
        if x.iloc[i] == seg.max(): hi.append(i)
        if x.iloc[i] == seg.min(): lo.append(i)
    return hi, lo

def divergenza(d):  # prezzo vs RSI(14) su ultime 40 barre; valida se l'ultimo swing e' confermato da non piu' di 5 barre
    c = d.Close; up = c.diff(); rsi = 100 - 100 / (1 + T.wil(up.clip(lower=0), 14) / T.wil((-up).clip(lower=0), 14)); x = d.iloc[-40:]; rs = rsi.iloc[-40:]; n = len(x)
    hi, lo = swings(x.High); _, lo2 = swings(x.Low); hi_c, lo_c = swings(x.Close)
    if len(hi_c) >= 2 and n - 1 - hi_c[-1] <= 5 + 2 and x.Close.iloc[hi_c[-1]] > x.Close.iloc[hi_c[-2]] and rs.iloc[hi_c[-1]] < rs.iloc[hi_c[-2]]: return "Ribassista"
    if len(lo_c) >= 2 and n - 1 - lo_c[-1] <= 5 + 2 and x.Close.iloc[lo_c[-1]] < x.Close.iloc[lo_c[-2]] and rs.iloc[lo_c[-1]] > rs.iloc[lo_c[-2]]: return "Rialzista"
    return None

def candele(d):  # ultime 3 barre; ritorna lista di nomi
    out = []; atr_ = d.High.sub(d.Low).iloc[-20:].mean()
    for k in (1, 2, 3):
        if len(d) < k + 2: break
        b = d.iloc[-k]; pb = d.iloc[-k - 1]; rng = b.High - b.Low; body = abs(b.Close - b.Open)
        if rng <= 0: continue
        up = b.High - max(b.Open, b.Close); lw = min(b.Open, b.Close) - b.Low; tag = "" if k == 1 else f" ({k - 1} gg fa)"
        if body <= 0.1 * rng: nome = "Doji"
        elif body >= 0.9 * rng: nome = "Marubozu bianco" if b.Close > b.Open else "Marubozu nero"
        elif lw >= 2 * body and up <= 0.3 * body: nome = "Martello" if d.Close.iloc[-k - 1] < d.Close.iloc[-k - 4] else "Hanging man" if len(d) > k + 4 else None
        elif up >= 2 * body and lw <= 0.3 * body: nome = "Shooting star"
        elif b.Close < b.Open and (b.Close - b.Low) <= 0.02 * rng and body >= 0.4 * rng: nome = "Candela nera con chiusura sul minimo"
        elif b.Close > b.Open and (b.High - b.Close) <= 0.02 * rng and body >= 0.4 * rng: nome = "Candela bianca con chiusura sul massimo"
        elif body >= 0.6 * rng and rng >= 1.3 * atr_: nome = "Candela lunga bianca" if b.Close > b.Open else "Candela lunga nera"
        elif body <= 0.3 * rng: nome = "Spinning top"
        else: nome = None
        if nome is None and b.Close > b.Open and pb.Close < pb.Open and b.Close >= pb.Open and b.Open <= pb.Close: nome = "Engulfing rialzista"
        if nome is None and b.Close < b.Open and pb.Close > pb.Open and b.Open >= pb.Close and b.Close <= pb.Open: nome = "Engulfing ribassista"
        if nome: out.append(nome + tag)
    return out

def stato(core, tr, c, a):
    """Stato del segnale con etichette NOSTRE, dal segnale core. Stop/target: stop = chiusura - 2 ATR, target = chiusura + 3 ATR (rischio/rendimento 1:1,5)."""
    if not core: return None
    s = core["segnale"]; base = {"core": s, "dal": core["dal"], "barre": core["barre"]}
    if str(s).startswith("BUY"):
        base["stato"] = "Acquisto" + (" (nuovo)" if core["barre"] <= 3 else " (in corso)"); lung = True
    elif str(s).startswith("EXIT") or s == "SELL": base["stato"] = "Uscita / non in posizione"; lung = False
    else: base["stato"] = "Neutrale (" + str(s) + ")"; lung = False
    if lung and not nn(a):
        base["stop"] = f(c.iloc[-1] - 2 * a, 3); base["target"] = f(c.iloc[-1] + 3 * a, 3)
    return base

def it(x, d=2): return f"{x:.{d}f}".replace(".", ",")

def commento(o):
    """Commento a regole; se due letture sono in conflitto lo dice invece di scegliere."""
    s, p = [], o["prezzo"]; v = o["var_pct"]; ind = o["ind"]; tr = o["tend"]
    s.append(f"Seduta {'positiva' if v > 0 else 'negativa' if v < 0 else 'invariata'} ({it(v)}%), chiusura a {it(p, 3)} (barra del {o['ultima']}).")
    if o["candele"]: s.append("Candela: " + o["candele"][0] + ".")
    if o["stats"]["vm_ok"] and o["stats"]["vm"][2]: s.append("Volumi " + ("in aumento" if o["stats"]["vm"][2] > 1.1 else "in calo" if o["stats"]["vm"][2] < 0.9 else "stabili") + " rispetto alla media a 21 giorni.")
    elif not o["stats"]["vm_ok"]: s.append("Volumi troppo bassi per essere significativi: non li usiamo nella lettura.")
    s.append(f"Tendenza breve {tr['breve'] or 'n.d.'}, media {tr['medio'] or 'n.d.'}, lunga {tr['lungo'] or 'n.d.'}.")
    conf = []
    rsi, st = ind.get("rsi"), ind.get("stoch")
    if rsi is not None and st is not None:
        if rsi > 70 or st > 80: s.append(f"RSI {it(rsi, 1)} e stocastico {it(st, 1)}: zona di ipercomprato, possibile pausa o ritracciamento.")
        elif rsi < 30 or st < 20: s.append(f"RSI {it(rsi, 1)} e stocastico {it(st, 1)}: zona di ipervenduto, possibile rimbalzo.")
        else: s.append(f"RSI {it(rsi, 1)} e stocastico {it(st, 1)}: nessuna zona estrema.")
    if o["divergenza"]: s.append(f"Divergenza {o['divergenza'].lower()} tra prezzo e RSI nelle ultime sedute: da confermare dal movimento successivo.")
    stt = o["stato"]
    if stt and stt["stato"].startswith("Acquisto"):
        if (rsi is not None and rsi > 70) or tr["breve"] == "Giu": conf.append("il segnale di acquisto convive con " + ("ipercomprato" if rsi and rsi > 70 else "tendenza breve debole") + ": cautela")
        s.append(f"Segnale core {stt['core']} dal {stt['dal']}; stop indicativo {it(stt['stop'], 3)}, target indicativo {it(stt['target'], 3)}." if stt.get("stop") else f"Segnale core {stt['core']} dal {stt['dal']}.")
    elif stt: s.append(f"Segnale core {stt['core']} dal {stt['dal']}: nessuna posizione indicata.")
    if tr["lungo"] == "Su" and tr["breve"] == "Giu": conf.append("trend lungo rialzista ma breve ribassista")
    if tr["lungo"] == "Giu" and tr["breve"] == "Su": conf.append("trend lungo ribassista ma breve rialzista")
    if conf: s.append("Letture in conflitto: " + "; ".join(conf) + ".")
    s.append("Lettura indicativa a regole proprie, non una raccomandazione.")
    return " ".join(s)

def build(t, d, U, ok):
    c = d.Close; u = U.get(t, {}); last, prev = d.iloc[-1], d.iloc[-2]; a = atr(d); x = T.calc(d)
    ind = {i["nome"]: i["valore"] for i in x["indicatori"]} if x else {}
    o = {"ticker": t, "isin": u.get("isin"), "nome": u.get("name", t), "borsa": "Borsa Italiana" if t.endswith(".MI") else "Xetra" if t.endswith(".DE") else t.split(".")[-1], "valuta": "EUR",
         "ultima": str(d.index[-1].date()), "aggiornato": ok, "generato": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
         "prezzo": f(last.Close, 3), "var": f(last.Close - prev.Close, 3), "var_pct": f((last.Close / prev.Close - 1) * 100, 2), "apertura": f(last.Open, 3), "min_gg": f(last.Low, 3), "max_gg": f(last.High, 3),
         "tend": {"breve": trend(c, 5, 2), "medio": trend(c, 21, 5), "lungo": trend(c, 63, 10)}, "stats": stats(d), "range5": range5(c), "pivot": T.pivots(d)["Classico"],
         "ind": {"rsi": ind.get("RSI(14)"), "stoch": stoch35(d)}, "divergenza": divergenza(d), "candele": candele(d), "stato": stato(T.core_sig(d), None, c, a), "atr": f(a, 4)}
    o["commento"] = commento(o); return o

def run(ts, tag):
    U = T.loadU(); res = []; fails = []; os.makedirs(OUT, exist_ok=True); os.makedirs(PARTS, exist_ok=True)
    for t in ts:
        try:
            d = yf.Ticker(t).history(period="2y", interval="1d", auto_adjust=False).dropna(subset=["Close"])
            if len(d) < 70: fails.append({"ticker": t, "motivo": f"storia insufficiente ({len(d)})"}); continue
            d.index = d.index.tz_localize(None); d, ok = T.recent(t, d); o = build(t, d, U, ok)
            json.dump(o, open(f"{OUT}/{t.replace('.', '_')}.json", "w"), ensure_ascii=False, allow_nan=False)
            res.append({"ticker": t, "isin": o["isin"], "nome": o["nome"], "borsa": o["borsa"], "ultima": o["ultima"], "agg": ok}); print("OK", t)
        except Exception as e: print("ERR", t, repr(e)[:100]); fails.append({"ticker": t, "motivo": repr(e)[:120]})
        time.sleep(0.5)
    json.dump(res, open(f"{PARTS}/part_{tag}.json", "w"), ensure_ascii=False); json.dump(fails, open(f"{PARTS}/falliti_{tag}.json", "w"), ensure_ascii=False)

def merge():
    rows = [r for p in sorted(glob.glob(f"{PARTS}/part_*.json")) for r in json.load(open(p))]
    if not rows: print("NESSUN DATO: indice non scritto"); sys.exit(1)
    os.makedirs(OUT, exist_ok=True)
    json.dump([x for p in sorted(glob.glob(f"{PARTS}/falliti_*.json")) for x in json.load(open(p))], open(f"{OUT}/falliti.json", "w"), ensure_ascii=False)
    json.dump({"generato": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC"), "righe": rows}, open(f"{OUT}/index.json", "w"), ensure_ascii=False); print("INDICE", len(rows))

if __name__ == "__main__":
    if "--merge" in sys.argv: merge(); sys.exit()
    a = sys.argv[1].strip() if len(sys.argv) > 1 else ""; sh = os.environ.get("SHARD", "")
    if a:
        if sh and sh.split("/")[0] != "0": sys.exit()
        run([x.strip() for x in a.split(",") if x.strip()], "manuale")
    elif sh:
        i, n = map(int, sh.split("/")); camp = {x.strip() for x in open("campione.txt") if x.strip()}; run(sorted(set(T.loadU()) | camp)[i::n], str(i))
    else: run([x.strip() for x in open("campione.txt") if x.strip()], "campione")
