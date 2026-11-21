from __future__ import annotations

from app.database import db
from app.playbook_engine import (
    compute_market_breadth,
    evaluate_live_momentum_board,
    _classify_option_timing,
    compute_order_block,
    _candles_to_df,
)

def run():
    print("Testing Smart Money verification on 17-Nov data...")
    # Load candles for LODHA and DELHIVERY
    c5_lodha = db.load_candles("LODHA", "5m", 100)
    c5_delhivery = db.load_candles("DELHIVERY", "5m", 100)

    print(f"LODHA 5m candles in DB: {len(c5_lodha)}")
    print(f"DELHIVERY 5m candles in DB: {len(c5_delhivery)}")

    if c5_lodha:
        df5 = _candles_to_df(c5_lodha)
        # Check first candle adverse open wick
        o0 = float(df5["open"].iloc[0])
        l0 = float(df5["low"].iloc[0])
        wick_pct = (o0 - l0) / o0 * 100.0
        print(f"LODHA open: {o0}, low0: {l0}, adverse wick: {wick_pct:.2f}%")

        # Check late morning 10:48 AM state
        ltp_1048 = 988.60
        vwap_1048 = 965.0
        c1h = db.load_candles("LODHA", "1h", 60)
        c1d = db.load_candles("LODHA", "1D", 60)
        ob = compute_order_block(df5, "LONG", ltp_1048, vwap_1048, c1d=c1d, c1h=c1h)
        print(f"LODHA at 10:48 AM (LTP ₹{ltp_1048}):")
        print(f"  OB Status: {ob['status']}")
        print(f"  OB Base: {ob['ob_zone']}")
        print(f"  Target R1: ₹{ob['r1']}, S1: ₹{ob['s1']}")
        print(f"  First Break Time: {ob.get('first_break_hhmm')} IST at ₹{ob.get('first_break_px')}")
        
        opt = _classify_option_timing(df5, "LONG", ltp_1048, vwap_1048, (ltp_1048 - o0)/o0*100.0, atr14=25.0, atr_pct=2.6)
        print(f"  Option Phase: {opt['option_phase']}")
        print(f"  Option Label: {opt['option_label']}")
        print(f"  Option Safe: {opt['option_safe']}")
        print(f"  Note: {opt['note']}")

    if c5_delhivery:
        df5_d = _candles_to_df(c5_delhivery)
        o0_d = float(df5_d["open"].iloc[0])
        l0_d = float(df5_d["low"].iloc[0])
        wick_d = (o0_d - l0_d) / o0_d * 100.0
        print(f"\nDELHIVERY open: {o0_d}, low0: {l0_d}, adverse wick: {wick_d:.2f}%")

        # Early morning 09:30 AM bar (bar 3)
        sub_df5 = df5_d.iloc[:4]
        ltp_0930 = float(sub_df5["close"].iloc[-1])
        vwap_0930 = float(sub_df5["close"].mean())
        ob_d = compute_order_block(sub_df5, "LONG", ltp_0930, vwap_0930)
        opt_d = _classify_option_timing(sub_df5, "LONG", ltp_0930, vwap_0930, (ltp_0930 - o0_d)/o0_d*100.0, atr14=12.0, atr_pct=3.0)
        print(f"DELHIVERY at 09:30 AM (LTP ₹{ltp_0930}):")
        print(f"  OB Status: {ob_d['status']}")
        print(f"  Option Phase: {opt_d['option_phase']}")
        print(f"  Option Safe: {opt_d['option_safe']}")

if __name__ == "__main__":
    run()
