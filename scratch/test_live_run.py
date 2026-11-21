import asyncio
from app.scanner import scanner
from app.database import db

async def test_live_refresh():
    res = await scanner.refresh_live_momentum()
    print("Refresh result:")
    print("  Board items:", res.get("count"))
    print("  New alerts:", res.get("new_alerts_count"))
    
    alerts = db.get_live_alerts(trade_date="2026-11-20")
    print("  Alerts in DB for 2026-11-20:", len(alerts))
    for a in alerts[:5]:
        print("   Alert:", a.get("symbol"), a.get("event_type"), a.get("price"), a.get("trigger_time"))

asyncio.run(test_live_refresh())
