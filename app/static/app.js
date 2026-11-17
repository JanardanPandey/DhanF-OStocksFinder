(() => {
  'use strict';

  const state = {
    setups: [],
    breadth: null,
    watchlist: [],
    liveMomentum: [],
    watchlistDates: [],
    watchlistDate: '',
    selectedSymbol: null,
    selectedSetup: null,
    timeframe: '15m',
    portfolio: null,
    positions: [],
    trades: [],
    universe: [],
    liveAlerts: [],
    knownAlertIds: new Set(),
    soundEnabled: true,
    initialAlertsLoaded: false,
    chart: null,
    candleSeries: null,
    volumeSeries: null,
    priceLines: [],
    chartData: null,
    chartLayers: {
      trade: true,
      sr: false,
      zones: false,
      signals: false,
    },
  };

  let audioContext = null;
  function playAlertChime() {
    if (!state.soundEnabled) return;
    try {
      const AudioCtx = window.AudioContext || window.webkitAudioContext;
      if (!AudioCtx) return;
      if (!audioContext) audioContext = new AudioCtx();
      if (audioContext.state === 'suspended') {
        audioContext.resume();
      }
      const now = audioContext.currentTime;
      const osc = audioContext.createOscillator();
      const gain = audioContext.createGain();
      osc.type = 'sine';
      osc.frequency.setValueAtTime(880, now);
      osc.frequency.exponentialRampToValueAtTime(1320, now + 0.12);
      gain.gain.setValueAtTime(0.18, now);
      gain.gain.exponentialRampToValueAtTime(0.001, now + 0.35);
      osc.connect(gain);
      gain.connect(audioContext.destination);
      osc.start(now);
      osc.stop(now + 0.35);
    } catch {
      // Audio autoplay policy
    }
  }

  const collator = new Intl.Collator(undefined, { numeric: true, sensitivity: 'base' });
  const fmtInr = (n) =>
    '₹' + Number(n || 0).toLocaleString('en-IN', { maximumFractionDigits: 2 });

  async function api(url, options = {}) {
    const isGet = (options.method || 'GET').toUpperCase() === 'GET';
    const reqUrl = isGet
      ? url + (url.includes('?') ? '&' : '?') + `_t=${Date.now()}`
      : url;
    const res = await fetch(reqUrl, {
      headers: { 'Content-Type': 'application/json', ...(options.headers || {}) },
      cache: 'no-store',
      ...options,
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.detail || `Request failed (${res.status})`);
    return data;
  }

  function toast(msg) {
    const region = document.getElementById('toast-region');
    const el = document.createElement('div');
    el.className = 'toast';
    el.textContent = msg;
    region.appendChild(el);
    setTimeout(() => el.remove(), 4200);
  }

  /* ------------------ Accessible Tabs ------------------ */
  function initTabs() {
    const tabs = [...document.querySelectorAll('[role="tab"]')];
    const panels = [...document.querySelectorAll('[role="tabpanel"]')];

    function activate(tab) {
      for (const t of tabs) {
        const active = t === tab;
        t.setAttribute('aria-selected', String(active));
        t.tabIndex = active ? 0 : -1;
      }
      for (const p of panels) {
        p.hidden = p.id !== tab.dataset.panel;
      }
      if (tab.dataset.panel === 'panel-ideas' && state.chart) {
        requestAnimationFrame(() => resizeChart());
      }
      if (tab.dataset.panel === 'panel-watchlist') {
        loadWatchlist().catch((err) => toast(err.message));
        loadLiveMomentum().catch((err) => toast(err.message));
      }
      if (tab.dataset.panel === 'panel-alerts') {
        loadLiveAlerts().catch((err) => toast(err.message));
      }
    }

    tabs.forEach((tab, idx) => {
      tab.addEventListener('click', () => activate(tab));
      tab.addEventListener('keydown', (e) => {
        let next = null;
        if (e.key === 'ArrowRight') next = tabs[(idx + 1) % tabs.length];
        if (e.key === 'ArrowLeft') next = tabs[(idx - 1 + tabs.length) % tabs.length];
        if (next) {
          e.preventDefault();
          next.focus();
          activate(next);
        }
      });
    });
  }

  /* ------------------ Sortable Tables ------------------ */
  function initSortableTable(tableId, statusId) {
    const table = document.getElementById(tableId);
    const statusEl = document.getElementById(statusId);
    if (!table) return;

    const buttons = table.querySelectorAll('thead th button.sort-btn');
    buttons.forEach((btn) => {
      btn.addEventListener('click', () => {
        const th = btn.closest('th');
        const tbody = table.tBodies[0];
        const dir = th.getAttribute('aria-sort') === 'ascending' ? 'descending' : 'ascending';
        for (const h of table.tHead.rows[0].cells) h.removeAttribute('aria-sort');
        th.setAttribute('aria-sort', dir);

        const factor = dir === 'ascending' ? 1 : -1;
        const type = th.dataset.type || 'text';
        const colIdx = th.cellIndex;
        const rows = [...tbody.rows];
        rows.sort((a, b) => {
          const va = a.cells[colIdx]?.dataset.value ?? a.cells[colIdx]?.textContent.trim() ?? '';
          const vb = b.cells[colIdx]?.dataset.value ?? b.cells[colIdx]?.textContent.trim() ?? '';
          if (type === 'number') return factor * (Number(va) - Number(vb));
          return factor * collator.compare(va, vb);
        });

        for (const row of rows) {
          if (typeof tbody.moveBefore === 'function') tbody.moveBefore(row, null);
          else tbody.appendChild(row);
        }
        if (statusEl) {
          statusEl.textContent = `Sorted by ${btn.textContent.trim()}, ${dir}, ${rows.length} rows`;
        }
      });
    });
  }

  /* ------------------ Native Dialog Light Dismiss Fallback ------------------ */
  function initDialogs() {
    if (!('closedBy' in HTMLDialogElement.prototype)) {
      const outside = (e, dialog) => {
        const r = dialog.getBoundingClientRect();
        return e.clientX < r.left || e.clientX > r.right || e.clientY < r.top || e.clientY > r.bottom;
      };
      for (const dialog of document.querySelectorAll('dialog[closedby="any"]')) {
        let downOutside = false;
        dialog.addEventListener('pointerdown', (e) => {
          downOutside = outside(e, dialog);
        });
        dialog.addEventListener('pointerup', (e) => {
          if (downOutside && outside(e, dialog)) dialog.close();
        });
      }
    }
    for (const dialog of document.querySelectorAll('dialog')) {
      dialog.addEventListener('close', () => {
        dialog.returnValue = '';
      });
    }
  }

  /* ------------------ Charting (Lightweight Charts) ------------------ */
  function ensureChart() {
    const container = document.getElementById('chart-container');
    if (state.chart || !window.LightweightCharts) return;

    state.chart = window.LightweightCharts.createChart(container, {
      layout: {
        background: { color: '#0f141f' },
        textColor: '#93a1bd',
      },
      grid: {
        vertLines: { color: 'rgba(37, 48, 71, 0.4)' },
        horzLines: { color: 'rgba(37, 48, 71, 0.4)' },
      },
      timeScale: {
        timeVisible: true,
        secondsVisible: false,
        borderColor: '#253047',
        rightOffset: 4,
        barSpacing: 10,
        minBarSpacing: 3,
      },
      rightPriceScale: { borderColor: '#253047' },
      crosshair: { mode: 0 },
    });

    state.candleSeries = state.chart.addCandlestickSeries({
      upColor: '#10b981',
      downColor: '#ef4444',
      borderVisible: false,
      wickUpColor: '#10b981',
      wickDownColor: '#ef4444',
    });

    state.volumeSeries = state.chart.addHistogramSeries({
      priceFormat: { type: 'volume' },
      priceScaleId: '',
    });
    state.volumeSeries.priceScale().applyOptions({
      scaleMargins: { top: 0.82, bottom: 0 },
    });

    const ro = new ResizeObserver(() => resizeChart());
    ro.observe(container);
  }

  function resizeChart() {
    const container = document.getElementById('chart-container');
    if (state.chart && container && container.clientWidth > 0) {
      state.chart.applyOptions({
        width: container.clientWidth,
        height: container.clientHeight || 540,
      });
    }
  }

  async function openChartModal(symbol, preferredTf = null, setupOverride = null) {
    if (!symbol) return;
    const dlg = document.getElementById('chart-dialog');
    if (dlg && !dlg.open) {
      dlg.showModal();
    }
    await loadChart(symbol, preferredTf, setupOverride);
    requestAnimationFrame(() => {
      resizeChart();
      state.chart?.timeScale().fitContent();
    });
  }

  function clearPriceLines() {
    if (!state.candleSeries) return;
    for (const pl of state.priceLines) {
      state.candleSeries.removePriceLine(pl);
    }
    state.priceLines = [];
  }

  function addLine(price, color, title, style = 2, axisLabelVisible = false, lineWidth = 1) {
    if (!state.candleSeries || !price) return;
    const pl = state.candleSeries.createPriceLine({
      price: Number(price),
      color,
      lineWidth,
      lineStyle: style,
      axisLabelVisible,
      title,
    });
    state.priceLines.push(pl);
  }

  function renderChartOverlays() {
    if (!state.candleSeries || !state.chartData) return;
    const data = state.chartData;
    const struct = data.structure || {};
    const activeSetup = state.selectedSetup;
    const layers = state.chartLayers;
    const istOffset = 19800;

    clearPriceLines();
    state.candleSeries.setMarkers([]);

    const drawnPrices = [];
    const nearExisting = (p) =>
      drawnPrices.some((x) => x > 0 && Math.abs(p - x) / x < 0.0025);
    const drawFiltered = (price, color, title, style, axisLabel, width = 1) => {
      const num = Number(price);
      if (!num || nearExisting(num)) return;
      drawnPrices.push(num);
      addLine(num, color, title, style, axisLabel, width);
    };

    const noteEl = document.getElementById('chart-layer-note');
    const anyExtraOn = layers.sr || layers.zones || layers.signals;

    // 1) Setup Trade Levels (clean Y-axis labels only for trade plan)
    if (layers.trade && activeSetup) {
      drawFiltered(activeSetup.entry_price, '#3b82f6', 'ENTRY', 0, true, 2);
      drawFiltered(activeSetup.stop_loss, '#ef4444', `SL ${activeSetup.sl_pct}%`, 2, true, 1);
      drawFiltered(activeSetup.tp1, '#22c55e', 'TP1', 3, true, 1);
      drawFiltered(activeSetup.tp2, '#10b981', `TP2 ${activeSetup.rr_ratio}R`, 3, true, 1);
    }

    // 2) Nearest 1 Support + 1 Resistance (auto-fallback if no setup & no extras active)
    const showFallbackSr = !activeSetup && !layers.sr && !layers.zones && !layers.signals;
    if (layers.sr || showFallbackSr) {
      const candles = data.candles || [];
      const ltp = candles.length ? candles[candles.length - 1].close : 0;
      const srList = struct.sr_zones || [];
      const sups = srList.filter((z) => z.kind === 'SUPPORT');
      const ress = srList.filter((z) => z.kind !== 'SUPPORT');
      const nearestSup = ltp
        ? sups.filter((z) => z.mid <= ltp).sort((a, b) => b.mid - a.mid)[0] || sups[0]
        : sups[0];
      const nearestRes = ltp
        ? ress.filter((z) => z.mid >= ltp).sort((a, b) => a.mid - b.mid)[0] || ress[0]
        : ress[0];
      const showSrAxis = !activeSetup || !layers.trade;
      if (nearestSup) drawFiltered(nearestSup.mid, '#64748b', 'SUP', 2, showSrAxis, 1);
      if (nearestRes) drawFiltered(nearestRes.mid, '#94a3b8', 'RES', 2, showSrAxis, 1);
    }

    // 3) Latest 1 OB + Latest 1 FVG (hidden axis pills to keep scale clutter-free)
    if (layers.zones) {
      const obs = struct.order_blocks || [];
      const dirPref = activeSetup?.direction === 'SHORT' ? 'BEARISH' : activeSetup?.direction === 'LONG' ? 'BULLISH' : null;
      const pickedOb = (dirPref && obs.filter((o) => o.direction === dirPref).slice(-1)[0]) || obs.slice(-1)[0];
      if (pickedOb) {
        const obColor = pickedOb.direction === 'BULLISH' ? '#10b981' : '#ef4444';
        const obTag = pickedOb.direction === 'BULLISH' ? 'OB+' : 'OB-';
        drawFiltered(pickedOb.high, obColor, `${obTag} Hi`, 1, false, 1);
        drawFiltered(pickedOb.low, obColor, `${obTag} Lo`, 1, false, 1);
      }
      const fvg = (struct.fvgs || []).slice(-1)[0];
      if (fvg) {
        drawFiltered(fvg.high, '#f59e0b', 'FVG Hi', 3, false, 1);
        drawFiltered(fvg.low, '#f59e0b', 'FVG Lo', 3, false, 1);
      }
    }

    // 4) Deduplicated, capped BOS / CHoCH / Sweep markers (max 1 marker per candle)
    if (layers.signals) {
      const byTime = new Map();
      const sweeps = (struct.sweeps || []).slice(-6);
      let keptSweeps = 0;
      let lastSweepTs = 0;
      for (let i = sweeps.length - 1; i >= 0 && keptSweeps < 2; i--) {
        const sw = sweeps[i];
        const ts = sw.ts + istOffset;
        if (!lastSweepTs || Math.abs(lastSweepTs - ts) > 0) {
          byTime.set(ts, {
            time: ts,
            position: sw.direction === 'BULLISH' ? 'belowBar' : 'aboveBar',
            color: '#38bdf8',
            shape: 'square',
            text: 'SWP',
          });
          lastSweepTs = ts;
          keptSweeps += 1;
        }
      }
      for (const b of (struct.bos || []).slice(-2)) {
        const ts = b.ts + istOffset;
        byTime.set(ts, {
          time: ts,
          position: b.direction === 'BULLISH' ? 'aboveBar' : 'belowBar',
          color: b.direction === 'BULLISH' ? '#10b981' : '#ef4444',
          shape: b.direction === 'BULLISH' ? 'arrowUp' : 'arrowDown',
          text: 'BOS',
        });
      }
      for (const c of (struct.choch || []).slice(-2)) {
        const ts = c.ts + istOffset;
        byTime.set(ts, {
          time: ts,
          position: c.direction === 'BULLISH' ? 'belowBar' : 'aboveBar',
          color: '#f59e0b',
          shape: 'circle',
          text: 'CH',
        });
      }
      const markers = Array.from(byTime.values()).sort((a, b) => a.time - b.time);
      state.candleSeries.setMarkers(markers);
    }

    if (noteEl) {
      if (showFallbackSr) {
        noteEl.textContent = 'No setup triggered · Showing nearest S/R';
      } else if (!anyExtraOn && activeSetup) {
        noteEl.textContent = 'Clean view · Toggle S/R, OB/FVG or BOS to inspect';
      } else {
        noteEl.textContent = '';
      }
    }
  }

  async function loadChart(symbol, preferredTf = null, setupOverride = null) {
    if (!symbol) return;
    state.selectedSymbol = symbol;
    if (preferredTf) {
      state.timeframe = preferredTf;
      document.querySelectorAll('.tf-btn').forEach((b) => {
        b.classList.toggle('active', b.dataset.tf === state.timeframe);
      });
    }

    try {
      const data = await api(`/api/chart/${encodeURIComponent(symbol)}?timeframe=${state.timeframe}`);
      const struct = data.structure || {};
      const activeSetup = setupOverride || (data.setups && data.setups[0]) || null;
      state.selectedSetup = activeSetup;
      state.chartData = data;

      document.getElementById('chart-symbol-title').textContent =
        `${data.symbol} · ${struct.trend || 'RANGING'} (${struct.structure_label || ''})`;
      document.getElementById('chart-symbol-meta').textContent =
        `${data.name} · Lot: ${data.lot_size} · TF: ${data.timeframe} · RSI: ${struct.rsi} · VWAP: ₹${struct.vwap} · Source: ${data.data_source.toUpperCase()}`;

      ensureChart();
      if (state.candleSeries && data.candles?.length) {
        // Shift timestamps by +19800s so Lightweight Charts renders IST cleanly
        const istOffset = 19800;
        const candles = data.candles.map((c) => ({
          time: c.time + istOffset,
          open: c.open,
          high: c.high,
          low: c.low,
          close: c.close,
        }));
        const vols = data.candles.map((c) => ({
          time: c.time + istOffset,
          value: c.volume,
          color: c.close >= c.open ? 'rgba(16, 185, 129, 0.35)' : 'rgba(239, 68, 68, 0.35)',
        }));
        state.candleSeries.setData(candles);
        state.volumeSeries.setData(vols);

        renderChartOverlays();
        resizeChart();
        state.chart.timeScale().fitContent();
      }

      renderSelectedSetupDetails(activeSetup, data);
    } catch (err) {
      toast(err.message);
    }
  }

  function renderSelectedSetupDetails(setup, chartData) {
    const box = document.getElementById('selected-setup-details');
    if (!setup) {
      box.innerHTML = `
        <p class="muted">No tight-SL setup triggered on ${chartData.symbol} right now. Inspecting raw OB, FVG &amp; S/R structure.</p>
      `;
      return;
    }

    const dirClass = setup.direction === 'LONG' ? 'tag-long' : 'tag-short';
    const reasonsHtml = (setup.reasons || []).map((r) => `<li>${escapeHtml(r)}</li>`).join('');
    const rankNum = setup.rank ? `#${setup.rank}` : '—';
    const pbTag = setup.playbook_tag || 'SMC_OB';
    const monsterBadge = setup.monster_runner
      ? '<span class="badge badge-monster">MONSTER DNA</span>'
      : '';
    const antiBadge = setup.anti_chasing
      ? '<span class="badge badge-anti">ANTI-CHASING</span>'
      : '';

    box.innerHTML = `
      <div class="row-actions" style="justify-content: space-between; flex-wrap: wrap; gap: 0.4rem;">
        <div class="row-actions" style="flex-wrap: wrap; gap: 0.35rem;">
          <span class="rank-pill ${Number(setup.rank) <= 3 ? 'rank-top' : ''}">Rank ${rankNum}</span>
          <span class="badge ${dirClass}">${setup.direction}</span>
          <span class="badge ${pbTag === 'CREAM_BULL_V2' ? 'badge-cream' : pbTag === 'STRICT_BEAR_V2' ? 'badge-bear' : ''}">${escapeHtml(pbTag)}</span>
          ${monsterBadge}
          ${antiBadge}
          <span class="badge ${setup.grade === 'A+' ? 'grade-aplus' : 'grade-a'}">${setup.grade} (${setup.score}/100)</span>
        </div>
        <button type="button" class="btn btn-primary btn-sm" id="btn-take-selected-trade">
          Take Paper Trade
        </button>
      </div>
      <div class="detail-grid">
        <div class="detail-metric"><span>Trigger Time</span><strong>${escapeHtml(setup.trigger_time || '—')}</strong></div>
        <div class="detail-metric"><span>Entry (${escapeHtml(setup.entry_mode)})</span><strong>₹${setup.entry_price}</strong></div>
        <div class="detail-metric"><span>Tight SL (${setup.sl_pct}%)</span><strong class="pnl-neg">₹${setup.stop_loss}</strong></div>
        <div class="detail-metric"><span>TP1 (50% + BE)</span><strong class="pnl-pos">₹${setup.tp1}</strong></div>
        <div class="detail-metric"><span>TP2 (${setup.rr_ratio}R)</span><strong class="pnl-pos">₹${setup.tp2}</strong></div>
        <div class="detail-metric"><span>Qty / F&amp;O Lot (${setup.lot_size})</span><strong>${setup.suggested_qty} sh (~${setup.fno_lots_equiv} lot)</strong></div>
      </div>
      <ul class="reason-list">${reasonsHtml}</ul>
    `;

    document.getElementById('btn-take-selected-trade')?.addEventListener('click', () => {
      openOrderModalFromSetup(setup);
    });
  }

  function escapeHtml(str) {
    return String(str)
      .replaceAll('&', '&amp;')
      .replaceAll('<', '&lt;')
      .replaceAll('>', '&gt;')
      .replaceAll('"', '&quot;');
  }

  function renderBreadth(breadth, setups) {
    state.breadth = breadth || null;
    const vwapEl = document.getElementById('breadth-vwap-pct');
    const openEl = document.getElementById('breadth-open-pct');
    const advEl = document.getElementById('breadth-adv-pct');
    const regEl = document.getElementById('breadth-regime-badge');
    const topBtn = document.getElementById('btn-jump-rank1');
    const liqEl = document.getElementById('breadth-liquidity-counts');

    if (!breadth) {
      if (liqEl) liqEl.textContent = '—';
      vwapEl.textContent = '—';
      openEl.textContent = '—';
      advEl.textContent = '—';
      regEl.textContent = '—';
      regEl.className = 'badge';
    } else {
      if (liqEl) {
        const bull = breadth.bullish_count || 0;
        const bear = breadth.bearish_count || 0;
        const macro = breadth.macro_bias ? ` (${breadth.macro_bias.replace('_', ' ')})` : '';
        liqEl.innerHTML = `<span style="color:#10b981;">🟢 ${bull} Bull</span> · <span style="color:#ef4444;">🔴 ${bear} Bear</span> <span style="font-size:0.72rem;font-weight:normal;color:#94a3b8;">${macro}</span>`;
      }
      const v = Number(breadth.abv_vwap_pct || 0);
      vwapEl.textContent = `${v}% (${breadth.sample_size || 0})`;
      vwapEl.className = v > 55 ? 'pnl-pos' : v < 45 ? 'pnl-neg' : '';
      openEl.textContent = `${breadth.green_open_pct ?? '—'}%`;
      advEl.textContent = `${breadth.nse_adv_pct ?? '—'}%`;
      const reg = breadth.regime || 'BALANCED';
      if (reg === 'BULL_FAVORABLE') {
        regEl.textContent = 'BULL (>55% · Bear Blocked)';
        regEl.className = 'badge badge-cream';
      } else if (reg === 'BEAR_FAVORABLE') {
        regEl.textContent = 'BEAR (<45% · Short Active)';
        regEl.className = 'badge badge-bear';
      } else {
        regEl.textContent = 'BALANCED (45–55%)';
        regEl.className = 'badge';
      }
    }

    const top = setups && setups.length ? setups[0] : null;
    if (top) {
      topBtn.textContent = `#${top.rank || 1} ${top.symbol} (${top.direction} · ${top.score})`;
      topBtn.disabled = false;
    } else {
      topBtn.textContent = 'No setups yet';
      topBtn.disabled = true;
    }
  }

  /* ------------------ Setups Scanner Rendering ------------------ */
  async function loadSetups() {
    const playbook = document.getElementById('filter-playbook')?.value || 'ALL';
    const style = document.getElementById('filter-style').value;
    const direction = document.getElementById('filter-dir').value;
    const grade = document.getElementById('filter-grade').value;
    const data = await api(
      `/api/setups?playbook=${encodeURIComponent(playbook)}&style=${style}&direction=${direction}&grade=${grade}`
    );
    state.setups = data.items || [];
    renderBreadth(data.breadth, state.setups);

    document.getElementById('ideas-count-badge').textContent = `${state.setups.length} setups`;
    const scanAt = data.scanner?.last_scan_at
      ? new Date(data.scanner.last_scan_at).toLocaleTimeString('en-IN')
      : '—';
    document.getElementById('scan-meta-text').textContent =
      `Last scan: ${scanAt} · Source: ${(data.scanner?.data_source || 'demo').toUpperCase()}`;

    const tbody = document.getElementById('ideas-tbody');
    tbody.innerHTML = '';

    for (const s of state.setups) {
      const tr = document.createElement('tr');
      if (state.selectedSetup?.id === s.id) tr.classList.add('row-selected');
      const rank = Number(s.rank || 999);
      const dirCls = s.direction === 'LONG' ? 'tag-long' : 'tag-short';
      const gradeCls = s.grade === 'A+' ? 'grade-aplus' : s.grade === 'A' ? 'grade-a' : '';
      const pbTag = s.playbook_tag || 'SMC_OB';
      const pbCls = pbTag === 'CREAM_BULL_V2' ? 'badge-cream' : pbTag === 'STRICT_BEAR_V2' ? 'badge-bear' : '';
      const monsterBadge = s.monster_runner ? ' <span class="badge badge-monster">MONSTER</span>' : '';
      const reasonShort = s.reason_summary || (s.reasons || []).slice(0, 2).join(' | ');

      tr.innerHTML = `
        <td data-value="${rank}"><span class="rank-pill ${rank <= 3 ? 'rank-top' : ''}">#${rank}</span></td>
        <td data-value="${s.symbol}"><strong>${s.symbol}</strong><div class="muted" style="font-size:0.72rem;">${s.style}</div></td>
        <td data-value="${escapeHtml(pbTag + ' ' + (s.trigger_time || ''))}">
          <div><span class="badge ${pbCls}">${escapeHtml(pbTag)}</span>${monsterBadge}</div>
          <div class="trigger-cell">${escapeHtml(s.trigger_time || '—')}</div>
        </td>
        <td data-value="${s.direction}"><span class="badge ${dirCls}">${s.direction}</span></td>
        <td data-value="${s.score}"><span class="badge ${gradeCls}">${s.grade} · ${s.score}</span></td>
        <td data-value="${s.entry_price}">₹${s.entry_price}</td>
        <td data-value="${s.sl_pct}">${s.sl_pct}%</td>
        <td data-value="${s.rr_ratio}">1:${s.rr_ratio}</td>
        <td data-value="${escapeHtml(reasonShort)}" class="reason-cell" title="${escapeHtml((s.reasons || []).join('\n'))}">${escapeHtml(reasonShort)}</td>
        <td><button type="button" class="btn btn-primary btn-sm btn-paper-row">Paper</button></td>
      `;

      tr.addEventListener('click', (e) => {
        if (e.target.closest('.btn-paper-row')) return;
        tbody.querySelectorAll('tr').forEach((r) => r.classList.remove('row-selected'));
        tr.classList.add('row-selected');
        const tf = s.style === 'SWING' ? '1h' : '15m';
        openChartModal(s.symbol, tf, s);
      });

      tr.querySelector('.btn-paper-row').addEventListener('click', () => {
        openOrderModalFromSetup(s);
      });

      tbody.appendChild(tr);
    }
  }

  /* ------------------ Paper Portfolio Rendering ------------------ */
  async function loadPortfolio() {
    const data = await api('/api/paper/portfolio');
    renderPortfolioData(data.portfolio, data.positions, data.trades);
  }

  function renderPortfolioData(p, positions, trades) {
    state.portfolio = p;
    state.positions = positions || [];
    if (trades) state.trades = trades;

    document.getElementById('header-equity').textContent = fmtInr(p.equity);
    const hdrPnl = document.getElementById('header-pnl');
    hdrPnl.textContent = `${p.total_pnl >= 0 ? '+' : ''}${fmtInr(p.total_pnl)} (${p.return_pct}%)`;
    hdrPnl.className = p.total_pnl >= 0 ? 'pnl-pos' : 'pnl-neg';

    document.getElementById('tab-open-count').textContent = String(p.open_count);
    document.getElementById('kpi-equity').textContent = fmtInr(p.equity);
    const kReal = document.getElementById('kpi-realized');
    kReal.textContent = fmtInr(p.realized_pnl);
    kReal.className = p.realized_pnl >= 0 ? 'pnl-pos' : 'pnl-neg';

    const kUnr = document.getElementById('kpi-unrealized');
    kUnr.textContent = fmtInr(p.unrealized_pnl);
    kUnr.className = p.unrealized_pnl >= 0 ? 'pnl-pos' : 'pnl-neg';

    document.getElementById('kpi-winrate').textContent = `${p.win_rate}% (${p.closed_trades_count})`;
    document.getElementById('kpi-pf').textContent = String(p.profit_factor);
    document.getElementById('kpi-avgr').textContent = `${p.avg_r_multiple}R`;

    // Sync risk inputs
    if (p.settings) {
      document.getElementById('cfg-risk-inr').value = p.settings.risk_per_trade_inr;
      document.getElementById('cfg-min-rr').value = p.settings.min_rr_ratio;
      document.getElementById('cfg-max-sl-intra').value = p.settings.max_sl_pct_intraday;
      document.getElementById('cfg-max-sl-swing').value = p.settings.max_sl_pct_swing;
      document.getElementById('cfg-auto-paper').checked = Boolean(p.settings.auto_paper_trade);
    }

    const posBody = document.getElementById('positions-tbody');
    posBody.innerHTML = '';
    if (!state.positions.length) {
      posBody.innerHTML = `<tr><td colspan="11" class="muted">No open paper positions. Click "Paper" on any setup to start tracking.</td></tr>`;
    } else {
      for (const pos of state.positions) {
        const tr = document.createElement('tr');
        const dirCls = pos.direction === 'LONG' ? 'tag-long' : 'tag-short';
        const pnlCls = pos.unrealized_pnl >= 0 ? 'pnl-pos' : 'pnl-neg';
        tr.innerHTML = `
          <td><strong>${pos.symbol}</strong></td>
          <td>${pos.style}</td>
          <td><span class="badge ${dirCls}">${pos.direction}</span></td>
          <td>${pos.remaining_qty}/${pos.initial_qty}</td>
          <td>₹${pos.entry_price}</td>
          <td>₹${pos.current_price}</td>
          <td>₹${pos.stop_loss} <span class="muted">(₹${pos.initial_stop_loss})</span></td>
          <td>₹${pos.tp1} / ₹${pos.tp2}</td>
          <td class="${pnlCls}">${fmtInr(pos.unrealized_pnl)}</td>
          <td class="${pnlCls}">${pos.current_r}R</td>
          <td>
            <div class="row-actions">
              <button type="button" class="btn btn-ghost btn-sm act-be">SL→BE</button>
              <button type="button" class="btn btn-danger-ghost btn-sm act-close">Close</button>
            </div>
          </td>
        `;
        tr.querySelector('.act-be').addEventListener('click', async () => {
          await api(`/api/paper/positions/${pos.id}/breakeven`, { method: 'POST' });
          toast(`${pos.symbol}: SL moved to breakeven ₹${pos.entry_price}`);
          loadPortfolio();
        });
        tr.querySelector('.act-close').addEventListener('click', async () => {
          await api(`/api/paper/positions/${pos.id}/close`, { method: 'POST' });
          toast(`${pos.symbol}: Paper position closed`);
          loadPortfolio();
        });
        posBody.appendChild(tr);
      }
    }

    if (trades) {
      const trBody = document.getElementById('trades-tbody');
      trBody.innerHTML = '';
      for (const t of trades) {
        const tr = document.createElement('tr');
        const dirCls = t.direction === 'LONG' ? 'tag-long' : 'tag-short';
        const pnlCls = t.net_pnl >= 0 ? 'pnl-pos' : 'pnl-neg';
        tr.innerHTML = `
          <td data-value="${t.symbol}"><strong>${t.symbol}</strong></td>
          <td data-value="${t.style}">${t.style}</td>
          <td data-value="${t.direction}"><span class="badge ${dirCls}">${t.direction}</span></td>
          <td data-value="${t.qty}">${t.qty}</td>
          <td data-value="${t.entry_price}">₹${t.entry_price}</td>
          <td data-value="${t.exit_price}">₹${t.exit_price}</td>
          <td data-value="${t.net_pnl}" class="${pnlCls}">${fmtInr(t.net_pnl)}</td>
          <td data-value="${t.r_multiple}" class="${pnlCls}">${t.r_multiple}R</td>
          <td data-value="${t.exit_reason}">${t.exit_reason}</td>
        `;
        trBody.appendChild(tr);
      }
    }
  }

  /* ------------------ Universe ------------------ */
  async function loadUniverse(q = '') {
    const data = await api(`/api/universe?q=${encodeURIComponent(q)}`);
    state.universe = data.items || [];
    if (!q) {
      document.getElementById('tab-universe-count').textContent = String(data.total);
    }
    const tbody = document.getElementById('universe-tbody');
    tbody.innerHTML = '';
    for (const item of state.universe.slice(0, 120)) {
      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td><strong>${item.symbol}</strong></td>
        <td>${escapeHtml(item.name)}</td>
        <td>${item.equity_security_id}</td>
        <td>${item.fno_lot_size}</td>
        <td><button type="button" class="btn btn-ghost btn-sm">Inspect Chart</button></td>
      `;
      tr.querySelector('button').addEventListener('click', () => {
        openChartModal(item.symbol, state.timeframe, null);
      });
      tbody.appendChild(tr);
    }
  }

  /* ------------------ Date-Wise T-1 Watchlist ------------------ */
  let lastCopyAnnouncement = '';
  const copyTimers = new WeakMap();

  function reportCopyFeedback(button, ok, count = 0) {
    const tip = document.getElementById('watchlist-copy-tip');
    if (!button || !tip) return;
    clearTimeout(copyTimers.get(button));
    button.removeAttribute('data-copied');
    button.removeAttribute('data-failed');
    let msg = ok ? `Copied ${count}` : 'Copy failed';
    if (msg === lastCopyAnnouncement) msg += '\u00A0';
    lastCopyAnnouncement = msg;
    tip.textContent = '';
    void button.offsetWidth;
    button.setAttribute(ok ? 'data-copied' : 'data-failed', '');
    requestAnimationFrame(() => {
      tip.textContent = msg;
    });
    copyTimers.set(
      button,
      setTimeout(() => {
        button.removeAttribute('data-copied');
        button.removeAttribute('data-failed');
        tip.textContent = '';
      }, 2200)
    );
  }

  function legacyCopyText(text) {
    const field = Object.assign(document.createElement('textarea'), { value: text });
    field.setAttribute('readonly', '');
    field.style.cssText = 'position:fixed;inset-block-start:0;opacity:0';
    document.body.append(field);
    field.select();
    try {
      return document.execCommand('copy');
    } finally {
      field.remove();
    }
  }

  async function copyWatchlistSymbols(button) {
    const srcEl = document.getElementById('watchlist-copy-source');
    const text = (srcEl?.textContent || '').trim();
    if (!text) {
      reportCopyFeedback(button, false);
      return;
    }
    try {
      await navigator.clipboard.writeText(text);
      reportCopyFeedback(button, true, state.watchlist.length);
    } catch {
      if (legacyCopyText(text)) {
        reportCopyFeedback(button, true, state.watchlist.length);
      } else {
        reportCopyFeedback(button, false);
      }
    }
  }

  async function loadWatchlist(explicitDate = null) {
    const dateSelect = document.getElementById('watchlist-date-select');
    const biasSelect = document.getElementById('watchlist-bias-filter');
    const chosenDate = explicitDate !== null ? explicitDate : (dateSelect?.value || state.watchlistDate || '');
    const bias = biasSelect?.value || 'ALL';
    const qs = new URLSearchParams({ bias });
    if (chosenDate) qs.set('target_date', chosenDate);

    const data = await api(`/api/watchlist?${qs.toString()}`);
    state.watchlist = data.items || [];
    state.watchlistDates = data.dates || [];
    state.watchlistDate = data.target_date || '';

    document.getElementById('tab-watchlist-count').textContent = String(data.total || 0);
    document.getElementById('watchlist-meta-badge').textContent =
      data.target_date
        ? `Target: ${data.target_date} (from T-1 ${data.t1_date || '—'}) · ${data.total} picks`
        : `${data.total} picks`;

    if (dateSelect) {
      dateSelect.innerHTML = '';
      for (const d of state.watchlistDates) {
        const opt = document.createElement('option');
        opt.value = d.target_date;
        opt.textContent = `${d.target_date} (T-1 ${d.t1_date} · #1 ${d.top_symbol || ''})`;
        if (d.target_date === state.watchlistDate) opt.selected = true;
        dateSelect.appendChild(opt);
      }
    }

    const csvLink = document.getElementById('btn-download-watchlist-csv');
    if (csvLink) {
      csvLink.href = state.watchlistDate
        ? `/api/watchlist/csv?target_date=${encodeURIComponent(state.watchlistDate)}`
        : '/api/watchlist/csv';
    }
    const copySrc = document.getElementById('watchlist-copy-source');
    if (copySrc) copySrc.textContent = data.copy_text || '';

    const tbody = document.getElementById('watchlist-tbody');
    tbody.innerHTML = '';
    if (!state.watchlist.length) {
      tbody.innerHTML = `<tr><td colspan="7" class="muted">No watchlist candidates match this filter. Click "Rebuild T-1".</td></tr>`;
      return;
    }

    for (const w of state.watchlist) {
      const tr = document.createElement('tr');
      const rank = Number(w.rank || 999);
      const dirCls = w.bias_hint === 'LONG' ? 'tag-long' : 'tag-short';
      const reasonsStr = (w.reasons || []).join(' | ');
      const tagCls = String(w.tags || '').includes('MONSTER')
        ? 'badge-monster'
        : w.anti_chasing
        ? 'badge-anti'
        : '';
      tr.innerHTML = `
        <td data-value="${rank}"><span class="rank-pill ${rank <= 3 ? 'rank-top' : ''}">#${rank}</span></td>
        <td data-value="${w.symbol}"><strong>${w.symbol}</strong><div class="muted" style="font-size:0.72rem;">PM ${w.pm_rv || 0}x · ATR ${w.atr_pct || 0}%</div></td>
        <td data-value="${w.bias_hint}"><span class="badge ${dirCls}">${w.bias_hint}</span></td>
        <td data-value="${w.score}"><strong>${w.score}</strong></td>
        <td data-value="${escapeHtml(w.trigger_time)}" class="trigger-cell">${escapeHtml(w.trigger_time)}</td>
        <td data-value="${escapeHtml((w.tags || '') + ' ' + reasonsStr)}" class="reason-cell">
          <div style="margin-bottom:0.2rem;"><span class="badge ${tagCls}">${escapeHtml(w.tags || 'WATCH')}</span></div>
          <div>${escapeHtml(reasonsStr)}</div>
        </td>
        <td><button type="button" class="btn btn-ghost btn-sm btn-wl-chart">Inspect</button></td>
      `;
      tr.querySelector('.btn-wl-chart').addEventListener('click', () => {
        const matchSetup = state.setups.find((s) => s.symbol === w.symbol) || null;
        openChartModal(w.symbol, '15m', matchSetup);
      });
      tbody.appendChild(tr);
    }
  }

  async function loadLiveMomentum(forceRefresh = false) {
    const bias = document.getElementById('live-mom-bias-filter')?.value || 'OPTION_READY';
    if (forceRefresh) {
      await api('/api/watchlist/live/refresh', { method: 'POST' });
    }
    const data = await api(`/api/watchlist/live?bias=${encodeURIComponent(bias)}`);
    state.liveMomentum = data.items || [];

    const badge = document.getElementById('live-momentum-badge');
    if (badge) {
      const timeStr = new Date().toLocaleTimeString('en-IN', { hour: '2-digit', minute: '2-digit', second: '2-digit' });
      badge.textContent = `${state.liveMomentum.length} confirmed · ${timeStr} IST`;
    }

    if (data.breadth) {
      renderBreadth(data.breadth, state.setups);
    }

    const tbody = document.getElementById('live-momentum-tbody');
    if (!tbody) return;
    tbody.innerHTML = '';
    if (!state.liveMomentum.length) {
      const emptyMsg = bias === 'OPTION_READY'
        ? 'No low-decay Pullback/Breakout option entries right now. Switch Filter to "All Momentum Movers" to inspect extended or pulling-back stocks.'
        : 'No stocks currently match all live momentum &amp; VWAP conditions for this filter. Click "Refresh Live Momentum" during market hours.';
      tbody.innerHTML = `<tr><td colspan="8" class="muted">${emptyMsg}</td></tr>`;
      return;
    }

    for (const m of state.liveMomentum) {
      const tr = document.createElement('tr');
      const rank = Number(m.rank || 999);
      const dirCls = m.direction === 'LONG' ? 'tag-long' : 'tag-short';
      const pbCls = m.status_tag === 'CREAM_BULL_V2'
        ? 'badge-cream'
        : m.status_tag === 'STRICT_BEAR_V2'
        ? 'badge-bear'
        : '';
      const optCls = m.option_safe
        ? 'badge-opt-ready'
        : m.option_phase === 'WAIT_PULLBACK'
        ? 'badge-opt-warn'
        : 'badge-opt-decay';
      const optPill = m.option_label
        ? `<span class="badge ${optCls}">${escapeHtml(m.option_label)}</span>`
        : '';
      const wlPill = m.in_watchlist
        ? ` <span class="badge badge-wl">T-1 #${m.watchlist_rank || ''}</span>`
        : '';
      const monsterBadge = m.monster_runner
        ? ' <span class="badge badge-monster">MONSTER</span>'
        : '';
      const moveCls = Number(m.move_pct) >= 0 ? 'pnl-pos' : 'pnl-neg';
      const reasonsStr = (m.reasons || []).join(' | ');
      const matchSetup = state.setups.find((s) => s.id === m.setup_id) ||
        state.setups.find((s) => s.symbol === m.symbol && s.direction === m.direction) ||
        null;

      const ob = m.order_block || {};
      const obStatus = ob.status || 'OB ACTIVE';
      const obCls = obStatus === 'OB_BREAKOUT_CONFIRMED'
        ? 'badge-cream'
        : obStatus === 'OB_RETEST_BOUNCE'
        ? 'badge-opt-ready'
        : 'badge-opt-warn';
      const opt = m.option_strike || {};
      const rankPercentile = m.percentile_rank ? m.percentile_rank.split(' ')[0] : '';

      tr.innerHTML = `
        <td data-value="${rank}">
          <span class="rank-pill ${rank <= 3 ? 'rank-top' : ''}">#${rank}</span>
          ${rankPercentile ? `<div class="muted" style="font-size:0.68rem;font-weight:700;margin-top:0.18rem;color:#38bdf8;">${rankPercentile}</div>` : ''}
        </td>
        <td data-value="${m.symbol}">
          <div><strong>${m.symbol}</strong>${wlPill}</div>
          <div style="font-size:0.74rem;font-weight:700;color:${m.direction === 'LONG' ? '#10b981' : '#ef4444'}">${escapeHtml(m.oi_build_up || '')}</div>
          <div class="muted" style="font-size:0.70rem;">Δ ${escapeHtml(m.vol_delta_shares || '—')} (${escapeHtml(m.vol_delta_status || '')})</div>
        </td>
        <td data-value="${m.direction}">
          <span class="badge ${dirCls}">${m.direction}</span>
          <div style="display:flex;flex-wrap:wrap;gap:0.25rem;margin-top:0.25rem;">${optPill}<span class="badge ${pbCls}">${escapeHtml(m.status_tag)}</span>${monsterBadge}</div>
          <div class="muted" style="font-size:0.68rem;margin-top:0.15rem;">Conv: ${m.conviction_score || m.score}/100</div>
        </td>
        <td data-value="${ob.ob_low || 0}">
          <div style="font-size:0.78rem;font-weight:700;">${escapeHtml(ob.ob_zone || '—')}</div>
          <div style="margin-top:0.15rem;"><span class="badge ${obCls}" style="font-size:0.66rem;">${escapeHtml(obStatus)}</span></div>
          <div style="font-size:0.68rem;color:#94a3b8;margin-top:0.2rem;line-height:1.2;">
            ${ob.weekly_ob?.zone && ob.weekly_ob.zone !== 'None' ? `<div><strong style="color:#f59e0b;">W:</strong> ${escapeHtml(ob.weekly_ob.zone)}</div>` : ''}
            ${ob.daily_ob?.zone && ob.daily_ob.zone !== 'None' ? `<div><strong style="color:#38bdf8;">D:</strong> ${escapeHtml(ob.daily_ob.zone)}</div>` : ''}
            ${ob.hourly_ob?.zone && ob.hourly_ob.zone !== 'None' ? `<div><strong style="color:#a855f7;">1H:</strong> ${escapeHtml(ob.hourly_ob.zone)}</div>` : ''}
            ${ob.htf_warning ? `<div style="color:#ef4444;font-weight:700;">⚠️ ${escapeHtml(ob.htf_warning)}</div>` : ''}
          </div>
          <div class="muted" style="font-size:0.68rem;margin-top:0.15rem;">SL ₹${ob.s1 ?? '—'} · R1 ₹${ob.r1 ?? '—'}</div>
        </td>
        <td data-value="${opt.strike || 0}">
          <div style="font-size:0.82rem;font-weight:700;color:#38bdf8;">🎯 ${escapeHtml(opt.option_name || '—')}</div>
          <div style="font-size:0.72rem;color:#cbd5e1;">~₹${opt.est_premium ?? '—'} · 1L (x${opt.lot_size ?? 1}): ₹${Number(opt.est_capital_inr || 0).toLocaleString('en-IN')}</div>
          <div class="muted" style="font-size:0.70rem;">SL ₹${opt.sl_premium ?? '—'} | TP ₹${opt.tp1_premium ?? '—'} (${opt.expected_roi || '+45%'})</div>
        </td>
        <td data-value="${m.move_pct}">
          <span class="${moveCls}" style="font-weight:700;">${m.move_pct >= 0 ? '+' : ''}${m.move_pct}% Open</span>
          <div class="muted" style="font-size:0.70rem;">Day ${m.day_pct >= 0 ? '+' : ''}${m.day_pct}%</div>
          <div class="muted" style="font-size:0.70rem;">VWAP ${m.vwap_diff_pct >= 0 ? '+' : ''}${m.vwap_diff_pct}% (₹${m.vwap})</div>
        </td>
        <td data-value="${escapeHtml(reasonsStr)}" class="reason-cell">
          <div class="trigger-cell" style="margin-bottom:0.15rem;font-size:0.74rem;font-weight:600;">${escapeHtml(m.trigger_time || '')}</div>
          <div class="muted" style="font-size:0.70rem;">${escapeHtml(reasonsStr)}</div>
        </td>
        <td>
          <div class="row-actions">
            <button type="button" class="btn btn-ghost btn-sm btn-lm-chart">Chart</button>
            <button type="button" class="btn btn-ghost btn-sm btn-lm-oc">Option Chain</button>
            ${matchSetup ? '<button type="button" class="btn btn-primary btn-sm btn-lm-paper">Paper</button>' : ''}
          </div>
        </td>
      `;

      tr.querySelector('.btn-lm-chart').addEventListener('click', () => {
        openChartModal(m.symbol, '15m', matchSetup);
      });
      tr.querySelector('.btn-lm-oc')?.addEventListener('click', () => {
        openOptionChainModal(m.symbol);
      });
      tr.querySelector('.btn-lm-paper')?.addEventListener('click', () => {
        if (matchSetup) openOrderModalFromSetup(matchSetup);
      });
      tbody.appendChild(tr);
    }
  }

  /* ------------------ Live Alerts Audit Timeline ------------------ */
  async function loadLiveAlerts() {
    const eventFilter = document.getElementById('alerts-event-filter')?.value || 'ALL';
    const symFilter = (document.getElementById('alerts-symbol-search')?.value || '').trim().toUpperCase();
    const data = await api(`/api/alerts/live?event_type=${encodeURIComponent(eventFilter)}&limit=150`);
    const alerts = data.alerts || [];
    state.liveAlerts = alerts;

    let hasNew = false;
    for (const a of alerts) {
      if (a.alert_id && !state.knownAlertIds.has(a.alert_id)) {
        state.knownAlertIds.add(a.alert_id);
        hasNew = true;
      }
    }
    if (hasNew && state.initialAlertsLoaded) {
      playAlertChime();
    }
    state.initialAlertsLoaded = true;

    const badge = document.getElementById('alerts-count-badge');
    if (badge) badge.textContent = `${alerts.length} alerts`;
    const tabCount = document.getElementById('tab-alerts-count');
    if (tabCount) tabCount.textContent = alerts.length;

    const tbody = document.getElementById('live-alerts-tbody');
    if (!tbody) return;
    tbody.innerHTML = '';

    const filtered = symFilter ? alerts.filter((a) => a.symbol.toUpperCase().includes(symFilter)) : alerts;

    if (!filtered.length) {
      tbody.innerHTML = `<tr><td colspan="8" class="muted">No live alerts logged yet today. Ticks and Order Block triggers are actively monitored.</td></tr>`;
      return;
    }

    for (const a of filtered) {
      const tr = document.createElement('tr');
      const dirCls = a.direction === 'LONG' ? 'tag-long' : 'tag-short';
      const ev = a.event_type || 'TRIGGER';
      const evCls = ev === 'OB_BREAKOUT'
        ? 'badge-breakout'
        : ev === 'TARGET_R1_HIT'
        ? 'badge-target'
        : ev === 'INVALIDATION_EXIT'
        ? 'badge-exit'
        : ev === 'INSTITUTIONAL_SQUEEZE'
        ? 'badge-squeeze'
        : ev === 'OB_RETEST_BOUNCE'
        ? 'badge-retest'
        : 'badge-cream';

      const moveCls = Number(a.move_pct) >= 0 ? 'pnl-pos' : 'pnl-neg';
      const reasonMain = (a.reasons && a.reasons[0]) || '';

      tr.innerHTML = `
        <td data-value="${a.epoch_ts || 0}">
          <span class="time-pill">${escapeHtml(a.timestamp_ist || '—')}</span>
        </td>
        <td data-value="${a.symbol}">
          <div><strong>${a.symbol}</strong> <span class="badge ${dirCls}">${a.direction}</span></div>
          <div class="muted" style="font-size:0.70rem;">₹${Number(a.price).toLocaleString('en-IN', {minimumFractionDigits: 2})}</div>
        </td>
        <td data-value="${ev}">
          <div><span class="badge ${evCls}">${escapeHtml(ev.replaceAll('_', ' '))}</span></div>
          <div class="muted" style="font-size:0.72rem;margin-top:0.2rem;">${escapeHtml(reasonMain)}</div>
        </td>
        <td data-value="${a.move_pct}">
          <span class="${moveCls}" style="font-weight:700;">${a.move_pct >= 0 ? '+' : ''}${a.move_pct}%</span>
          <div class="muted" style="font-size:0.70rem;">VWAP ₹${a.vwap}</div>
        </td>
        <td data-value="${a.ob_zone}">
          <div style="font-size:0.78rem;font-weight:700;">${escapeHtml(a.ob_zone || '—')}</div>
          <div class="muted" style="font-size:0.70rem;">S1 ₹${a.s1} · R1 ₹${a.r1}</div>
        </td>
        <td data-value="${a.option_est_premium}">
          <div style="font-size:0.80rem;font-weight:700;color:#38bdf8;">🎯 ${escapeHtml(a.option_strike_name || '—')}</div>
          <div style="font-size:0.72rem;color:#cbd5e1;">Prem: ~₹${a.option_est_premium} · 1L: ₹${Number(a.option_lot_capital || 0).toLocaleString('en-IN')}</div>
          <div class="muted" style="font-size:0.70rem;">SL ₹${a.option_sl_premium} | TP1 ₹${a.option_tp1_premium}</div>
        </td>
        <td data-value="${escapeHtml(a.oi_build_up)}">
          <div style="font-size:0.76rem;font-weight:700;color:${a.direction === 'LONG' ? '#10b981' : '#ef4444'}">${escapeHtml(a.oi_build_up || '—')}</div>
          <div class="muted" style="font-size:0.70rem;">Delta ${escapeHtml(a.vol_delta_shares || '—')}</div>
        </td>
        <td>
          <div class="row-actions">
            <button type="button" class="btn btn-ghost btn-sm btn-alt-oc">Option Chain</button>
            <button type="button" class="btn btn-primary btn-sm btn-alt-chart">Chart</button>
          </div>
        </td>
      `;

      tr.querySelector('.btn-alt-oc').addEventListener('click', () => {
        openOptionChainModal(a.symbol);
      });
      tr.querySelector('.btn-alt-chart').addEventListener('click', () => {
        const matchSetup = state.setups.find((s) => s.symbol === a.symbol);
        openChartModal(a.symbol, '15m', matchSetup);
      });

      tbody.appendChild(tr);
    }
  }

  /* ------------------ Option Chain & Squeeze Modal ------------------ */
  async function openOptionChainModal(symbol) {
    const ocDlg = document.getElementById('option-chain-dialog');
    if (!ocDlg) return;
    ocDlg.showModal();

    document.getElementById('oc-title').textContent = `Option Chain — ${symbol}`;
    document.getElementById('oc-meta').textContent = 'Loading live strikes, Open Interest and unwinding squeeze...';
    document.getElementById('oc-summary-card').innerHTML = '<p class="muted">Fetching real-time chain...</p>';
    document.getElementById('oc-tbody').innerHTML = '<tr><td colspan="7" class="muted">Loading...</td></tr>';

    try {
      const data = await api(`/api/option-chain/${encodeURIComponent(symbol)}`);
      const spotPx = Number(data.underlying_ltp ?? data.underlying_price ?? 0);
      document.getElementById('oc-title').textContent = `Option Chain — ${data.symbol} (Spot ₹${spotPx.toLocaleString('en-IN', {minimumFractionDigits: 2})})`;
      document.getElementById('oc-meta').textContent = `Lot Size: ${data.lot_size} · ATM: ${data.atm_strike} · Max Call OI: ${data.max_call_oi_strike} · Max Put OI: ${data.max_put_oi_strike}`;

      const sq = data.squeeze_active ? '🔥 YES (CALL SQUEEZE)' : 'NORMAL';
      const optName = data.recommended_option || data.recommended_strike_name || '—';
      const prem = data.recommended_premium ?? '—';
      const sl = data.recommended_sl ?? '—';
      const tp1 = data.recommended_tp1 ?? '—';
      const cap = data.recommended_capital ? ` · 1L: ₹${Number(data.recommended_capital).toLocaleString('en-IN')}` : '';

      document.getElementById('oc-summary-card').innerHTML = `
        <div class="oc-summary-item">
          <span>Underlying Spot</span>
          <strong>₹${spotPx.toLocaleString('en-IN', {minimumFractionDigits: 2})}</strong>
        </div>
        <div class="oc-summary-item">
          <span>ATM Strike</span>
          <strong>${data.atm_strike}</strong>
        </div>
        <div class="oc-summary-item">
          <span>Max Call OI (Wall)</span>
          <strong style="color:#ef4444;">${data.max_call_oi_strike}</strong>
        </div>
        <div class="oc-summary-item">
          <span>Squeeze Status</span>
          <strong style="color:${data.squeeze_active ? '#ec4899' : '#10b981'};">${sq}</strong>
        </div>
        <div class="oc-summary-item" style="grid-column: span 2;">
          <span>Recommended Strike Pick</span>
          <strong style="color:#f59e0b;">🎯 ${escapeHtml(optName)} (~₹${prem}${cap} · SL ₹${sl} · TP ₹${tp1})</strong>
        </div>
      `;

      const tbody = document.getElementById('oc-tbody');
      tbody.innerHTML = '';
      const recStrike = Number(data.recommended_strike || 0);
      const atmStrike = Number(data.atm_strike || 0);

      for (const s of (data.strikes || [])) {
        const tr = document.createElement('tr');
        const stNum = Number(s.strike ?? s.strike_price ?? 0);
        if (stNum === recStrike) {
          tr.classList.add('strike-row-rec');
        } else if (stNum === atmStrike) {
          tr.classList.add('strike-row-atm');
        }

        const ceOi = Number(s.ce_oi ?? s.call_oi ?? 0);
        const ceChg = Number(s.ce_chg_oi ?? s.call_oi_change_pct ?? 0);
        const ceLtp = Number(s.ce_ltp ?? s.call_ltp ?? 0);

        const peOi = Number(s.pe_oi ?? s.put_oi ?? 0);
        const peChg = Number(s.pe_chg_oi ?? s.put_oi_change_pct ?? 0);
        const peLtp = Number(s.pe_ltp ?? s.put_ltp ?? 0);

        const cUnw = ceChg < 0 ? 'unwinding-cell' : ceChg > 0 ? 'buildup-cell' : '';
        const pUnw = peChg < 0 ? 'unwinding-cell' : peChg > 0 ? 'buildup-cell' : '';

        tr.innerHTML = `
          <td>${ceOi.toLocaleString('en-IN')}</td>
          <td class="${cUnw}">${ceChg > 0 ? '+' : ''}${ceChg.toLocaleString('en-IN')}</td>
          <td style="font-weight:700;">₹${ceLtp.toFixed(1)}</td>
          <td class="oc-strike-head" style="font-size:0.88rem;"><strong>${stNum}</strong>${stNum === recStrike ? ' <span class="badge badge-target" style="font-size:0.62rem;">PICK</span>' : ''}${stNum === atmStrike ? ' <span class="badge" style="font-size:0.62rem;">ATM</span>' : ''}</td>
          <td style="font-weight:700;">₹${peLtp.toFixed(1)}</td>
          <td class="${pUnw}">${peChg > 0 ? '+' : ''}${peChg.toLocaleString('en-IN')}</td>
          <td>${peOi.toLocaleString('en-IN')}</td>
        `;
        tbody.appendChild(tr);
      }
    } catch (err) {
      toast(err.message);
      document.getElementById('oc-summary-card').innerHTML = `<p class="pnl-neg">${escapeHtml(err.message)}</p>`;
    }
  }

  /* ------------------ Modals & Actions ------------------ */
  function openOrderModalFromSetup(s) {
    const dlg = document.getElementById('order-dialog');
    document.getElementById('ord-setup-id').value = s.id || '';
    document.getElementById('ord-symbol').value = s.symbol;
    document.getElementById('ord-style').value = s.style;
    document.getElementById('ord-direction').value = s.direction;
    document.getElementById('ord-qty').value = s.suggested_qty;
    document.getElementById('ord-entry').value = s.entry_price;
    document.getElementById('ord-sl').value = s.stop_loss;
    document.getElementById('ord-tp1').value = s.tp1;
    document.getElementById('ord-tp2').value = s.tp2;
    updateOrderPreview();
    dlg.showModal();
  }

  function updateOrderPreview() {
    const entry = Number(document.getElementById('ord-entry').value || 0);
    const sl = Number(document.getElementById('ord-sl').value || 0);
    const tp2 = Number(document.getElementById('ord-tp2').value || 0);
    const qty = Number(document.getElementById('ord-qty').value || 0);
    const risk = Math.abs(entry - sl) * qty;
    const reward = Math.abs(tp2 - entry) * qty;
    const rr = risk > 0 ? (reward / risk).toFixed(2) : '0.00';
    document.getElementById('ord-risk-preview').textContent =
      `Planned Risk: ${fmtInr(risk)} · Potential Reward (TP2): ${fmtInr(reward)} · R:R 1:${rr}`;
  }

  async function refreshHealth() {
    const h = await api('/api/health');
    const pill = document.getElementById('dhan-status-pill');
    const txt = document.getElementById('dhan-status-text');
    const diag = document.getElementById('auth-diag-msg');
    const banner = document.getElementById('token-expired-banner');

    if (banner) {
      const isLive = Boolean(h.auth?.connected);
      banner.hidden = isLive;
      banner.style.display = isLive ? 'none' : 'flex';
    }

    if (h.auth?.connected) {
      pill.classList.add('status-live');
      pill.classList.remove('status-demo');
      txt.textContent = `LIVE DHAN (${h.auth.client_id})`;
      diag.textContent = `Connected to DhanHQ v2 as Client ${h.auth.client_id}.`;
    } else if (h.scanner?.data_source === 'cached') {
      pill.classList.remove('status-live');
      pill.classList.add('status-demo');
      txt.textContent = 'CACHED F&O DATA (Real 216 Stocks)';
      diag.textContent = `Offline / Cached mode active (${h.universe_count || 216} F&O symbols loaded from SQLite cache). Save a fresh 24h Dhan token anytime for live ticks.`;
    } else {
      pill.classList.remove('status-live');
      pill.classList.add('status-demo');
      txt.textContent = 'DEMO DATA (Connect Dhan Token)';
      diag.textContent = `Dhan status: ${h.auth?.error || 'Not connected'}. Running local simulator until valid 24h access token is saved.`;
    }
    if (h.auth?.client_id) {
      if (!document.getElementById('cfg-client-id').value) {
        document.getElementById('cfg-client-id').value = h.auth.client_id;
      }
      if (!document.getElementById('quick-client-id').value) {
        document.getElementById('quick-client-id').value = h.auth.client_id;
      }
    }
    if (h.auth?.app_id && !document.getElementById('cfg-app-id').value) {
      document.getElementById('cfg-app-id').value = h.auth.app_id;
    }
    const hint = document.getElementById('oauth-env-hint');
    if (hint) {
      hint.textContent = h.auth?.has_api_key && h.auth?.has_api_secret
        ? `API Key (${h.auth.app_id}) & Secret loaded from .env. Enter your Dhan Client ID above and click Generate Dhan Login Link.`
        : 'Enter your Dhan App ID and App Secret (or set DHAN_API_KEY & DHAN_API_SECRET in .env).';
    }
  }

  function wireEvents() {
    document.getElementById('btn-scan').addEventListener('click', async () => {
      const btn = document.getElementById('btn-scan');
      btn.disabled = true;
      btn.textContent = 'Scanning...';
      try {
        const res = await api('/api/scan', { method: 'POST', body: JSON.stringify({}) });
        toast(`Scanned ${res.scanned_count} F&O stocks → ${res.setups_count} ranked setups`);
        await Promise.all([loadSetups(), loadWatchlist(), loadLiveMomentum(), loadPortfolio()]);
      } catch (err) {
        toast(err.message);
      } finally {
        btn.disabled = false;
        btn.textContent = 'Run Scan';
      }
    });

    ['filter-playbook', 'filter-style', 'filter-dir', 'filter-grade'].forEach((id) => {
      document.getElementById(id)?.addEventListener('change', loadSetups);
    });

    document.getElementById('btn-jump-rank1')?.addEventListener('click', () => {
      if (state.setups.length > 0) {
        const top = state.setups[0];
        openChartModal(top.symbol, top.style === 'SWING' ? '1h' : '15m', top);
        toast(`Inspecting #${top.rank || 1} Pick: ${top.symbol} (${top.direction})`);
      }
    });

    document.getElementById('watchlist-date-select')?.addEventListener('change', (e) => {
      loadWatchlist(e.target.value);
    });
    document.getElementById('watchlist-bias-filter')?.addEventListener('change', () => {
      loadWatchlist();
    });
    document.getElementById('btn-build-watchlist')?.addEventListener('click', async () => {
      const btn = document.getElementById('btn-build-watchlist');
      btn.disabled = true;
      try {
        const res = await api('/api/watchlist/build', { method: 'POST' });
        toast(`Built ${res.target_date} watchlist (${res.total} ranked picks)`);
        await loadWatchlist(res.target_date || '');
      } catch (err) {
        toast(err.message);
      } finally {
        btn.disabled = false;
      }
    });
    document.getElementById('btn-copy-watchlist')?.addEventListener('click', (e) => {
      copyWatchlistSymbols(e.currentTarget);
    });
    document.getElementById('live-mom-bias-filter')?.addEventListener('change', () => {
      loadLiveMomentum(false);
    });
    document.getElementById('btn-refresh-live-momentum')?.addEventListener('click', async () => {
      const btn = document.getElementById('btn-refresh-live-momentum');
      btn.disabled = true;
      btn.textContent = 'Refreshing...';
      try {
        await loadLiveMomentum(true);
        toast(`Live Momentum Board updated (${state.liveMomentum.length} confirmed)`);
      } catch (err) {
        toast(err.message);
      } finally {
        btn.disabled = false;
        btn.textContent = 'Refresh Live Momentum';
      }
    });

    document.querySelectorAll('.tf-btn').forEach((btn) => {
      btn.addEventListener('click', () => {
        if (state.selectedSymbol) {
          loadChart(state.selectedSymbol, btn.dataset.tf, state.selectedSetup);
        }
      });
    });

    const chartDlg = document.getElementById('chart-dialog');
    document.getElementById('close-chart-btn')?.addEventListener('click', () => {
      chartDlg?.close();
    });

    const chartToolbar = document.querySelector('.chart-toolbar');
    const layerBtns = Array.from(document.querySelectorAll('.layer-btn[data-layer]'));
    layerBtns.forEach((btn) => {
      btn.setAttribute('aria-pressed', String(Boolean(state.chartLayers[btn.dataset.layer])));
      btn.addEventListener('click', () => {
        const key = btn.dataset.layer;
        if (!key) return;
        state.chartLayers[key] = !state.chartLayers[key];
        btn.setAttribute('aria-pressed', String(state.chartLayers[key]));
        renderChartOverlays();
      });
    });

    document.getElementById('btn-chart-fit')?.addEventListener('click', () => {
      resizeChart();
      state.chart?.timeScale().fitContent();
    });

    if (chartToolbar && !('focusGroup' in HTMLElement.prototype)) {
      const tbItems = Array.from(chartToolbar.querySelectorAll('button'));
      tbItems.forEach((b, i) => (b.tabIndex = i ? -1 : 0));
      chartToolbar.addEventListener('focusin', ({ target }) => {
        if (tbItems.includes(target)) {
          tbItems.forEach((b) => (b.tabIndex = b === target ? 0 : -1));
        }
      });
      chartToolbar.addEventListener('keydown', (e) => {
        const idx = tbItems.indexOf(document.activeElement);
        if (idx === -1) return;
        const rtl = getComputedStyle(chartToolbar).direction === 'rtl';
        const steps = { ArrowRight: rtl ? -1 : 1, ArrowLeft: rtl ? 1 : -1 };
        let nextIdx = null;
        if (e.key in steps) {
          nextIdx = (idx + steps[e.key] + tbItems.length) % tbItems.length;
        } else if (e.key === 'Home') {
          nextIdx = 0;
        } else if (e.key === 'End') {
          nextIdx = tbItems.length - 1;
        }
        if (nextIdx !== null) {
          e.preventDefault();
          tbItems[nextIdx].focus();
        }
      });
    }

    // Settings dialog
    const settingsDlg = document.getElementById('settings-dialog');
    document.getElementById('btn-open-settings').addEventListener('click', () => settingsDlg.showModal());
    document.getElementById('close-settings-btn').addEventListener('click', () => settingsDlg.close());

    // Quick Token Popup Dialog & Banner
    const quickTokenDlg = document.getElementById('quick-token-dialog');
    document.getElementById('btn-quick-token-popup')?.addEventListener('click', () => quickTokenDlg?.showModal());
    document.getElementById('dhan-status-pill')?.addEventListener('click', () => quickTokenDlg?.showModal());
    document.getElementById('close-quick-token-btn')?.addEventListener('click', () => quickTokenDlg?.close());
    document.getElementById('cancel-quick-token-btn')?.addEventListener('click', () => quickTokenDlg?.close());

    document.getElementById('btn-save-quick-token')?.addEventListener('click', async () => {
      const btn = document.getElementById('btn-save-quick-token');
      const client_id = document.getElementById('quick-client-id').value.trim();
      const access_token = document.getElementById('quick-access-token').value.trim();
      if (!access_token) {
        toast('Please paste your fresh Dhan 24h access token');
        return;
      }
      btn.disabled = true;
      btn.textContent = 'Activating...';
      try {
        const res = await api('/api/dhan/auth/update-token', {
          method: 'POST',
          body: JSON.stringify({ client_id, access_token }),
        });
        await refreshHealth();
        if (res.auth?.connected) {
          toast('Connected to Live Dhan! Ticks are streaming.');
          quickTokenDlg?.close();
          const b = document.getElementById('token-expired-banner');
          if (b) b.hidden = true;
          document.getElementById('quick-access-token').value = '';
        } else {
          toast(`Token saved! Auth check: ${res.auth?.error || 'Updated'}`);
          quickTokenDlg?.close();
        }
      } catch (err) {
        toast(err.message);
      } finally {
        btn.disabled = false;
        btn.textContent = 'Save & Activate Immediately';
      }
    });

    document.getElementById('btn-save-dhan-token').addEventListener('click', async () => {
      try {
        const client_id = document.getElementById('cfg-client-id').value.trim();
        const access_token = document.getElementById('cfg-access-token').value.trim();
        const res = await api('/api/dhan/auth/update-token', {
          method: 'POST',
          body: JSON.stringify({ client_id, access_token }),
        });
        await refreshHealth();
        if (res.auth?.connected) {
          toast('Connected to DhanHQ v2 Live Data!');
          settingsDlg.close();
        } else {
          toast(`Auth error: ${res.auth?.error || 'Invalid token'}`);
        }
      } catch (err) {
        toast(err.message);
      }
    });

    document.getElementById('btn-gen-consent').addEventListener('click', async () => {
      try {
        const client_id = document.getElementById('cfg-client-id').value.trim();
        const app_id = document.getElementById('cfg-app-id').value.trim();
        const app_secret = document.getElementById('cfg-app-secret').value.trim();
        const res = await api('/api/dhan/auth/generate-consent', {
          method: 'POST',
          body: JSON.stringify({ client_id, app_id, app_secret }),
        });
        const box = document.getElementById('consent-link-box');
        box.hidden = false;
        box.innerHTML = `Open Dhan Login: <a href="${res.loginUrl}" target="_blank" rel="noopener">${res.loginUrl}</a> (if redirect URL is http://127.0.0.1:8000/dhan/callback, it activates automatically; otherwise paste tokenId below)`;
      } catch (err) {
        toast(err.message);
      }
    });

    document.getElementById('btn-consume-consent').addEventListener('click', async () => {
      try {
        const token_id = document.getElementById('cfg-token-id').value.trim();
        const app_id = document.getElementById('cfg-app-id').value.trim();
        const app_secret = document.getElementById('cfg-app-secret').value.trim();
        const res = await api('/api/dhan/auth/consume-consent', {
          method: 'POST',
          body: JSON.stringify({ token_id, app_id, app_secret }),
        });
        await refreshHealth();
        toast(res.auth?.connected ? 'Dhan token activated!' : 'Check Dhan credentials');
      } catch (err) {
        toast(err.message);
      }
    });

    document.getElementById('btn-save-risk').addEventListener('click', async () => {
      try {
        await api('/api/settings/risk', {
          method: 'POST',
          body: JSON.stringify({
            risk_per_trade_inr: Number(document.getElementById('cfg-risk-inr').value),
            min_rr_ratio: Number(document.getElementById('cfg-min-rr').value),
            max_sl_pct_intraday: Number(document.getElementById('cfg-max-sl-intra').value),
            max_sl_pct_swing: Number(document.getElementById('cfg-max-sl-swing').value),
            auto_paper_trade: document.getElementById('cfg-auto-paper').checked,
          }),
        });
        toast('Risk rules updated. Run Scan to apply to setups.');
        await loadPortfolio();
      } catch (err) {
        toast(err.message);
      }
    });

    // Order dialog
    const orderDlg = document.getElementById('order-dialog');
    document.getElementById('close-order-btn').addEventListener('click', () => orderDlg.close());
    document.getElementById('cancel-order-btn').addEventListener('click', () => orderDlg.close());
    document.getElementById('btn-custom-paper').addEventListener('click', () => {
      document.getElementById('ord-setup-id').value = '';
      document.getElementById('ord-symbol').value = state.selectedSymbol || 'RELIANCE';
      document.getElementById('ord-qty').value = 10;
      orderDlg.showModal();
    });
    ['ord-entry', 'ord-sl', 'ord-tp2', 'ord-qty'].forEach((id) => {
      document.getElementById(id).addEventListener('input', updateOrderPreview);
    });

    document.getElementById('order-form').addEventListener('submit', async (e) => {
      e.preventDefault();
      try {
        await api('/api/paper/open', {
          method: 'POST',
          body: JSON.stringify({
            setup_id: document.getElementById('ord-setup-id').value || null,
            symbol: document.getElementById('ord-symbol').value.trim(),
            style: document.getElementById('ord-style').value,
            direction: document.getElementById('ord-direction').value,
            qty: Number(document.getElementById('ord-qty').value),
            entry_price: Number(document.getElementById('ord-entry').value),
            stop_loss: Number(document.getElementById('ord-sl').value),
            tp1: Number(document.getElementById('ord-tp1').value),
            tp2: Number(document.getElementById('ord-tp2').value),
          }),
        });
        orderDlg.close();
        toast('Paper trade opened!');
        await loadPortfolio();
      } catch (err) {
        toast(err.message);
      }
    });

    document.getElementById('btn-refresh-mtm').addEventListener('click', async () => {
      const res = await api('/api/paper/refresh-mtm', { method: 'POST' });
      renderPortfolioData(res.portfolio, res.positions, null);
      if (res.events?.length) {
        toast(`Triggered: ${res.events.map((ev) => `${ev.symbol} ${ev.event}`).join(', ')}`);
        await loadPortfolio();
      } else {
        toast('MTM refreshed');
      }
    });

    document.getElementById('btn-reset-paper').addEventListener('click', async () => {
      await api('/api/paper/reset', { method: 'POST' });
      toast('Paper account reset to initial capital');
      await loadPortfolio();
    });

    document.getElementById('universe-search').addEventListener('input', (e) => {
      loadUniverse(e.target.value);
    });

    document.getElementById('btn-sync-universe').addEventListener('click', async () => {
      const res = await api('/api/universe/sync', { method: 'POST' });
      toast(`Synced ${res.total} NSE F&O stocks`);
      loadUniverse();
    });

    // Live Alerts & Option Chain Controls
    document.getElementById('close-oc-btn')?.addEventListener('click', () => {
      document.getElementById('option-chain-dialog')?.close();
    });

    document.getElementById('btn-toggle-sound')?.addEventListener('click', () => {
      state.soundEnabled = !state.soundEnabled;
      const stateEl = document.getElementById('sound-state');
      const iconEl = document.getElementById('sound-icon');
      if (stateEl) stateEl.textContent = state.soundEnabled ? 'ON' : 'OFF';
      if (iconEl) iconEl.textContent = state.soundEnabled ? '🔔' : '🔕';
      toast(`Audio Alert Chime: ${state.soundEnabled ? 'Enabled' : 'Muted'}`);
      if (state.soundEnabled) playAlertChime();
    });

    document.getElementById('alerts-event-filter')?.addEventListener('change', () => {
      loadLiveAlerts();
    });

    document.getElementById('alerts-symbol-search')?.addEventListener('input', () => {
      loadLiveAlerts();
    });

    document.getElementById('btn-refresh-alerts')?.addEventListener('click', async () => {
      const btn = document.getElementById('btn-refresh-alerts');
      btn.disabled = true;
      btn.textContent = 'Refreshing...';
      try {
        await loadLiveAlerts();
        toast(`Live Alerts updated (${state.liveAlerts.length} logged)`);
      } catch (err) {
        toast(err.message);
      } finally {
        btn.disabled = false;
        btn.textContent = 'Refresh Alerts';
      }
    });

    document.getElementById('btn-clear-alerts')?.addEventListener('click', async () => {
      if (confirm('Clear today\'s live alert history?')) {
        await api('/api/alerts/clear', { method: 'POST' });
        state.knownAlertIds.clear();
        await loadLiveAlerts();
        toast('Alert history cleared');
      }
    });
  }

  async function boot() {
    initTabs();
    initDialogs();
    initSortableTable('ideas-table', 'ideas-sort-status');
    initSortableTable('watchlist-table', 'watchlist-sort-status');
    initSortableTable('live-momentum-table', 'live-momentum-sort-status');
    initSortableTable('live-alerts-table', 'live-alerts-sort-status');
    initSortableTable('trades-table', 'trades-sort-status');
    wireEvents();

    await Promise.allSettled([
      refreshHealth(),
      loadSetups(),
      loadWatchlist(),
      loadLiveMomentum(),
      loadLiveAlerts(),
      loadPortfolio(),
      loadUniverse(),
    ]);

    // Periodic Live Alert & Momentum Poller (every 18 seconds)
    setInterval(async () => {
      try {
        await loadLiveAlerts();
      } catch {
        // silent background check
      }
    }, 18000);

    // Auto refresh MTM when positions exist
    setInterval(async () => {
      if (state.positions.length > 0) {
        try {
          const res = await api('/api/paper/refresh-mtm', { method: 'POST' });
          renderPortfolioData(res.portfolio, res.positions, res.events?.length ? undefined : null);
          if (res.events?.length) {
            toast(`Execution: ${res.events.map((e) => `${e.symbol} ${e.event}`).join(', ')}`);
            await loadPortfolio();
          }
        } catch {
          // silent background tick
        }
      }
    }, 8000);
  }

  document.addEventListener('DOMContentLoaded', boot);
})();
