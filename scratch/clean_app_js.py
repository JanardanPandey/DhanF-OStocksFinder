with open('app/static/app.js', 'r') as f:
    text = f.read()

block_marker = '  /* ------------------ HTF Liquidity Radar (BSL / SSL) ------------------ */'
parts = text.split(block_marker)

base_clean = parts[0]
for p in parts[1:]:
    idx = p.find('  /* ------------------')
    if idx != -1:
        base_clean += p[idx:]

target = '  /* ------------------ Option Chain & Squeeze Modal ------------------ */'

liq_code = '''  /* ------------------ HTF Liquidity Radar (BSL / SSL) ------------------ */
  async function loadLiquidityRadar() {
    const dateSel = document.getElementById('liquidity-date-select')?.value || '2026-11-18';
    const tbody = document.getElementById('liquidity-tbody');
    if (!tbody) return;
    tbody.innerHTML = '<tr><td colspan="8" class="muted">Scanning HTF liquidity pools & point-in-time confirmations...</td></tr>';

    try {
      const data = await api(`/api/liquidity/radar?trade_date=${encodeURIComponent(dateSel)}`);
      const setups = data.setups || [];

      const badge = document.getElementById('liquidity-count-badge');
      if (badge) badge.textContent = `${setups.length} setups`;
      const tabCount = document.getElementById('tab-liquidity-count');
      if (tabCount) tabCount.textContent = setups.length;

      if (!setups.length) {
        tbody.innerHTML = `<tr><td colspan="8" class="muted">No liquidity sweep or retest setups confirmed for ${escapeHtml(dateSel)}.</td></tr>`;
        return;
      }

      tbody.innerHTML = '';
      for (const s of setups) {
        const tr = document.createElement('tr');
        const dirCls = s.direction === 'LONG' ? 'tag-long' : 'tag-short';
        const setupCls = s.setup_name.includes('Sweep') ? 'badge-breakout' : 'badge-retest';

        tr.innerHTML = `
          <td><span class="time-pill">${escapeHtml(s.trigger_time)}</span></td>
          <td>
            <div><strong>${escapeHtml(s.symbol)}</strong> <span class="badge ${dirCls}">${escapeHtml(s.direction)}</span></div>
          </td>
          <td>
            <div style="font-weight:700;">${escapeHtml(s.level_type)}</div>
            <div class="muted" style="font-size:0.72rem;">Level: ₹${Number(s.level_price).toLocaleString('en-IN', {minimumFractionDigits: 2})}</div>
          </td>
          <td>
            <div><span class="badge ${setupCls}">${escapeHtml(s.setup_name)}</span></div>
            <div class="muted" style="font-size:0.72rem;margin-top:0.2rem;">${escapeHtml(s.notes || '')}</div>
          </td>
          <td>
            <div>Entry: <strong>₹${Number(s.entry_price).toLocaleString('en-IN', {minimumFractionDigits: 2})}</strong></div>
            <div class="muted" style="font-size:0.72rem;color:#ef4444;">SL: ₹${Number(s.stop_loss).toLocaleString('en-IN', {minimumFractionDigits: 2})} (Risk: ${s.risk_pts} pts)</div>
          </td>
          <td>
            <div style="color:#10b981;font-weight:700;">₹${Number(s.target_price).toLocaleString('en-IN', {minimumFractionDigits: 2})}</div>
            <div class="muted" style="font-size:0.72rem;">Reward: +${s.reward_pts} pts</div>
          </td>
          <td>
            <div style="font-weight:700;color:#38bdf8;">${escapeHtml(s.rr_ratio)}</div>
            <div class="muted" style="font-size:0.72rem;">RVOL ${s.rvol}x</div>
          </td>
          <td>
            <div class="row-actions">
              <button type="button" class="btn btn-ghost btn-sm btn-liq-oc">Options</button>
              <button type="button" class="btn btn-primary btn-sm btn-liq-chart">Chart</button>
            </div>
          </td>
        `;

        tr.querySelector('.btn-liq-oc').addEventListener('click', () => {
          openOptionChainModal(s.symbol);
        });
        tr.querySelector('.btn-liq-chart').addEventListener('click', () => {
          const matchSetup = state.setups.find((st) => st.symbol === s.symbol);
          openChartModal(s.symbol, '15m', matchSetup);
        });
        tbody.appendChild(tr);
      }
    } catch (err) {
      tbody.innerHTML = `<tr><td colspan="8" class="muted error">Failed to load liquidity radar: ${escapeHtml(err.message)}</td></tr>`;
    }
  }

'''

final_text = base_clean.replace(target, liq_code + target)

listener_target = 'document.getElementById("btn-refresh-live-momentum")?.addEventListener("click", () => {'
listener_addition = '''document.getElementById('btn-refresh-liquidity')?.addEventListener('click', () => {
      loadLiquidityRadar().catch((err) => toast(err.message));
    });
    document.getElementById('liquidity-date-select')?.addEventListener('change', () => {
      loadLiquidityRadar().catch((err) => toast(err.message));
    });
    '''

final_text = final_text.replace(listener_target, listener_addition + listener_target)

# Also load on boot
boot_target = 'loadLiveAlerts().catch(() => {});'
boot_addition = '''loadLiveAlerts().catch(() => {});
    loadLiquidityRadar().catch(() => {});'''
final_text = final_text.replace(boot_target, boot_addition)

with open('app/static/app.js', 'w') as f:
    f.write(final_text)

print('app.js successfully updated!')
