// DOM side of the simulator: log, toasts, telemetry, capture card, minimap, labels, summary.
const $ = (id) => document.getElementById(id);

export const PRETTY = {
  early_blight: 'Early blight', healthy: 'Healthy', late_blight: 'Late blight',
  septoria_leaf_spot: 'Septoria leaf spot',
};
const TREAT = { protectant_fungicide: 'protectant', late_blight_fungicide: 'late-blight fungicide' };
const ACTION_TEXT = { spray: 'SPRAY', no_spray: 'NO SPRAY', monitor: 'MONITOR', inspect: 'INSPECT', hold: 'HOLD' };
export const STATUS_CSS = {
  pending: '#cfd8dc', healthy: '#4caf50', monitor: '#ffd54f', spray: '#ff8a3d', lb: '#e5484d',
  inspect: '#b388ff', done: '#29b6f6', gap: '#90a4ae',
};

export const fmtTime = (s) => {
  s = Math.max(0, Math.round(s));
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`;
};

export function statusFor(decision) {
  if (!decision) return 'inspect';
  switch (decision.action) {
    case 'spray': return decision.treatment === 'late_blight_fungicide' ? 'lb' : 'spray';
    case 'no_spray': return 'healthy';
    case 'monitor': return 'monitor';
    default: return 'inspect';
  }
}

export class UI {
  constructor() {
    this.logEl = $('log');
    this.toastsEl = $('toasts');
    this.labelsEl = $('labels');
    this.labels = new Map();
    this.captures = [];
    this.showGradcam = false;
    $('capToggle').addEventListener('click', () => {
      this.showGradcam = !this.showGradcam;
      if (this.current) this.showCapture(this.current, false);
    });
  }

  reset() {
    this.logEl.innerHTML = '';
    for (const el of this.labels.values()) el.el.remove();
    this.labels.clear();
    this.captures = [];
    $('strip').innerHTML = '';
    $('capBody').hidden = true; $('capEmpty').hidden = false; $('capIdx').textContent = '';
    this.current = null;
    this.hideStill();
    this.setPhase('preflight');
  }

  log(t, text, cls = '') {
    const row = document.createElement('div');
    row.innerHTML = `<time>${fmtTime(t)}</time><span class="${cls}"></span>`;
    row.lastChild.textContent = text;
    this.logEl.appendChild(row);
    while (this.logEl.children.length > 400) this.logEl.firstChild.remove();
    this.logEl.scrollTop = this.logEl.scrollHeight;
  }

  toast(text, sub = '', cls = '', ms = 3200) {
    const el = document.createElement('div');
    el.className = `toast ${cls}`;
    el.innerHTML = '<div></div>' + (sub ? '<div class="t2"></div>' : '');
    el.firstChild.textContent = text;
    if (sub) el.lastChild.textContent = sub;
    this.toastsEl.appendChild(el);
    while (this.toastsEl.children.length > 3) this.toastsEl.firstChild.remove();
    setTimeout(() => el.remove(), ms);
  }

  setPhase(phase) {
    const order = ['markers', 'map', 'scan', 'plan', 'spray', 'return'];
    const idx = order.indexOf(phase);
    document.querySelectorAll('#phases li').forEach((li) => {
      const i = order.indexOf(li.dataset.phase);
      li.classList.toggle('active', i === idx);
      li.classList.toggle('done', phase === 'done' || (idx >= 0 && i < idx));
    });
    const cap = {
      markers: ['Step 1 of 6', 'The drone flies to each corner the operator marked and drops an ArUco marker. The markers will define the field boundary.'],
      map: ['Step 2 of 6', 'From 15 m one photo covers the whole field: the 4 markers fix the boundary and size, and every plant becomes a data point in metres.'],
      scan: ['Step 3 of 6', 'Flying low, row by row, the drone photographs each plant. Every photo goes to the disease model, then severity and the dosage rules.'],
      plan: ['Step 4 of 6', 'Healthy and mild plants are skipped. The shortest route over the plants that need spray is computed and split into tank loads.'],
      spray: ['Step 5 of 6', 'The drone flies the route and gives each target its own dose. Press M to take over; the route is re-planned when you hand back.'],
      return: ['Step 6 of 6', 'Returning to the home pad to land.'],
    }[phase];
    const capEl = $('caption');
    if (cap) { $('capStep').textContent = cap[0]; $('capText').textContent = cap[1]; }
    capEl.hidden = !cap;
    const names = { preflight: 'Pre-flight', markers: 'Placing markers', map: 'Mapping field', scan: 'Scanning plants',
      plan: 'Planning route', spray: 'Spraying', return: 'Returning home', done: 'Complete' };
    $('tPhase').textContent = names[phase] ?? phase;
  }

  setMode(mode, paused) {
    const el = $('mode');
    const label = paused ? 'PAUSED' : ({ AUTO: 'AUTO', MANUAL: 'MANUAL', RTH: 'RETURN HOME' })[mode];
    el.textContent = label;
    el.className = 'mode ' + (paused ? 'paused' : mode === 'MANUAL' ? 'manual' : mode === 'RTH' ? 'rth' : 'auto');
    $('banner').hidden = mode !== 'MANUAL';
    $('caption').classList.toggle('hidden', mode === 'MANUAL');
  }

  telemetry(d, m) {
    $('tTime').textContent = fmtTime(m.flightTime);
    $('tAlt').textContent = `${d.pos.z.toFixed(1)} m`;
    $('tSpd').textContent = `${d.speed().toFixed(1)} m/s`;
    $('tPos').textContent = `${d.pos.x.toFixed(1)}, ${d.pos.y.toFixed(1)}`;
    $('tDist').textContent = `${d.odometer.toFixed(0)} m`;
    const b = d.batteryPct();
    const bar = $('mBatt');
    bar.style.width = `${b}%`; bar.classList.toggle('low', b < 25);
    $('tBatt').textContent = `${b.toFixed(0)}%`;
    $('mTank').style.width = `${Math.min(100, 100 * d.tankMl / d.cfg.tank_capacity_ml)}%`;
    $('tTank').textContent = `${d.tankMl.toFixed(0)} mL`;
    $('tMix').textContent = d.tankMl > 0.5 ? `${TREAT[d.tankTreatment] ?? 'mix'} mix` : 'empty';
    $('tMarkers').textContent = `${m.markersPlaced} / 4`;
    $('tFound').textContent = m.plantsFound == null ? '—' : String(m.plantsFound);
    $('tShot').textContent = m.plantsFound == null ? '—' : `${m.photos} / ${m.scanTotal}`;
    $('tSprayed').textContent = m.targets == null ? '—' : `${m.targetsDone} / ${m.targets}`;
  }

  showStill(src, title) {
    const img = $('camStill');
    img.src = src; img.hidden = false;
    $('camTitle').textContent = title;
  }

  hideStill() {
    $('camStill').hidden = true;
    $('camTitle').textContent = 'Drone camera · live';
  }

  showCapture(res, addToStrip = true) {
    this.current = res;
    $('capEmpty').hidden = true; $('capBody').hidden = false;
    const hasHeat = !!res.gradcam;
    $('capToggle').hidden = !hasHeat;
    $('capToggle').textContent = this.showGradcam ? 'Photo' : 'Grad-CAM';
    $('capImg').src = (this.showGradcam && hasHeat) ? res.gradcam : res.photo;
    const label = $('capLabel'), action = $('capAction');
    if (!res.found) {
      label.textContent = res.kind === 'gap' ? 'Gap in the row' : 'Plant not visible';
      $('capConf').textContent = '—'; $('capConfBar').style.width = '0%';
      $('capProbs').innerHTML = ''; $('capSev').textContent = '—';
      action.className = 'pill ' + (res.kind === 'gap' ? 'gap' : 'inspect');
      action.textContent = res.kind === 'gap' ? 'EMPTY' : 'INSPECT';
      $('capDose').textContent = '';
      $('capReason').textContent = res.message || '';
    } else {
      label.textContent = `${PRETTY[res.label] ?? res.label}${res.recovered ? ' · found on low pass' : ''}`;
      label.style.color = res.label === 'healthy' ? '#8fe39a' : '#ffb38a';
      const c = Math.round(res.confidence * 100);
      $('capConf').textContent = `${c}%`;
      const bar = $('capConfBar');
      bar.style.width = `${c}%`;
      bar.style.background = res.uncertain ? '#b388ff' : '#7ee08a';
      $('capProbs').innerHTML = Object.entries(res.probabilities || {})
        .map(([k, v]) => `<div><span>${(PRETTY[k] ?? k).replace(' leaf spot', '')}</span><div class="bar"><i style="width:${(v * 100).toFixed(0)}%"></i></div><b>${(v * 100).toFixed(0)}%</b></div>`)
        .join('');
      $('capSev').textContent = res.label === 'healthy' ? 'none' :
        `${res.severity_category} (${(res.severity_score * 100).toFixed(0)}% leaf area)${res.severity_valid ? '' : ' · unreliable'}`;
      const d = res.decision;
      action.className = 'pill ' + (d.action === 'spray' && d.treatment === 'late_blight_fungicide' ? 'lb' : d.action);
      action.textContent = ACTION_TEXT[d.action] ?? d.action;
      $('capDose').textContent = d.action === 'spray' ? `${d.dosage_ml} mL · ${TREAT[d.treatment] ?? d.treatment}` : '';
      $('capReason').textContent = d.reason;
    }
    $('capIdx').textContent = res.plant_id ? `plant #${res.plant_id}` : '';
    if (addToStrip) {
      this.captures.push(res);
      const img = document.createElement('img');
      img.src = res.photo;
      img.style.borderColor = STATUS_CSS[res.found ? statusFor(res.decision) : (res.kind === 'gap' ? 'gap' : 'inspect')];
      img.title = res.found ? `#${res.plant_id} ${PRETTY[res.label]}` : 'no plant';
      img.addEventListener('click', () => this.showCapture(res, false));
      const strip = $('strip');
      strip.prepend(img);
      while (strip.children.length > 9) strip.lastChild.remove();
    }
  }

  // ── 3D labels ──────────────────────────────────────────────
  setLabel(key, text, cls, pos) {
    let entry = this.labels.get(key);
    if (!entry) {
      const el = document.createElement('div');
      this.labelsEl.appendChild(el);
      entry = { el };
      this.labels.set(key, entry);
    }
    entry.el.className = `lab ${cls}`;
    entry.el.textContent = text;
    entry.pos = pos;
  }

  removeLabel(key) {
    const e = this.labels.get(key);
    if (e) { e.el.remove(); this.labels.delete(key); }
  }

  updateLabels(project, enabled, exclude) {
    this.labelsEl.style.display = enabled ? '' : 'none';
    if (!enabled) return;
    for (const e of this.labels.values()) {
      const p = project(e.pos);
      const hidden = exclude && p.x > exclude.left && p.x < exclude.right && p.y > exclude.top && p.y < exclude.bottom + 20;
      if (!p.visible || hidden) { e.el.style.display = 'none'; continue; }
      e.el.style.display = '';
      e.el.style.left = `${p.x}px`;
      e.el.style.top = `${p.y}px`;
    }
  }

  // ── minimap ────────────────────────────────────────────────
  drawMinimap(world, mission, drone) {
    const cv = $('minimap');
    const ctx = cv.getContext('2d');
    const W = cv.width, H = cv.height;
    ctx.clearRect(0, 0, W, H);
    if (!world) return;
    const xmin = Math.min(-3, world.home.x_m - 1.5), xmax = world.field_width_m + 1;
    const ymin = Math.min(-2.5, world.home.y_m - 1.5), ymax = world.field_height_m + 1;
    const s = Math.min((W - 16) / (xmax - xmin), (H - 16) / (ymax - ymin));
    const ox = (W - s * (xmax - xmin)) / 2, oy = (H - s * (ymax - ymin)) / 2;
    const X = (x) => ox + (x - xmin) * s, Y = (y) => H - oy - (y - ymin) * s;
    ctx.fillStyle = 'rgba(122,90,60,.55)';
    ctx.fillRect(X(-0.6), Y(world.field_height_m + 0.6), (world.field_width_m + 1.2) * s, (world.field_height_m + 1.2) * s);
    // planned corners / measured boundary
    const poly = (pts, color, dash) => {
      ctx.setLineDash(dash || []); ctx.strokeStyle = color; ctx.lineWidth = 1.5;
      ctx.beginPath(); pts.forEach(([x, y], i) => (i ? ctx.lineTo(X(x), Y(y)) : ctx.moveTo(X(x), Y(y))));
      ctx.closePath(); ctx.stroke(); ctx.setLineDash([]);
    };
    poly(world.planned_corners, mission.boundary ? 'rgba(255,213,79,.35)' : '#ffd54f', [4, 3]);
    if (mission.boundary) poly(mission.boundary, '#5cc8ff');
    // route
    if (mission.routeSorties) {
      mission.routeSorties.forEach((so, i) => {
        ctx.strokeStyle = so.treatment === 'late_blight_fungicide' ? '#e5484d' : '#ff8a3d';
        ctx.lineWidth = 1.6;
        ctx.beginPath();
        const start = (i === 0 && mission.routeStart) ? mission.routeStart : [world.home.x_m, world.home.y_m];
        ctx.moveTo(X(start[0]), Y(start[1]));
        so.stops.forEach((st) => ctx.lineTo(X(st.x_m), Y(st.y_m)));
        ctx.lineTo(X(world.home.x_m), Y(world.home.y_m));
        ctx.stroke();
      });
    }
    // data points
    for (const p of Object.values(mission.points)) {
      ctx.fillStyle = STATUS_CSS[p.status] ?? STATUS_CSS.pending;
      ctx.beginPath(); ctx.arc(X(p.x_m), Y(p.y_m), Math.max(2.2, s * 0.17), 0, Math.PI * 2); ctx.fill();
    }
    // markers
    for (const mk of mission.markers) {
      ctx.fillStyle = '#f5f5f5';
      ctx.fillRect(X(mk.x_m) - 3, Y(mk.y_m) - 3, 6, 6);
    }
    // home
    ctx.fillStyle = '#e7efe9'; ctx.font = 'bold 11px sans-serif'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    ctx.fillText('H', X(world.home.x_m), Y(world.home.y_m));
    // drone
    const f = drone.forward();
    const px = X(drone.pos.x), py = Y(drone.pos.y);
    const ang = Math.atan2(-f.y, f.x);
    ctx.save(); ctx.translate(px, py); ctx.rotate(ang);
    ctx.fillStyle = mission.mode === 'MANUAL' ? '#ffb74d' : '#5cc8ff';
    ctx.beginPath(); ctx.moveTo(8, 0); ctx.lineTo(-5, 5); ctx.lineTo(-2, 0); ctx.lineTo(-5, -5); ctx.closePath(); ctx.fill();
    ctx.restore();
  }

  // ── summary ────────────────────────────────────────────────
  showSummary(sum, local, simId, waypointFiles) {
    const pct = (v) => (v == null ? '—' : `${v}%`);
    const ch = sum.chemical, rt = sum.route, cl = sum.classification, mp = sum.mapping || {};
    const man = sum.manual || {};
    const bar = (v, max, cls = '') => `<div class="bar"><i class="${cls}" style="width:${Math.max(1, 100 * v / max)}%"></i></div>`;
    const chemMax = Math.max(ch.blanket_ml, ch.used_ml, 1);
    const distMax = Math.max(rt.lawnmower_distance_m || 1, rt.spray_phase_distance_m || 0);
    const acts = sum.actions || {};
    const files = (waypointFiles || []).map((f, i) =>
      `<a href="/drone/mission/${simId}/${i + 1}.waypoints" download="${f}">Sortie ${i + 1} · QGroundControl .waypoints</a>`).join('');
    document.getElementById('sumBody').innerHTML = `
      <p class="lead">${sum.field.plants} plants on a ${sum.field.width_m.toFixed(2)} m × ${sum.field.height_m.toFixed(2)} m field.
      ${sum.spray_targets} needed spray; ${(acts.no_spray || 0) + (acts.monitor || 0)} were left alone.</p>
      <div class="kpis">
        <div class="kpi"><span>Chemical saved</span><b class="good">${pct(ch.saved_pct)}</b><small>${ch.used_ml} mL vs ${ch.blanket_ml} mL blanket</small></div>
        <div class="kpi"><span>Spray route vs blanket pass</span><b class="good">${pct(rt.saved_vs_lawnmower_pct)}</b><small>${rt.spray_phase_distance_m} m vs ${rt.lawnmower_distance_m} m</small></div>
        <div class="kpi"><span>Flight time</span><b>${fmtTime(local.flightTime)}</b><small>${local.distance.toFixed(0)} m flown in total</small></div>
        <div class="kpi"><span>Battery used</span><b>${local.battery.toFixed(0)}%</b><small>${local.energy.toFixed(1)} Wh${local.swaps ? ` · ${local.swaps} swap` : ''}</small></div>
      </div>
      <div class="cmp">
        <div class="line"><span>Chemical · this mission</span>${bar(ch.used_ml, chemMax)}<b>${ch.used_ml} mL</b></div>
        <div class="line"><span>Chemical · blanket spray</span>${bar(ch.blanket_ml, chemMax, 'base')}<b>${ch.blanket_ml} mL</b></div>
        <div class="line"><span>Spray-phase distance</span>${bar(rt.spray_phase_distance_m || 0, distMax)}<b>${rt.spray_phase_distance_m} m</b></div>
        <div class="line"><span>Blanket lawn-mower pass</span>${bar(rt.lawnmower_distance_m || 0, distMax, 'base')}<b>${rt.lawnmower_distance_m} m</b></div>
      </div>
      <div class="sum-cols">
        <div>
          <h3>What the drone did</h3>
          <table>
            <tr><td>Markers placed / read</td><td>${local.markersPlaced} / ${local.markersFound}</td></tr>
            <tr><td>Plants located (overview + low pass)</td><td>${sum.plants_mapped}${sum.recovered_on_low_pass ? ` (${sum.recovered_on_low_pass} on low pass)` : ''}</td></tr>
            <tr><td>Plants photographed and classified</td><td>${sum.plants_photographed}</td></tr>
            <tr><td>Spray / no spray / monitor / inspect</td><td>${acts.spray || 0} / ${acts.no_spray || 0} / ${acts.monitor || 0} / ${acts.inspect || 0}</td></tr>
            <tr><td>Targets fully sprayed</td><td>${sum.targets_fully_sprayed} / ${sum.spray_targets}</td></tr>
            <tr><td>Cost of chemical (₹, indicative)</td><td>${ch.cost_inr} vs ${ch.blanket_cost_inr}</td></tr>
            <tr><td>Route re-plans</td><td>${sum.replans}</td></tr>
          </table>
          <h3>Pilot (manual control)</h3>
          <table>
            <tr><td>Times the pilot took over</td><td>${man.overrides || 0}</td></tr>
            <tr><td>Time under manual control</td><td>${fmtTime(man.time_s || 0)}</td></tr>
            <tr><td>Sprayed by hand</td><td>${(man.spray_ml || 0).toFixed(1)} mL${man.nontarget_ml ? ` (${man.nontarget_ml.toFixed(1)} mL off-target)` : ''}</td></tr>
          </table>
        </div>
        <div>
          <h3>Check against ground truth (synthetic field)</h3>
          <table>
            <tr><td>Plants found by mapping</td><td>${mp.matched ?? '—'} / ${mp.true_plants ?? '—'}</td></tr>
            <tr><td>Mean position error</td><td>${mp.mean_position_error_m != null ? (mp.mean_position_error_m * 100).toFixed(1) + ' cm' : '—'}</td></tr>
            <tr><td>Disease correctly identified</td><td>${cl.correct} / ${cl.total} (${cl.accuracy != null ? (cl.accuracy * 100).toFixed(1) : '—'}%)</td></tr>
            <tr><td>Diseased plants acted on</td><td>${sum.diseased_acted_on} / ${sum.diseased_plants}</td></tr>
            <tr><td>Healthy plants sprayed</td><td>${sum.healthy_plants_sprayed}</td></tr>
          </table>
          <p class="note">${sum.note}</p>
        </div>
      </div>
      <div class="downloads">
        ${files}
        <a href="/sim/${simId}/files/flight_log.json" target="_blank">Flight log (JSON)</a>
        <a href="/sim/${simId}/files/sim_summary.json" target="_blank">Mission summary (JSON)</a>
        <a href="/decisions/${simId}" target="_blank">All plant decisions (API)</a>
      </div>`;
    document.getElementById('summary').hidden = false;
  }
}
