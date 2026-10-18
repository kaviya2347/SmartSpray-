// Mission executor: the drone's autopilot, the six mission phases, and the
// pilot's keyboard override. The backend is the drone's camera + onboard AI.
import * as THREE from '../vendor/three.bundle.min.js';
import { api } from './api.js';
import { toV } from './world3d.js';
import { PRETTY, statusFor } from './ui.js';

const MARKER_ALT = 3.0;        // transit altitude while placing markers
const DROP_ALT = 1.2;          // release height for a marker
const MAP_ALT = 15.0;          // overview photo altitude
const SCAN_ALT = 2.5;          // low-altitude photo of each plant
const SCAN_SPEED = 2.0;
const TRANSIT_ALT = 2.5;
const SPRAY_ALT = 1.6;         // hover height above the canopy while spraying
const GIMBAL_HALF_ANGLE = 0.31;   // ~35 degree camera footprint (1.6 m at 2.5 m)
const PHASES = ['markers', 'map', 'scan', 'plan', 'spray', 'return'];

const wrapAngle = (a) => Math.atan2(Math.sin(a), Math.cos(a));

export class Mission {
  constructor({ world3d, drone, particles, ui, session, options }) {
    Object.assign(this, { world3d, drone, particles, ui, session, options });
    this.world = session.world;
    this.cfg = session.mission_config;
    this.simId = session.sim_id;
    this.mode = 'AUTO';
    this.paused = false;
    this.phase = 'preflight';
    this.simTime = 0;
    this.flightTime = 0;
    this.goal = null;
    this.rth = null;
    this.keys = {};
    this.markers = [];
    this.markersPlaced = 0;
    this.markersFound = 0;
    this.points = {};               // plant_id -> {x_m, y_m, status, decision}
    this.plantsFound = null;
    this.scanTotal = 0;
    this.photos = 0;
    this.targets = null;
    this.targetsDone = 0;
    this.sprayedMl = {};
    this.events = [];
    this.track = [];
    this.phaseStats = {};
    this.manual = { overrides: 0, time_s: 0, spray_ml: 0, nontarget_ml: 0, soil_ml: 0 };
    this.sprayJob = null;
    this.replanRequested = false;
    this.queue = [];
    this.active = null;
    this.routeSorties = null;
    this.routeStart = null;
    this.boundary = null;
    this.warp = 1;
    this.swaps = 0;
    this.lastTrack = -1;
    this.lastFence = -10;
    this.burst = null;
    this.inSortie = false;
    this.sortieDist = 0;
    this.finished = false;
    this.started = false;
  }

  // ── logging ───────────────────────────────────────────────────────
  log(text, cls = 'info') {
    this.events.push({ t: +this.simTime.toFixed(2), phase: this.phase, mode: this.mode, text });
    this.ui.log(this.simTime, text, cls);
  }

  setPhase(phase) {
    const now = { t: this.simTime, d: this.drone.odometer };
    if (this._phaseMark) {
      const prev = this.phaseStats[this._phaseMark.phase] || { time_s: 0, distance_m: 0 };
      prev.time_s += now.t - this._phaseMark.t;
      prev.distance_m += now.d - this._phaseMark.d;
      this.phaseStats[this._phaseMark.phase] = prev;
    }
    this._phaseMark = { phase, ...now };
    this.phase = phase;
    this.ui.setPhase(phase);
    this.onPhase?.(phase);
  }

  // ── autopilot primitives (promises resolved by update()) ──────────
  goto(x, y, alt, speed = this.cfg.cruise_speed_mps, tol = 0.08) {
    return new Promise((resolve) => { this.goal = { type: 'goto', x, y, alt, speed, tol, resolve }; });
  }

  wait(seconds) {
    return new Promise((resolve) => { this.goal = { type: 'wait', until: this.simTime + seconds, resolve }; });
  }

  async takeoff(alt) {
    this.drone.motors = true;
    if (this.drone.pos.z < 0.2) this.log('Motors armed · take-off', 'ok');
    return this.goto(this.drone.pos.x, this.drone.pos.y, alt, 1.0, 0.1);
  }

  async land() {
    const r = await this.goto(this.drone.pos.x, this.drone.pos.y, 0, 0.6, 0.05);
    if (r !== 'aborted') { this.drone.motors = false; this.drone.vel.set(0, 0, 0); this.drone.pos.z = 0; }
    return r;
  }

  // True right after a goal or spray job finished: the main loop then lets the
  // mission's async code issue its next command before simulating further.
  takeYield() { const y = this._yield; this._yield = false; return y; }

  abortGoal() {
    if (this.goal) { const g = this.goal; this.goal = null; g.resolve('aborted'); }
    if (this.sprayJob) { const j = this.sprayJob; this.sprayJob = null; j.resolve('aborted'); }
  }

  _goalCommand(g, dt) {
    const d = this.drone;
    if (g.type === 'wait') {
      if (this.simTime >= g.until) { this.goal = null; this._yield = true; g.resolve('ok'); }
      return { x: 0, y: 0, z: 0, yaw: 0 };
    }
    if (!d.motors && g.alt > 0.05) d.motors = true;
    const dx = g.x - d.pos.x, dy = g.y - d.pos.y, dz = g.alt - d.pos.z;
    const dxy = Math.hypot(dx, dy);
    let vx = 0, vy = 0;
    if (dxy > 1e-4) {
      // climb before long horizontal legs when far below the target altitude
      const climbing = dz > 1.5 && d.pos.z < 0.8;
      const v = climbing ? 0 : Math.min(g.speed, Math.sqrt(2 * 2.4 * dxy) * 0.92, dxy * 4 + 0.05);
      vx = dx / dxy * v; vy = dy / dxy * v;
    }
    const vz = THREE.MathUtils.clamp(dz * 1.6, -1.4, 1.4);
    let yaw = 0;
    if (dxy > 0.7) yaw = THREE.MathUtils.clamp(wrapAngle(Math.atan2(dy, dx) - d.heading) * 2.2, -1.5, 1.5);
    if (dxy < g.tol && Math.abs(dz) < 0.06 && d.speed() < 0.25 && Math.abs(d.vel.z) < 0.25) {
      this.goal = null; this._yield = true; g.resolve('ok');
    }
    return { x: vx, y: vy, z: vz, yaw };
  }

  _manualCommand() {
    const k = this.keys, d = this.drone;
    const fast = k.shift ? 4.5 : 2.2;
    const fwd = (k.w || k.arrowup ? 1 : 0) - (k.s || k.arrowdown ? 1 : 0);
    const right = (k.d || k.arrowright ? 1 : 0) - (k.a || k.arrowleft ? 1 : 0);
    const up = (k.r || k.pageup ? 1 : 0) - (k.f || k.pagedown ? 1 : 0);
    const yaw = (k.q ? 1 : 0) - (k.e ? 1 : 0);
    if (up > 0 && !d.motors) { d.motors = true; this.log('Pilot armed motors', 'pilot'); }
    if (up < 0 && d.pos.z < 0.03 && d.motors) { d.motors = false; this.log('Pilot landed', 'pilot'); }
    const f = d.forward(), r = { x: f.y, y: -f.x };
    let vx = f.x * fwd + r.x * right, vy = f.y * fwd + r.y * right;
    const n = Math.hypot(vx, vy);
    if (n > 1) { vx /= n; vy /= n; }
    const cmd = { x: vx * fast, y: vy * fast, z: up * 1.3, yaw: yaw * 1.4 };
    if (d.pos.z > 25 && cmd.z > 0) cmd.z = 0;
    this._geofence(cmd);
    return cmd;
  }

  _fence() {
    const pts = (this.boundary ?? this.world.planned_corners).concat([[this.world.home.x_m, this.world.home.y_m]]);
    const xs = pts.map((p) => p[0]), ys = pts.map((p) => p[1]);
    return { x0: Math.min(...xs) - 3, x1: Math.max(...xs) + 3, y0: Math.min(...ys) - 3, y1: Math.max(...ys) + 3 };
  }

  _geofence(cmd) {
    const f = this._fence(), p = this.drone.pos;
    let hit = false;
    if ((p.x <= f.x0 && cmd.x < 0) || (p.x >= f.x1 && cmd.x > 0)) { cmd.x = 0; hit = true; }
    if ((p.y <= f.y0 && cmd.y < 0) || (p.y >= f.y1 && cmd.y > 0)) { cmd.y = 0; hit = true; }
    if (hit && this.simTime - this.lastFence > 4) {
      this.lastFence = this.simTime;
      this.ui.toast('Geofence', 'The drone stays within 3 m of the marked field.', 'warn', 2200);
      this.log('Geofence: manual input outside the field boundary was blocked', 'warn');
    }
  }

  // ── mode switching ────────────────────────────────────────────────
  toggleManual() {
    if (!this.started || this.finished) return;
    if (this.mode === 'MANUAL' || this.mode === 'RTH') this.resumeAuto();
    else {
      this.mode = 'MANUAL';
      this.rth = null;
      this.manual.overrides++;
      this.manualStart = this.simTime;
      this.log('PILOT OVERRIDE — manual control, autopilot paused', 'pilot');
      this.ui.toast('Manual control', 'Fly with W A S D · Q E turn · R F height · Space to spray', 'warn');
    }
    this.ui.setMode(this.mode, this.paused);
  }

  resumeAuto() {
    if (this.mode === 'MANUAL') {
      this.manual.time_s += this.simTime - (this.manualStart ?? this.simTime);
      this._endBurst();
    }
    this.mode = 'AUTO';
    this.rth = null;
    this.log('Autopilot resumed', 'ok');
    if (this.phase === 'spray') {
      this.replanRequested = true;
      this.abortGoal();
    }
    this.ui.setMode(this.mode, this.paused);
  }

  returnHome(reason = 'Pilot command') {
    if (!this.started || this.finished) return;
    if (this.mode === 'MANUAL') { this.manual.time_s += this.simTime - (this.manualStart ?? this.simTime); this._endBurst(); }
    this.mode = 'RTH';
    const h = this.world.home;
    this.rth = { stage: 0, alt: Math.max(TRANSIT_ALT, this.drone.pos.z), x: h.x_m, y: h.y_m };
    this.log(`Return to home: ${reason}`, 'warn');
    this.ui.setMode(this.mode, this.paused);
  }

  _rthCommand(dt) {
    const r = this.rth, d = this.drone;
    if (r.stage === 0) {                                  // climb / transit to home
      const fake = { type: 'goto', x: r.x, y: r.y, alt: r.alt, speed: this.cfg.cruise_speed_mps, tol: 0.1, resolve: () => { r.stage = 1; } };
      const saved = this.goal; this.goal = fake;
      const c = this._goalCommand(fake, dt);
      this.goal = saved;
      return c;
    }
    if (r.stage === 1) {
      const fake = { type: 'goto', x: r.x, y: r.y, alt: 0, speed: 0.6, tol: 0.05, resolve: () => { r.stage = 2; } };
      const saved = this.goal; this.goal = fake;
      const c = this._goalCommand(fake, dt);
      this.goal = saved;
      return c;
    }
    if (r.stage === 2) {
      d.motors = false;
      d.vel.set(0, 0, 0);
      d.pos.z = 0;
      r.stage = 3;
      if (d.batteryPct() < 50) {
        d.batteryWh = d.cfg.battery_wh; this.swaps++; this._lowBatt = false;
        this.log('Battery swapped at home', 'info');
      }
      this.log('Landed at home. Press M (or Start) to resume the mission.', 'warn');
      this.ui.toast('Landed at home', 'Press M to resume the mission', 'warn', 4000);
    }
    return { x: 0, y: 0, z: 0, yaw: 0 };
  }

  togglePause() {
    this.paused = !this.paused;
    this.ui.setMode(this.mode, this.paused);
    this.log(this.paused ? 'Simulation paused' : 'Simulation resumed', 'info');
  }

  // ── spraying ──────────────────────────────────────────────────────
  _credit(pid, ml) { this.sprayedMl[pid] = (this.sprayedMl[pid] || 0) + ml; }

  _nearestPoint(maxDist = 0.5) {
    let best = null, bd = maxDist;
    for (const [pid, p] of Object.entries(this.points)) {
      const dd = Math.hypot(p.x_m - this.drone.pos.x, p.y_m - this.drone.pos.y);
      if (dd < bd) { bd = dd; best = Number(pid); }
    }
    return best;
  }

  _dispense(dt) {
    const d = this.drone;
    const flow = this.cfg.spray_flow_ml_per_s * dt;
    if (this.mode === 'AUTO' && this.sprayJob) {
      const job = this.sprayJob;
      const ml = Math.min(flow, job.remaining, d.tankMl);
      if (ml <= 1e-6) { this.sprayJob = null; this._yield = true; job.resolve(d.tankMl <= 1e-6 && job.remaining > 1e-6 ? 'empty' : 'ok'); return false; }
      job.remaining -= ml;
      d.setTank(d.tankMl - ml);
      this._credit(job.pid, ml);
      return true;
    }
    if (this.mode === 'MANUAL' && this.keys[' '] && d.tankMl > 0 && d.motors && d.pos.z < 4.5) {
      const ml = Math.min(flow, d.tankMl);
      d.setTank(d.tankMl - ml);
      const pid = this._nearestPoint(0.5);
      this.manual.spray_ml += ml;
      if (!this.burst) this.burst = { ml: 0, soil: 0, plants: {} };
      this.burst.ml += ml;
      if (pid != null) {
        this._credit(pid, ml);
        const pt = this.points[pid];
        if (pt.decision?.action !== 'spray') this.manual.nontarget_ml += ml;
        this.burst.plants[pid] = (this.burst.plants[pid] || 0) + ml;
        this._checkDone(pid);
      } else { this.manual.soil_ml += ml; this.burst.soil += ml; }
      return true;
    }
    if (this.mode === 'MANUAL' && this.keys[' '] && d.tankMl <= 0 && !this._warnedEmpty) {
      this._warnedEmpty = true;
      this.ui.toast('Tank empty', 'Return home (H) to refill', 'warn');
    }
    this._endBurst();
    return false;
  }

  _endBurst() {
    if (!this.burst) return;
    const b = this.burst; this.burst = null;
    if (b.ml < 0.05) return;
    const parts = Object.entries(b.plants).filter(([, ml]) => ml >= 0.05).map(([pid, ml]) => {
      const tgt = this.points[pid]?.decision?.action === 'spray';
      return `${ml.toFixed(1)} mL on plant #${pid}${tgt ? ' (target)' : ' (not a target)'}`;
    });
    if (b.soil >= 0.05) parts.push(`${b.soil.toFixed(1)} mL on bare soil`);
    this.log(`Pilot sprayed ${b.ml.toFixed(1)} mL: ${parts.join(', ')}`, 'pilot');
  }

  _checkDone(pid) {
    const pt = this.points[pid];
    if (!pt || pt.decision?.action !== 'spray' || pt.status === 'done') return;
    if ((this.sprayedMl[pid] || 0) >= 0.95 * pt.decision.dosage_ml) {
      pt.status = 'done';
      this.targetsDone++;
      this.world3d.setPointStatus(pid, 'done');
      this.world3d.markWet(pid);
      this.ui.setLabel(`p${pid}`, `✓ ${pt.decision.dosage_ml} mL`, 'dose done', toV(pt.x_m, pt.y_m, 1.25));
    }
  }

  sprayAt(pid, ml) {
    return new Promise((resolve) => { this.sprayJob = { pid, remaining: ml, resolve }; });
  }

  // ── per-step update ───────────────────────────────────────────────
  update(dt) {
    if (this.paused || !this.started) return;
    // While the autopilot is between commands (e.g. waiting for the model or
    // the planner), sim time stands still instead of being lost.
    if (this.mode === 'AUTO' && !this.goal && !this.sprayJob && !this.finished) return;
    this.simTime += dt;
    const d = this.drone;
    let cmd;
    if (this.mode === 'MANUAL') cmd = this._manualCommand();
    else if (this.mode === 'RTH' && this.rth) cmd = this._rthCommand(dt);
    else if (this.goal) cmd = this._goalCommand(this.goal, dt);
    else cmd = { x: 0, y: 0, z: 0, yaw: 0 };
    const sprayingNow = this._dispense(dt);
    d.spraying = sprayingNow;
    const px = d.pos.x, py = d.pos.y;
    d.step(dt, cmd);
    if (this.inSortie) this.sortieDist += Math.hypot(d.pos.x - px, d.pos.y - py);
    if (d.motors) this.flightTime += dt;
    if (sprayingNow) this.particles.emit(d.nozzleWorld(), 520, dt, d.vel);
    if (this.simTime - this.lastTrack >= 0.5) {
      this.lastTrack = this.simTime;
      this.track.push([+this.simTime.toFixed(1), +d.pos.x.toFixed(2), +d.pos.y.toFixed(2), +d.pos.z.toFixed(2), this.mode[0]]);
      this.world3d.pushTrail(d.pos.x, d.pos.y, d.pos.z + 0.3);
    }
    if (d.batteryPct() < 20 && d.motors && this.mode === 'AUTO' && !this._lowBatt) {
      this._lowBatt = true;
      this.ui.toast('Battery low', 'Returning home for a battery swap', 'bad');
      this.log('Battery below 20 % — returning home', 'bad');
      this.returnHome('battery low');
    }
  }

  // ── the mission ───────────────────────────────────────────────────
  async run() {
    this.started = true;
    this.ui.setMode(this.mode, this.paused);
    this.log(`Mission ${this.simId} started · ${this.world.plants.length}-plant field`, 'ok');
    try {
      await this._phaseMarkers();
      await this._phaseMap();
      await this._phaseScan();
      const plan = await this._phasePlan();
      if (plan) await this._phaseSpray(plan);
      await this._phaseReturn();
      await this._finish();
    } catch (err) {
      console.error(err);
      this.log(`Mission stopped: ${err.message}`, 'bad');
      this.ui.toast('Mission stopped', err.message, 'bad', 8000);
    }
  }

  async _phaseMarkers() {
    this.setPhase('markers');
    await this.takeoff(MARKER_ALT);
    const corners = this.world.planned_corners;
    for (let i = 0; i < 4; i++) {
      const [cx, cy] = corners[i];
      this.log(`Flying to corner ${i + 1} of 4 to drop marker M${i}`, 'info');
      await this.goto(cx, cy, MARKER_ALT);
      await this.goto(cx, cy, DROP_ALT, 1.0, 0.06);
      await this.wait(0.3);
      await this._dropMarker(i);
      await this.goto(cx, cy, MARKER_ALT, 1.2, 0.15);
    }
  }

  async _dropMarker(id) {
    const d = this.drone;
    const mesh = this.world3d.markerMesh(id);
    const start = toV(d.pos.x, d.pos.y, d.pos.z + 0.1);
    mesh.position.copy(start);
    mesh.rotation.y = d.heading;
    this.world3d.addDroppedMarker(id, mesh);
    d.setMarkers(d.markersOnboard - 1);
    // landing point: release point plus a little drift
    const lx = d.pos.x + (Math.random() - 0.5) * 0.06, ly = d.pos.y + (Math.random() - 0.5) * 0.06;
    const fallH = d.pos.z + 0.1;
    const fallT = Math.sqrt(2 * fallH / 9.81);
    const t0 = this.simTime, heading = d.heading;
    this._anim = (now) => {
      const t = Math.min(1, (now - t0) / fallT);
      mesh.position.copy(toV(lx, ly, Math.max(0.0125, fallH * (1 - t * t))));
      mesh.rotation.y = heading * (1 - t);
      mesh.rotation.x = Math.sin(t * Math.PI) * 0.25;
      if (t >= 1) { mesh.rotation.set(0, 0, 0); this._anim = null; }
    };
    await this.wait(fallT + 0.05);
    this._anim?.(t0 + fallT);
    this.markers.push({ marker_id: id, x_m: lx, y_m: ly });
    this.markersPlaced = this.markers.length;
    this.ui.setLabel(`m${id}`, `M${id}`, 'marker', toV(lx, ly, 0.6));
    const res = await api.dropMarker(this.simId, id, lx, ly);
    this.log(`Marker M${id} placed (${res.offset_from_planned_m * 100 | 0} cm from the planned corner)`, 'ok');
  }

  async _phaseMap() {
    this.setPhase('map');
    const W = this.world.field_width_m, H = this.world.field_height_m;
    this.log(`Climbing to ${MAP_ALT} m for the overview photo`, 'info');
    this.world3d.showFootprint(true);
    this.footHalf = Math.atan((Math.max(W, H) / 2 + 1.4) / MAP_ALT);
    await this.goto(W / 2, H / 2, MAP_ALT, this.cfg.cruise_speed_mps, 0.25);
    await this.wait(0.6);
    this.world3d.flash(); this.drone.flash();
    this.log('Overview photo taken — finding markers and plants…', 'info');
    const ov = await api.overview(this.simId);
    this.ui.showStill(ov.image_annotated, `Overview · ${ov.plants_detected} plants · ${ov.markers_found} markers`);
    setTimeout(() => this.ui.hideStill(), 6000);
    this.markersFound = ov.markers_found;
    if (!ov.calibration_valid) throw new Error('Could not read all four markers — boundary unknown');
    this.boundary = ov.boundary;
    this.world3d.setBoundary(ov.boundary);
    const b = ov.boundary;
    this.ui.setLabel('dimW', `${ov.field_width_m.toFixed(2)} m`, 'dim', toV((b[0][0] + b[1][0]) / 2, (b[0][1] + b[1][1]) / 2 - 0.5, 0.3));
    this.ui.setLabel('dimH', `${ov.field_height_m.toFixed(2)} m`, 'dim', toV((b[0][0] + b[3][0]) / 2 - 0.7, (b[0][1] + b[3][1]) / 2, 0.3));
    this.log(`Boundary locked from ${ov.markers_found}/4 markers: ${ov.field_width_m.toFixed(2)} m × ${ov.field_height_m.toFixed(2)} m (reprojection ${ov.reprojection_error_px} px)`, 'ok');
    this.ui.toast(`Boundary ${ov.field_width_m.toFixed(1)} × ${ov.field_height_m.toFixed(1)} m · ${ov.plants_detected} plants located`,
      ov.suspected_gaps ? `${ov.suspected_gaps} gap(s) in the rows will be checked on the low pass` : 'Every plant now has a position in metres', '', 4200);
    const plants = [...ov.plants].sort((a, b2) => (a.y_m - b2.y_m) || (a.x_m - b2.x_m));
    plants.forEach((p, i) => setTimeout(() => {
      this.points[p.plant_id] = { x_m: p.x_m, y_m: p.y_m, status: 'pending' };
      this.world3d.addDataPoint(p);
    }, i * 18));
    this.plantsFound = ov.plants_detected;
    this.scanTargets = ov.scan_targets;
    this.scanTotal = ov.scan_targets.length;
    const ev = ov.eval;
    this.log(`${ov.plants_detected} plants located${ov.suspected_gaps ? `, ${ov.suspected_gaps} row gap(s) to check` : ''} · check vs ground truth: ${ev.matched}/${ev.true_plants}, mean error ${(ev.mean_position_error_m * 100).toFixed(1)} cm`, 'ok');
    await this.wait(1.6);
  }

  async _phaseScan() {
    this.setPhase('scan');
    this.footHalf = GIMBAL_HALF_ANGLE;
    const targets = this.scanTargets;
    this.log(`Low-altitude scan: ${targets.length} photo stops at ${SCAN_ALT} m, row by row`, 'info');
    const pending = [];
    for (const t of targets) {
      await this.goto(t.x_m, t.y_m, SCAN_ALT, SCAN_SPEED, 0.07);
      await this.wait(0.2);
      this.world3d.flash(); this.drone.flash();
      this.photos++;
      const p = api.capture(this.simId, t.target_id, this.drone.pos.x, this.drone.pos.y, this.options.gradcam())
        .then((res) => this._onCapture(t, res))
        .catch((err) => { this._resolved = (this._resolved || 0) + 1; this.log(`Photo ${t.target_id} failed: ${err.message}`, 'bad'); });
      pending.push(p);
      while (pending.length - this._resolvedCount() > 6) await this.wait(0.3);   // let the model catch up
    }
    if (pending.length) {
      this.log('Scan complete — waiting for the last photos to be classified', 'info');
      await Promise.all(pending);
    }
    this.world3d.showFootprint(false);
  }

  _resolvedCount() { return this._resolved || 0; }

  _onCapture(t, res) {
    this._resolved = (this._resolved || 0) + 1;
    this.ui.showCapture(res);
    if (!res.found) {
      if (t.kind === 'gap') this.log(`Gap at (${t.x_m.toFixed(1)}, ${t.y_m.toFixed(1)}) checked — no plant`, 'info');
      else {
        this.points[res.plant_id] && (this.points[res.plant_id].status = 'inspect');
        this.world3d.setPointStatus(res.plant_id, 'inspect');
        this.log(`#${res.plant_id} not visible in the close-up → manual inspection`, 'warn');
      }
      return;
    }
    if (res.recovered) {
      const p = { plant_id: res.plant_id, x_m: res.position[0], y_m: res.position[1] };
      this.points[res.plant_id] = { ...p, status: 'pending' };
      this.world3d.addDataPoint(p);
      this.plantsFound++;
      this.log(`Plant #${res.plant_id} found in a row gap on the low pass`, 'ok');
    }
    const st = statusFor(res.decision);
    const pt = this.points[res.plant_id];
    if (pt) { pt.status = st; pt.decision = res.decision; pt.label = res.label; }
    this.world3d.setPointStatus(res.plant_id, st);
    const d = res.decision;
    const pos = toV(pt?.x_m ?? res.position[0], pt?.y_m ?? res.position[1], 1.25);
    if (d.action === 'spray') this.ui.setLabel(`p${res.plant_id}`, `${d.dosage_ml} mL`, `dose${st === 'lb' ? ' lb' : ''}`, pos);
    else if (d.action === 'inspect') this.ui.setLabel(`p${res.plant_id}`, '? inspect', 'inspect', pos);
    const conf = Math.round(res.confidence * 100);
    const sev = res.label === 'healthy' ? '' : ` · severity ${res.severity_category}`;
    const act = d.action === 'spray' ? `SPRAY ${d.dosage_ml} mL` : d.action.replace('_', ' ').toUpperCase();
    this.log(`#${res.plant_id} ${PRETTY[res.label]} ${conf}%${sev} → ${act}`, d.action === 'spray' ? 'warn' : d.action === 'inspect' ? 'bad' : 'ok');
  }

  async _phasePlan() {
    this.setPhase('plan');
    await this.goto(this.drone.pos.x, this.drone.pos.y, TRANSIT_ALT + 1.5, 1.0, 0.2);
    this.log('Deciding every plant and computing the shortest spray route…', 'info');
    const plan = await api.plan(this.simId);
    const s = plan.summary, m = plan.metrics, a = s.actions;
    for (const dcs of plan.decisions) {          // final decisions (identical rules, whole field)
      const pt = this.points[dcs.plant_id];
      if (pt) pt.decision = { action: dcs.action, dosage_ml: dcs.dosage_ml, treatment: dcs.treatment, reason: dcs.reason };
    }
    this.targets = a.spray || 0;
    this.targetsDone = 0;
    const skipped = (a.no_spray || 0) + (a.monitor || 0);
    if (plan.status === 'no_targets' || !plan.sorties.length) {
      this.log('No plant needs spraying — nothing to fly', 'ok');
      this.ui.toast('No spraying needed', `${skipped} plants checked`, '', 5000);
      return null;
    }
    this.routeSorties = plan.sorties.map((so) => ({ treatment: so.treatment, stops: so.stops }));
    this.routeStart = null;
    this.world3d.setRoute(this.routeSorties, this.world.home);
    const saveDist = m.distance_saved_vs_blanket_pct;
    this.log(`Plan: ${this.targets} plants need spray, ${skipped} skipped (${a.no_spray || 0} healthy, ${a.monitor || 0} mild)${a.inspect ? `, ${a.inspect} for manual inspection` : ''}`, 'ok');
    this.log(`Shortest route ${m.optimised_distance_m} m in ${m.sorties} tank load(s) vs ${m.blanket_distance_m} m blanket pass (${saveDist}% shorter) · id-order route ${m.id_order_distance_m} m`, 'ok');
    this.log(`Chemical ${s.chemical_used_ml} mL vs ${s.blanket_baseline_ml} mL blanket (${s.chemical_saved_pct}% less) · est. flight ${(m.flight_time_s / 60).toFixed(1)} min`, 'ok');
    this.ui.toast(`Spray plan: ${this.targets} of ${Object.keys(this.points).length} plants`,
      `Route ${m.optimised_distance_m} m vs ${m.blanket_distance_m} m blanket · chemical −${s.chemical_saved_pct}%`, 'big', 6000);
    this.plan = plan;
    await this.wait(2.5);
    return plan;
  }

  async _homeAndRefill(sortie) {
    const h = this.world.home, d = this.drone;
    if (Math.hypot(d.pos.x - h.x_m, d.pos.y - h.y_m) > 0.15 || d.pos.z > 0.05) {
      let r = await this.goto(h.x_m, h.y_m, Math.max(TRANSIT_ALT, d.pos.z), this.cfg.cruise_speed_mps, 0.12);
      if (r === 'aborted') return r;
      r = await this.land();
      if (r === 'aborted') return r;
    }
    this.inSortie = false;
    const load = Math.min(this.cfg.tank_capacity_ml, Math.ceil(sortie.load_ml * 1.05 + 1));
    const leftover = d.tankMl;
    this.log(`Refilling: ${load} mL ${sortie.treatment.replace(/_/g, ' ')} mix${leftover > 0.5 ? ` (${leftover.toFixed(0)} mL of the old mix returned)` : ''}`, 'info');
    d.setTank(0, sortie.treatment);
    this.warp = 8;
    const steps = 20;
    for (let i = 1; i <= steps; i++) {
      const r = await this.wait(this.cfg.refill_time_s / steps);
      if (r === 'aborted') { this.warp = 1; return r; }
      d.setTank(load * i / steps, sortie.treatment);
    }
    if (d.batteryPct() < 50) {
      d.batteryWh = d.cfg.battery_wh; this.swaps++; this._lowBatt = false;
      this.log('Battery swapped during refill', 'info');
    }
    this.warp = 1;
    const r = await this.takeoff(TRANSIT_ALT);
    if (r !== 'aborted') this.inSortie = true;
    return r;
  }

  _remainingTargets() {
    const ids = [];
    const pending = [this.active, ...this.queue].filter(Boolean);
    for (const so of pending) for (const st of so.stops) {
      if ((this.sprayedMl[st.plant_id] || 0) < 0.95 * st.dosage_ml) ids.push(st.plant_id);
    }
    return [...new Set(ids)];
  }

  async _replan() {
    const d = this.drone;
    const remaining = this._remainingTargets();
    if (!remaining.length) { this.active = null; this.queue = []; return; }
    const pos = [d.pos.x, d.pos.y];
    const r = await api.replan(this.simId, pos, d.tankMl, d.tankTreatment, remaining);
    this.queue = r.sorties.map((so) => ({
      treatment: so.treatment, load_ml: so.load_ml, fromCurrent: so.from_current_position,
      // keep the full planned dose; the spray loop tops up whatever is missing
      stops: so.stops.filter((st) => st.dosage_ml - (this.sprayedMl[st.plant_id] || 0) > 0.05),
    }));
    this.active = null;
    this.routeSorties = this.queue;
    this.routeStart = this.queue[0]?.fromCurrent ? pos : null;
    if (!this.queue[0]?.fromCurrent) this.routeStart = null;
    this.world3d.setRoute(this.queue, this.world.home, this.routeStart);
    this.log(`Route re-planned from (${pos[0].toFixed(1)}, ${pos[1].toFixed(1)}): ${remaining.length} targets left, ${r.distance_m} m, ${r.sorties.length} sortie(s)`, 'ok');
    this.ui.toast('Route re-optimised', `${remaining.length} targets left · ${r.distance_m} m from here`, '', 3500);
  }

  async _phaseSpray(plan) {
    this.setPhase('spray');
    this.queue = plan.sorties.map((so) => ({ treatment: so.treatment, load_ml: so.load_ml, stops: so.stops.map((st) => ({ ...st })), fromCurrent: false }));
    this.active = null;
    let n = 0;
    for (;;) {
      if (this.replanRequested) { this.replanRequested = false; await this._replan(); continue; }
      if (!this.active) {
        if (!this.queue.length) break;
        this.active = this.queue.shift();
        n++;
        this.log(`Sortie ${n}: ${this.active.stops.length} plants, ${this.active.treatment.replace(/_/g, ' ')}`, 'info');
        if (!this.active.fromCurrent) {
          const r = await this._homeAndRefill(this.active);
          if (r === 'aborted') { this.queue.unshift(this.active); this.active = null; n--; continue; }
        } else if (!this.drone.motors || this.drone.pos.z < 0.5) {
          const r = await this.takeoff(TRANSIT_ALT);
          if (r === 'aborted') continue;
        }
        this.inSortie = true;
      }
      if (!this.active.stops.length) { this.active = null; continue; }
      const st = this.active.stops[0];
      let r = await this.goto(st.x_m, st.y_m, SPRAY_ALT, this.cfg.cruise_speed_mps, 0.06);
      if (r === 'aborted') continue;
      const need = Math.max(0, st.dosage_ml - (this.sprayedMl[st.plant_id] || 0));
      if (need > 0.05) {
        if (this.drone.tankMl < need - 0.05) {
          this.log(`Not enough mix for #${st.plant_id} — re-planning`, 'warn');
          this.replanRequested = true;
          continue;
        }
        r = await this.sprayAt(st.plant_id, need);
        if (r === 'aborted') continue;
        this._checkDone(st.plant_id);
        const pt = this.points[st.plant_id];
        const topUp = need < st.dosage_ml - 0.05 ? ` (topped up ${need.toFixed(1)} mL)` : '';
        this.log(`Sprayed #${st.plant_id}: ${st.dosage_ml} mL${topUp} (${PRETTY[pt?.label] ?? st.disease})`, 'ok');
      }
      this.active.stops.shift();
      this.routeSorties = [this.active, ...this.queue];
    }
  }

  async _phaseReturn() {
    this.setPhase('return');
    const h = this.world.home, d = this.drone;
    if (d.motors || d.pos.z > 0.05 || Math.hypot(d.pos.x - h.x_m, d.pos.y - h.y_m) > 0.2) {
      await this.goto(h.x_m, h.y_m, Math.max(TRANSIT_ALT, d.pos.z), this.cfg.cruise_speed_mps, 0.12);
      await this.land();
    }
    this.inSortie = false;
    if (this.drone.tankMl > 0.5) this.log(`Landed · ${this.drone.tankMl.toFixed(0)} mL unused mix returned to the station`, 'ok');
    else this.log('Landed at home', 'ok');
  }

  async _finish() {
    this.setPhase('done');
    this.finished = true;
    const d = this.drone;
    const telemetry = {
      flight_time_s: +this.flightTime.toFixed(1), distance_m: +d.odometer.toFixed(2),
      energy_wh: +d.energyWh.toFixed(2),
      battery_used_pct: +(100 * d.energyWh / this.cfg.battery_wh).toFixed(1),
      phases: Object.fromEntries(Object.entries(this.phaseStats).map(([k, v]) => [k, { time_s: +v.time_s.toFixed(1), distance_m: +v.distance_m.toFixed(2) }])),
      sprayed_ml: Object.fromEntries(Object.entries(this.sprayedMl).map(([k, v]) => [k, +v.toFixed(2)])),
      manual: { ...this.manual, time_s: +this.manual.time_s.toFixed(1) },
      spray_route_m: +this.sortieDist.toFixed(2),
      events: this.events, track: this.track,
    };
    const sum = await api.finish(this.simId, telemetry);
    this.log(`Mission complete · chemical ${sum.chemical.used_ml} mL vs ${sum.chemical.blanket_ml} mL blanket (${sum.chemical.saved_pct}% less)`, 'ok');
    this.ui.showSummary(sum, {
      flightTime: this.flightTime, distance: d.odometer, battery: 100 * d.energyWh / this.cfg.battery_wh,
      energy: d.energyWh, swaps: this.swaps, markersPlaced: this.markersPlaced, markersFound: this.markersFound,
    }, this.simId, this.plan?.waypoint_files);
    this.summary = sum;
  }
}
