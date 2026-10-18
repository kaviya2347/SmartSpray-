// Entry point: wires the scene, the drone, the mission and the controls together.
import * as THREE from '../vendor/three.bundle.min.js';
import { OrbitControls } from '../vendor/three.bundle.min.js';
import { api } from './api.js';
import { World3D, toV } from './world3d.js';
import { Drone, SprayParticles } from './drone.js';
import { Mission } from './mission.js';
import { UI } from './ui.js';

const $ = (id) => document.getElementById(id);
const canvas = $('view');
const world3d = new World3D(canvas);
const ui = new UI();
const particles = new SprayParticles(world3d.scene);
const controls = new OrbitControls(world3d.camera, canvas);
controls.enableDamping = true;
controls.maxPolarAngle = Math.PI * 0.48;
controls.minDistance = 3; controls.maxDistance = 90;
controls.enabled = false;

let drone = null, mission = null, session = null;
let speed = 2, camMode = 'auto', lastField = { preset: 'small', seed: 7 };
let labelsOn = true;
const camState = { pos: new THREE.Vector3(-6, 12, 10), look: new THREE.Vector3(5, 0, -4) };
world3d.camera.position.copy(camState.pos);

function resize() {
  world3d.resize(window.innerWidth, window.innerHeight);
}
window.addEventListener('resize', resize);
resize();

// ── backend check ──────────────────────────────────────────────────────
(async () => {
  const el = $('backend');
  try {
    await api.health();
    el.textContent = 'Backend connected';
    el.className = 'backend ok';
    $('btnBrief').disabled = false;
  } catch (e) {
    el.textContent = 'Backend not reachable — start it with: uvicorn app.main:app';
    el.className = 'backend bad';
  }
})();

// ── session / mission setup ────────────────────────────────────────────
async function prepare(preset, seed) {
  $('btnBrief').disabled = true;
  $('btnBrief').textContent = 'Preparing…';
  try {
    session = await api.newSession(seed, preset);
  } catch (e) {
    $('backend').textContent = e.message; $('backend').className = 'backend bad';
    $('btnBrief').disabled = false; $('btnBrief').textContent = 'Prepare mission';
    return;
  }
  lastField = { preset, seed };
  ui.reset();
  world3d.setWorld(session.world);
  if (drone) world3d.scene.remove(drone.group);
  drone = new Drone(session.mission_config);
  drone.reset(session.world.home);
  world3d.scene.add(drone.group);
  mission = new Mission({
    world3d, drone, particles, ui, session,
    options: { gradcam: () => $('optGradcam').checked },
  });
  mission.onPhase = () => { if (camMode === 'auto') camState.snap = false; };
  const w = session.world;
  ui.log(0, `Field ready: ${w.plants.length} plants in ${w.rows} rows, ${w.field_width_m} m × ${w.field_height_m} m (synthetic, seed ${w.seed})`, 'ok');
  ui.log(0, 'Operator marked 4 field corners (yellow flags). Press Start mission.', 'info');
  ui.setMode('AUTO', false);
  $('briefing').hidden = true;
  $('summary').hidden = true;
  $('btnStart').disabled = false;
  $('btnStart').textContent = 'Start mission';
  $('btnPause').disabled = true;
  $('btnPause').textContent = 'Pause';
  $('btnBrief').disabled = false; $('btnBrief').textContent = 'Prepare mission';
  camState.pos.copy(toV(w.home.x_m - 6, w.home.y_m - 7, 7));
  world3d.camera.position.copy(camState.pos);
  controls.target.copy(toV(w.field_width_m / 2, w.field_height_m / 2, 0));
}

$('btnBrief').addEventListener('click', () => {
  const [preset, seedSpec] = document.querySelector('input[name=field]:checked').value.split(':');
  const seed = seedSpec === 'random' ? 1 + Math.floor(Math.random() * 9998) : Number(seedSpec);
  prepare(preset, seed === 7 && seedSpec === 'random' ? 8 : seed);
});

$('btnStart').addEventListener('click', () => {
  if (!mission) return;
  if (!mission.started) {
    mission.run();
    $('btnPause').disabled = false;
    $('btnStart').disabled = true;
  } else if (mission.mode === 'RTH' || mission.mode === 'MANUAL') {
    mission.resumeAuto();
  }
});
$('btnPause').addEventListener('click', () => { if (mission?.started) mission.togglePause(); });
$('btnAgain').addEventListener('click', () => prepare(lastField.preset, lastField.seed));
$('btnNew').addEventListener('click', () => { $('summary').hidden = true; $('briefing').hidden = false; });
$('btnClose').addEventListener('click', () => { $('summary').hidden = true; });
$('optLabels').addEventListener('change', (e) => { labelsOn = e.target.checked; });

function segment(id, onPick) {
  $(id).addEventListener('click', (e) => {
    const b = e.target.closest('button');
    if (!b) return;
    $(id).querySelectorAll('button').forEach((x) => x.classList.toggle('on', x === b));
    onPick(b.dataset.v);
  });
}
segment('speed', (v) => { speed = Number(v); });
segment('cams', (v) => setCam(v));

function setCam(v) {
  camMode = v;
  $('cams').querySelectorAll('button').forEach((x) => x.classList.toggle('on', x.dataset.v === v));
  controls.enabled = v === 'orbit';
  if (v === 'orbit' && session) controls.target.copy(toV(session.world.field_width_m / 2, session.world.field_height_m / 2, 0));
}

// ── keyboard ───────────────────────────────────────────────────────────
const FLY_KEYS = new Set(['w', 'a', 's', 'd', 'q', 'e', 'r', 'f', ' ', 'shift', 'arrowup', 'arrowdown', 'arrowleft', 'arrowright', 'pageup', 'pagedown']);
let lastHint = 0;
window.addEventListener('keydown', (e) => {
  if (e.target.tagName === 'INPUT' && e.target.type !== 'checkbox' && e.target.type !== 'radio') return;
  const k = e.key.toLowerCase();
  if (FLY_KEYS.has(k)) {
    e.preventDefault();
    if (mission) {
      mission.keys[k] = true;
      if (mission.started && !mission.finished && mission.mode !== 'MANUAL' && k !== 'shift' && performance.now() - lastHint > 4000) {
        lastHint = performance.now();
        ui.toast('Autopilot is flying', 'Press M to take manual control', '', 2400);
      }
    }
    return;
  }
  if (e.repeat || !mission) return;
  if (k === 'm') mission.toggleManual();
  else if (k === 'h') mission.returnHome();
  else if (k === 'p') mission.togglePause();
  else if (['1', '2', '3', '4'].includes(k)) setCam(['follow', 'top', 'orbit', 'fpv'][Number(k) - 1]);
  else if (k === '0') setCam('auto');
});
window.addEventListener('keyup', (e) => { if (mission) mission.keys[e.key.toLowerCase()] = false; });
window.addEventListener('blur', () => { if (mission) mission.keys = {}; });

// ── camera ─────────────────────────────────────────────────────────────
function autoView() {
  if (!mission) return 'orbitHome';
  if (mission.mode === 'MANUAL') return 'follow';
  const h = session.world.home;
  if (mission.started && drone && !drone.motors && Math.hypot(drone.pos.x - h.x_m, drone.pos.y - h.y_m) < 0.5) return 'home';
  switch (mission.phase) {
    case 'map': return 'wide';
    case 'plan': return 'top';
    case 'done': return 'wide';
    case 'preflight': return 'orbitHome';
    default: return 'follow';
  }
}

function updateCamera(dt) {
  const cam = world3d.camera;
  if (!session) return cam;
  const W = session.world.field_width_m, H = session.world.field_height_m;
  const mode = camMode === 'auto' ? autoView() : camMode;
  if (mode === 'orbit') { controls.update(); return cam; }
  if (mode === 'fpv' && drone) {
    const f = drone.forward();
    const c = world3d.fpvCam;
    c.position.copy(toV(drone.pos.x + f.x * 0.25, drone.pos.y + f.y * 0.25, drone.pos.z + 0.2));
    c.lookAt(toV(drone.pos.x + f.x * 4, drone.pos.y + f.y * 4, Math.max(0, drone.pos.z - 2.2)));
    return c;
  }
  let pos, look;
  if (mode === 'follow' && drone) {
    const f = drone.forward();
    const back = 5.2 + drone.pos.z * 0.4;
    pos = toV(drone.pos.x - f.x * back, drone.pos.y - f.y * back, drone.pos.z + 2.8 + drone.pos.z * 0.25);
    look = toV(drone.pos.x + f.x * 1.2, drone.pos.y + f.y * 1.2, Math.max(0, drone.pos.z - 0.6));
  } else if (mode === 'top') {
    pos = toV(W / 2 - 1, H / 2 - 9.5, 19);
    look = toV(W / 2 - 1, H / 2 - 0.5, 0);
  } else if (mode === 'wide') {
    const dz = drone ? drone.pos.z : 0;
    pos = toV(W / 2 - 5, H / 2 - 34, 10 + dz * 0.8);
    look = toV(W / 2, H / 2, dz * 0.6);
  } else if (mode === 'home') {
    const h = session.world.home;
    pos = toV(h.x_m + 4.2, h.y_m + 3.4, 3.4);
    look = toV(h.x_m - 0.4, h.y_m, 0.3);
  } else {   // orbitHome: slow orbit before take-off
    const t = performance.now() / 1000 * 0.06;
    pos = toV(W / 2 + Math.cos(t) * 15, H / 2 + Math.sin(t) * 15 - 2, 9);
    look = toV(W / 2, H / 2, 0);
  }
  const k = 1 - Math.exp(-dt * 3.2);
  camState.pos.lerp(pos, k);
  camState.look.lerp(look, k);
  cam.position.copy(camState.pos);
  cam.lookAt(camState.look);
  controls.target.copy(camState.look);
  return cam;
}

// ── render loop ────────────────────────────────────────────────────────
let last = performance.now(), hud = 0;
async function frame(now) {
  const dtReal = Math.min(0.05, (now - last) / 1000);
  last = now;
  if (mission) {
    const simDt = dtReal * speed * mission.warp;
    const steps = Math.max(1, Math.ceil(simDt / (1 / 60)));
    for (let i = 0; i < steps; i++) {
      mission.update(simDt / steps);
      // A goal just finished: let the mission's async code queue the next one
      // (a few microtask turns) so no simulated time is skipped.
      if (mission.takeYield()) for (let k = 0; k < 8; k++) await null;
    }
    mission._anim?.(mission.simTime);
    particles.update(mission.paused ? 0 : dtReal * Math.min(speed, 4));
  }
  if (drone) {
    drone.render(dtReal * (mission?.paused ? 0 : 1));
    if (mission?.footHalf && world3d.frustum?.visible) world3d.updateFootprint(drone.pos.x, drone.pos.y, drone.pos.z + 0.2, mission.footHalf);
  }
  world3d.tick(dtReal);
  const cam = updateCamera(dtReal);
  const r = world3d.renderer;
  const W = window.innerWidth, H = window.innerHeight;
  r.setScissorTest(false);
  r.setViewport(0, 0, W, H);
  r.render(world3d.scene, cam);
  // picture-in-picture: the drone's nadir gimbal camera
  const pip = $('camview').getBoundingClientRect();
  if (drone && pip.width > 0 && $('camStill').hidden) {
    const dc = world3d.droneCam;
    dc.aspect = pip.width / pip.height; dc.updateProjectionMatrix();
    dc.position.copy(toV(drone.pos.x, drone.pos.y, drone.pos.z + 0.18));
    dc.up.set(0, 0, -1);
    dc.lookAt(toV(drone.pos.x, drone.pos.y, 0));
    drone.group.visible = false;
    r.setScissorTest(true);
    r.setScissor(pip.left, H - pip.bottom, pip.width, pip.height);
    r.setViewport(pip.left, H - pip.bottom, pip.width, pip.height);
    r.render(world3d.scene, dc);
    r.setScissorTest(false);
    drone.group.visible = true;
  }
  ui.updateLabels((v) => world3d.project(v, cam, W, H), labelsOn, pip);
  hud += dtReal;
  if (mission && hud > 0.1) {
    hud = 0;
    ui.telemetry(drone, mission);
    ui.drawMinimap(session.world, mission, drone);
    const st = $('btnStart');
    if (mission.started && !mission.finished) {
      const canResume = mission.mode === 'RTH' || mission.mode === 'MANUAL';
      st.disabled = !canResume;
      st.textContent = canResume ? 'Resume autopilot' : 'Mission running';
    } else if (mission.finished) { st.disabled = true; st.textContent = 'Mission complete'; }
    $('btnPause').textContent = mission.paused ? 'Resume' : 'Pause';
    $('keys').classList.toggle('compact', mission.started && mission.mode !== 'MANUAL');
  }
  requestAnimationFrame(frame);
}
requestAnimationFrame(frame);

// Exposed for automated checks in a headless browser.
window.__sim = {
  get mission() { return mission; }, get drone() { return drone; }, world3d,
  setSpeed: (v) => { speed = v; }, prepare, setCam,
};
