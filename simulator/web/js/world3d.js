// 3D scene: environment, field, plants, markers, boundary, route, camera footprint.
// Field frame: x east, y north (metres). three.js frame: x, up = y, z = -north.
import * as THREE from '../vendor/three.bundle.min.js';

export const toV = (x, y, alt = 0) => new THREE.Vector3(x, alt, -y);

export const STATUS = {
  pending: 0xcfd8dc, healthy: 0x4caf50, monitor: 0xffd54f, spray: 0xff8a3d,
  lb: 0xe5484d, inspect: 0xb388ff, done: 0x29b6f6, gap: 0x90a4ae,
};
export const TREATMENT_COLOR = { protectant_fungicide: 0xff8a3d, late_blight_fungicide: 0xe5484d };

function rng(seed) {                       // small deterministic PRNG (mulberry32)
  let a = seed >>> 0;
  return () => {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function canvasTexture(w, h, draw, repeat) {
  const c = document.createElement('canvas');
  c.width = w; c.height = h;
  draw(c.getContext('2d'), w, h);
  const tex = new THREE.CanvasTexture(c);
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.anisotropy = 8;
  if (repeat) { tex.wrapS = tex.wrapT = THREE.RepeatWrapping; tex.repeat.set(repeat, repeat); }
  return tex;
}

function speckle(ctx, w, h, n, colors, rmin, rmax, r) {
  for (let i = 0; i < n; i++) {
    ctx.fillStyle = colors[Math.floor(r() * colors.length)];
    const s = rmin + r() * (rmax - rmin);
    ctx.fillRect(r() * w, r() * h, s, s);
  }
}

export class World3D {
  constructor(canvas) {
    // ?quality=low turns off shadows and anti-aliasing for slow machines.
    this.low = new URLSearchParams(window.location.search).get('quality') === 'low';
    const renderer = new THREE.WebGLRenderer({ canvas, antialias: !this.low });
    renderer.setPixelRatio(this.low ? 1 : Math.min(window.devicePixelRatio || 1, 2));
    renderer.shadowMap.enabled = !this.low;
    renderer.shadowMap.type = THREE.PCFShadowMap;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.05;
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer = renderer;

    this.scene = new THREE.Scene();
    this.scene.fog = new THREE.Fog(0xd9e6ee, 45, 170);
    this.camera = new THREE.PerspectiveCamera(50, 1, 0.1, 600);
    this.fpvCam = new THREE.PerspectiveCamera(78, 1, 0.05, 400);
    this.droneCam = new THREE.PerspectiveCamera(56, 4 / 3, 0.05, 300);   // gimbal camera, nadir

    this.dynamic = new THREE.Group();          // everything that belongs to one mission
    this.scene.add(this.dynamic);
    this._buildEnvironment();
    this.texLoader = new THREE.TextureLoader();
    this.markerMeshes = {};
    this.points = {};                          // plant_id -> data-point visuals
    this.labels = [];
  }

  resize(w, h) {
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h; this.camera.updateProjectionMatrix();
    this.fpvCam.aspect = w / h; this.fpvCam.updateProjectionMatrix();
  }

  // ── static environment ────────────────────────────────────────────────
  _buildEnvironment() {
    const s = this.scene;
    const sky = new THREE.Mesh(new THREE.SphereGeometry(320, 32, 16), new THREE.ShaderMaterial({
      side: THREE.BackSide, depthWrite: false, fog: false,
      uniforms: { top: { value: new THREE.Color(0x3d7fc4) }, bottom: { value: new THREE.Color(0xdbe8f0) } },
      vertexShader: 'varying vec3 vP; void main(){ vP = position; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }',
      fragmentShader: 'uniform vec3 top; uniform vec3 bottom; varying vec3 vP; void main(){ float h = clamp(normalize(vP).y, 0.0, 1.0); gl_FragColor = vec4(mix(bottom, top, pow(h, 0.55)), 1.0); }',
    }));
    sky.renderOrder = -1;
    this.sky = sky;
    s.add(sky);

    s.add(new THREE.HemisphereLight(0xe3f0ff, 0x5a4630, 1.05));
    const sun = new THREE.DirectionalLight(0xfff1dc, 2.4);
    sun.castShadow = true;
    sun.shadow.mapSize.set(2048, 2048);
    const sc = sun.shadow.camera;
    sc.left = -16; sc.right = 16; sc.top = 16; sc.bottom = -16; sc.near = 1; sc.far = 80;
    sun.shadow.bias = -0.0004;
    sun.shadow.normalBias = 0.02;
    this.sun = sun;
    s.add(sun, sun.target);

    const r = rng(42);
    const grass = canvasTexture(256, 256, (ctx, w, h) => {
      ctx.fillStyle = '#6f8f47'; ctx.fillRect(0, 0, w, h);
      speckle(ctx, w, h, 2600, ['#5f803c', '#7f9f52', '#86a656', '#6a8a42', '#56763a'], 1, 3, r);
    }, 90);
    const ground = new THREE.Mesh(new THREE.PlaneGeometry(420, 420),
      new THREE.MeshStandardMaterial({ map: grass, roughness: 1 }));
    ground.rotation.x = -Math.PI / 2;
    ground.position.y = -0.01;
    ground.receiveShadow = true;
    s.add(ground);
  }

  _trees(cx, cy, seed) {
    const r = rng(seed);
    const n = 70;
    const trunk = new THREE.InstancedMesh(new THREE.CylinderGeometry(0.12, 0.18, 1, 6),
      new THREE.MeshStandardMaterial({ color: 0x5b4330, roughness: 1 }), n);
    const crown = new THREE.InstancedMesh(new THREE.IcosahedronGeometry(1, 0),
      new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.95, flatShading: true }), n);
    const m = new THREE.Matrix4(), q = new THREE.Quaternion(), c = new THREE.Color();
    for (let i = 0; i < n; i++) {
      const ang = r() * Math.PI * 2;
      const dist = 20 + r() * 34;
      const x = cx + Math.cos(ang) * dist * 1.25, z = -(cy + Math.sin(ang) * dist);
      const h = 2.2 + r() * 2.5, w = 1.4 + r() * 1.6;
      m.compose(new THREE.Vector3(x, h / 2, z), q, new THREE.Vector3(1, h, 1));
      trunk.setMatrixAt(i, m);
      q.setFromEuler(new THREE.Euler(r(), r() * 3, r()));
      m.compose(new THREE.Vector3(x, h + w * 0.55, z), q, new THREE.Vector3(w, w * 1.15, w));
      crown.setMatrixAt(i, m);
      crown.setColorAt(i, c.setHSL(0.27 + r() * 0.06, 0.45, 0.24 + r() * 0.1));
      q.identity();
    }
    trunk.castShadow = crown.castShadow = true;
    const g = new THREE.Group(); g.add(trunk, crown);
    return g;
  }

  // ── per-mission content ───────────────────────────────────────────────
  setWorld(world) {
    this.clearDynamic();
    this.world = world;
    const W = world.field_width_m, H = world.field_height_m;
    this.center = { x: W / 2, y: H / 2 };
    const sun = this.sun;
    sun.position.copy(toV(W / 2 - 12, H / 2 - 9, 24));
    sun.target.position.copy(toV(W / 2, H / 2, 0));
    this.sky.position.copy(toV(W / 2, H / 2, 0));
    this.dynamic.add(this._trees(W / 2, H / 2, world.seed + 3));

    // soil bed with planting rows
    const pad = 1.2, fw = W + 2 * pad, fh = H + 2 * pad;
    const ppm = 64;
    const r = rng(world.seed + 11);
    const soil = canvasTexture(Math.round(fw * ppm), Math.round(fh * ppm), (ctx, w, h) => {
      ctx.fillStyle = '#7a5a3c'; ctx.fillRect(0, 0, w, h);
      for (let row = 0; row < world.rows; row++) {          // raised beds along each row
        const y = (fh - (pad + 1 + row)) * ppm;
        ctx.fillStyle = 'rgba(140,104,70,0.55)';
        ctx.fillRect(0, y - 0.32 * ppm, w, 0.64 * ppm);
        ctx.fillStyle = 'rgba(60,42,26,0.45)';
        ctx.fillRect(0, y + 0.42 * ppm, w, 0.12 * ppm);
      }
      speckle(ctx, w, h, Math.round(w * h / 60), ['#6c4f34', '#86664a', '#5e432c', '#8f7154', '#735438'], 1, 3, r);
      ctx.strokeStyle = 'rgba(70,52,33,0.8)'; ctx.lineWidth = 6;
      ctx.strokeRect(3, 3, w - 6, h - 6);
    });
    const bed = new THREE.Mesh(new THREE.PlaneGeometry(fw, fh),
      new THREE.MeshStandardMaterial({ map: soil, roughness: 1 }));
    bed.rotation.x = -Math.PI / 2;
    bed.position.copy(toV(W / 2, H / 2, 0.002));
    bed.receiveShadow = true;
    this.dynamic.add(bed);

    this._plants(world);
    this._home(world.home);
    this._plannedCorners(world.planned_corners);

    // trail of the actual flight path
    const max = 20000;
    const tg = new THREE.BufferGeometry();
    tg.setAttribute('position', new THREE.BufferAttribute(new Float32Array(max * 3), 3));
    tg.setDrawRange(0, 0);
    this.trail = new THREE.Line(tg, new THREE.LineBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.45 }));
    this.trail.frustumCulled = false;
    this.trailCount = 0; this.trailMax = max;
    this.dynamic.add(this.trail);

    // camera footprint (frustum edges + ground patch)
    const fg = new THREE.BufferGeometry();
    fg.setAttribute('position', new THREE.BufferAttribute(new Float32Array(16 * 3), 3));
    this.frustum = new THREE.LineSegments(fg, new THREE.LineBasicMaterial({ color: 0x5cc8ff, transparent: true, opacity: 0.7 }));
    this.frustum.frustumCulled = false;
    this.footprint = new THREE.Mesh(new THREE.PlaneGeometry(1, 1),
      new THREE.MeshBasicMaterial({ color: 0x5cc8ff, transparent: true, opacity: 0.12, depthWrite: false }));
    this.footprint.rotation.x = -Math.PI / 2;
    this.flashLevel = 0;
    this.dynamic.add(this.frustum, this.footprint);
    this.showFootprint(false);
  }

  clearDynamic() {
    for (const child of [...this.dynamic.children]) {
      this.dynamic.remove(child);
      child.traverse?.((o) => { o.geometry?.dispose?.(); });
    }
    this.markerMeshes = {}; this.points = {}; this.route = null; this.boundary = null;
  }

  _plants(world) {
    const leavesPer = 18;
    const n = world.plants.length * leavesPer;
    const leaves = new THREE.InstancedMesh(new THREE.IcosahedronGeometry(1, 0),
      new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.82, flatShading: true }), n);
    const mounds = new THREE.InstancedMesh(new THREE.CylinderGeometry(0.36, 0.44, 0.06, 14),
      new THREE.MeshStandardMaterial({ color: 0x5a3f28, roughness: 1 }), world.plants.length);
    const greens = [0x2e7d32, 0x388e3c, 0x43a047, 0x4f9a3a, 0x2f6f2f];
    const disease = {
      early_blight: { f: 0.38, c: [0x9a7b3c, 0xb8a03e, 0x6e5230, 0xc4b45a] },
      late_blight: { f: 0.52, c: [0x4a3826, 0x3a2e21, 0x655336, 0x2f3a24] },
      septoria_leaf_spot: { f: 0.32, c: [0xa7a37a, 0x86704c, 0xc9c49b, 0x7d8a5a] },
    };
    const m = new THREE.Matrix4(), q = new THREE.Quaternion(), e = new THREE.Euler(), c = new THREE.Color();
    let i = 0;
    world.plants.forEach((p, k) => {
      const r = rng(world.seed * 1000 + p.gid);
      const R = 0.36 * (p.size_m / 0.855);
      m.compose(toV(p.x_m, p.y_m, 0.03), q.identity(), new THREE.Vector3(1, 1, 1));
      mounds.setMatrixAt(k, m);
      const dz = disease[p.true_label];
      for (let j = 0; j < leavesPer; j++) {
        const t = (j + 0.5) / leavesPer;
        const ang = j * 2.39996 + r() * 0.4;
        const rad = R * Math.sqrt(t) * (0.85 + r() * 0.2);
        const hgt = 0.14 + (1 - (rad / R) ** 2) * 0.36 + r() * 0.06;
        const pos = toV(p.x_m + Math.cos(ang) * rad, p.y_m + Math.sin(ang) * rad, hgt);
        e.set(r() * 1.2 - 0.6, r() * Math.PI * 2, r() * 1.2 - 0.6);
        q.setFromEuler(e);
        const sz = 0.1 + r() * 0.05;
        m.compose(pos, q, new THREE.Vector3(sz * 1.35, sz * 0.55, sz));
        leaves.setMatrixAt(i, m);
        let col = greens[Math.floor(r() * greens.length)];
        if (dz && r() < dz.f) col = dz.c[Math.floor(r() * dz.c.length)];
        c.setHex(col).offsetHSL(0, 0, (r() - 0.5) * 0.06);
        leaves.setColorAt(i, c);
        i++;
      }
    });
    leaves.castShadow = true; leaves.receiveShadow = true;
    mounds.receiveShadow = true;
    this.dynamic.add(mounds, leaves);
  }

  _home(home) {
    const g = new THREE.Group();
    const tex = canvasTexture(256, 256, (ctx, w) => {
      ctx.fillStyle = '#2b3136'; ctx.beginPath(); ctx.arc(w / 2, w / 2, w / 2 - 2, 0, Math.PI * 2); ctx.fill();
      ctx.strokeStyle = '#f5f5f5'; ctx.lineWidth = 10; ctx.beginPath(); ctx.arc(w / 2, w / 2, w / 2 - 24, 0, Math.PI * 2); ctx.stroke();
      ctx.fillStyle = '#f5f5f5'; ctx.font = 'bold 120px sans-serif'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
      ctx.fillText('H', w / 2, w / 2 + 6);
    });
    const pad = new THREE.Mesh(new THREE.CylinderGeometry(0.8, 0.8, 0.05, 40),
      [new THREE.MeshStandardMaterial({ color: 0x2b3136 }), new THREE.MeshStandardMaterial({ map: tex }), new THREE.MeshStandardMaterial({ color: 0x2b3136 })]);
    pad.position.y = 0.025; pad.receiveShadow = true;
    g.add(pad);
    // refill station + shed for scale
    const station = new THREE.Group();
    const table = new THREE.Mesh(new THREE.BoxGeometry(0.9, 0.7, 0.5), new THREE.MeshStandardMaterial({ color: 0x47606f, roughness: 0.7 }));
    table.position.y = 0.35; table.castShadow = true;
    const jerry1 = new THREE.Mesh(new THREE.BoxGeometry(0.22, 0.3, 0.14), new THREE.MeshStandardMaterial({ color: 0xff8a3d }));
    jerry1.position.set(-0.18, 0.85, 0); jerry1.castShadow = true;
    const jerry2 = new THREE.Mesh(new THREE.BoxGeometry(0.22, 0.3, 0.14), new THREE.MeshStandardMaterial({ color: 0xe5484d }));
    jerry2.position.set(0.18, 0.85, 0); jerry2.castShadow = true;
    station.add(table, jerry1, jerry2);
    station.position.set(-1.7, 0, 0.2);
    const shed = new THREE.Group();
    const walls = new THREE.Mesh(new THREE.BoxGeometry(2.6, 2.1, 2.0), new THREE.MeshStandardMaterial({ color: 0xc9b38a, roughness: 0.9 }));
    walls.position.y = 1.05; walls.castShadow = walls.receiveShadow = true;
    const roof = new THREE.Mesh(new THREE.ConeGeometry(2.0, 0.8, 4), new THREE.MeshStandardMaterial({ color: 0x8a3b2e, roughness: 0.8 }));
    roof.position.y = 2.5; roof.rotation.y = Math.PI / 4; roof.scale.set(1.0, 1, 0.78); roof.castShadow = true;
    shed.add(walls, roof);
    shed.position.set(-4.4, 0, 1.4);
    g.add(station, shed);
    g.position.copy(toV(home.x_m, home.y_m, 0));
    this.dynamic.add(g);
  }

  _plannedCorners(corners) {
    const pts = [...corners, corners[0]].map(([x, y]) => toV(x, y, 0.04));
    const geo = new THREE.BufferGeometry().setFromPoints(pts);
    const line = new THREE.Line(geo, new THREE.LineDashedMaterial({ color: 0xffd54f, dashSize: 0.35, gapSize: 0.25, transparent: true, opacity: 0.9 }));
    line.computeLineDistances();
    this.planned = line;
    this.dynamic.add(line);
    // corner flags the operator set on the map
    this.flags = corners.map(([x, y]) => {
      const f = new THREE.Group();
      const pole = new THREE.Mesh(new THREE.CylinderGeometry(0.015, 0.015, 0.9, 6), new THREE.MeshStandardMaterial({ color: 0xdddddd }));
      pole.position.y = 0.45;
      const flag = new THREE.Mesh(new THREE.PlaneGeometry(0.28, 0.18), new THREE.MeshStandardMaterial({ color: 0xffd54f, side: THREE.DoubleSide }));
      flag.position.set(0.14, 0.8, 0);
      f.add(pole, flag);
      f.position.copy(toV(x - 0.55, y - 0.55, 0));
      this.dynamic.add(f);
      return f;
    });
  }

  // ── markers ───────────────────────────────────────────────────────────
  markerMesh(id) {
    const plate = this.world.marker.plate_m;
    const tex = this.texLoader.load(`/sim/aruco/${id}.png`);
    tex.colorSpace = THREE.SRGBColorSpace;
    tex.magFilter = THREE.NearestFilter;
    const side = new THREE.MeshStandardMaterial({ color: 0xf2f2f2 });
    const top = new THREE.MeshStandardMaterial({ map: tex, roughness: 0.6 });
    const mesh = new THREE.Mesh(new THREE.BoxGeometry(plate, 0.025, plate), [side, side, top, side, side, side]);
    mesh.castShadow = true; mesh.receiveShadow = true;
    return mesh;
  }

  addDroppedMarker(id, mesh) {
    this.markerMeshes[id] = mesh;
    this.dynamic.add(mesh);
  }

  setBoundary(corners) {
    if (this.boundary) this.dynamic.remove(this.boundary);
    const g = new THREE.Group();
    const pts = [...corners, corners[0]].map(([x, y]) => toV(x, y, 0.06));
    const line = new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts), new THREE.LineBasicMaterial({ color: 0x5cc8ff }));
    g.add(line);
    const wallMat = new THREE.MeshBasicMaterial({ color: 0x5cc8ff, transparent: true, opacity: 0.13, side: THREE.DoubleSide, depthWrite: false });
    for (let i = 0; i < 4; i++) {
      const a = corners[i], b = corners[(i + 1) % 4];
      const len = Math.hypot(b[0] - a[0], b[1] - a[1]);
      const wall = new THREE.Mesh(new THREE.PlaneGeometry(len, 0.5), wallMat);
      wall.position.copy(toV((a[0] + b[0]) / 2, (a[1] + b[1]) / 2, 0.25));
      wall.rotation.y = Math.atan2(b[1] - a[1], b[0] - a[0]);
      g.add(wall);
    }
    this.boundary = g;
    this.dynamic.add(g);
    if (this.planned) this.planned.material.opacity = 0.25;
  }

  // ── data points (detected plants) ─────────────────────────────────────
  addDataPoint(plant) {
    const g = new THREE.Group();
    const ringMat = new THREE.MeshBasicMaterial({ color: STATUS.pending, transparent: true, opacity: 0.55, depthWrite: false });
    const ring = new THREE.Mesh(new THREE.RingGeometry(0.48, 0.54, 40), ringMat);
    ring.rotation.x = -Math.PI / 2; ring.position.y = 0.035;
    const pin = new THREE.Mesh(new THREE.CylinderGeometry(0.012, 0.012, 0.95, 6), new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.6 }));
    pin.position.y = 0.475;
    const headMat = new THREE.MeshBasicMaterial({ color: STATUS.pending });
    const head = new THREE.Mesh(new THREE.SphereGeometry(0.06, 12, 8), headMat);
    head.position.y = 0.98;
    g.add(ring, pin, head);
    g.position.copy(toV(plant.x_m, plant.y_m, 0));
    g.scale.setScalar(0.001);
    g.userData = { born: performance.now(), ringMat, headMat, plant };
    this.points[plant.plant_id] = g;
    this.dynamic.add(g);
    return g;
  }

  setPointStatus(plantId, status) {
    const g = this.points[plantId];
    if (!g) return;
    g.userData.ringMat.color.setHex(STATUS[status] ?? STATUS.pending);
    g.userData.ringMat.opacity = status === 'pending' ? 0.55 : 0.92;
    g.userData.headMat.color.setHex(STATUS[status] ?? STATUS.pending);
    g.userData.status = status;
  }

  markWet(plantId) {
    const g = this.points[plantId];
    if (!g || g.userData.wet) return;
    const wet = new THREE.Mesh(new THREE.CircleGeometry(0.44, 32),
      new THREE.MeshBasicMaterial({ color: 0x1e88e5, transparent: true, opacity: 0, depthWrite: false }));
    wet.rotation.x = -Math.PI / 2; wet.position.y = 0.03;
    g.add(wet);
    g.userData.wet = wet;
  }

  // ── route ─────────────────────────────────────────────────────────────
  setRoute(sorties, home, startPos = null) {
    if (this.route) this.dynamic.remove(this.route);
    const g = new THREE.Group();
    sorties.forEach((s, idx) => {
      const color = TREATMENT_COLOR[s.treatment] ?? 0x5cc8ff;
      const begin = (idx === 0 && startPos) ? startPos : [home.x_m, home.y_m];
      const pts = [begin, ...s.stops.map((st) => [st.x_m, st.y_m]), [home.x_m, home.y_m]];
      const path = new THREE.CurvePath();
      for (let i = 0; i < pts.length - 1; i++) {
        path.add(new THREE.LineCurve3(toV(pts[i][0], pts[i][1], 0.12), toV(pts[i + 1][0], pts[i + 1][1], 0.12)));
      }
      const tube = new THREE.Mesh(new THREE.TubeGeometry(path, Math.max(8, pts.length * 8), 0.035, 6, false),
        new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.9 }));
      g.add(tube);
      for (let i = 0; i < pts.length - 1; i++) {             // direction arrows
        const a = pts[i], b = pts[i + 1];
        const len = Math.hypot(b[0] - a[0], b[1] - a[1]);
        if (len < 0.6) continue;
        const cone = new THREE.Mesh(new THREE.ConeGeometry(0.09, 0.24, 8), new THREE.MeshBasicMaterial({ color }));
        cone.position.copy(toV((a[0] + b[0]) / 2, (a[1] + b[1]) / 2, 0.12));
        cone.rotation.set(0, 0, 0);
        const dir = toV(b[0] - a[0], b[1] - a[1], 0).normalize();
        cone.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir);
        g.add(cone);
      }
    });
    this.route = g;
    this.dynamic.add(g);
  }

  clearRoute() { if (this.route) { this.dynamic.remove(this.route); this.route = null; } }

  // ── per-frame helpers ────────────────────────────────────────────────
  pushTrail(x, y, alt) {
    if (!this.trail || this.trailCount >= this.trailMax) return;
    const arr = this.trail.geometry.attributes.position.array;
    arr.set([x, alt, -y], this.trailCount * 3);
    this.trailCount++;
    this.trail.geometry.attributes.position.needsUpdate = true;
    this.trail.geometry.setDrawRange(0, this.trailCount);
  }

  showFootprint(on) { if (this.frustum) { this.frustum.visible = on; this.footprint.visible = on; } }

  updateFootprint(x, y, alt, halfAngle) {
    if (!this.frustum || !this.frustum.visible) return;
    const s = Math.max(0.05, alt * Math.tan(halfAngle));
    const sy = s * 0.75;
    const apex = toV(x, y, alt - 0.12);
    const c = [toV(x - s, y - sy, 0.04), toV(x + s, y - sy, 0.04), toV(x + s, y + sy, 0.04), toV(x - s, y + sy, 0.04)];
    const p = [];
    for (const k of c) p.push(apex, k);
    for (let i = 0; i < 4; i++) p.push(c[i], c[(i + 1) % 4]);
    const arr = this.frustum.geometry.attributes.position.array;
    p.forEach((v, i) => arr.set([v.x, v.y, v.z], i * 3));
    this.frustum.geometry.attributes.position.needsUpdate = true;
    this.footprint.position.copy(toV(x, y, 0.045));
    this.footprint.scale.set(2 * s, 2 * sy, 1);
    this.footprint.material.opacity = 0.12 + this.flashLevel * 0.6;
  }

  flash() { this.flashLevel = 1; }

  tick(dt) {
    this.flashLevel = Math.max(0, this.flashLevel - dt * 3);
    const now = performance.now();
    for (const g of Object.values(this.points)) {
      const age = (now - g.userData.born) / 1000;
      if (age < 0.6) g.scale.setScalar(Math.max(0.001, 1 - Math.pow(1 - age / 0.6, 3) * 1.0) * (1 + 0.25 * Math.sin(age * 10) * (1 - age / 0.6)));
      else if (g.scale.x !== 1) g.scale.setScalar(1);
      const wet = g.userData.wet;
      if (wet && wet.material.opacity < 0.42) wet.material.opacity += dt * 0.6;
    }
  }

  project(v, cam = this.camera, w, h) {
    const p = v.clone().project(cam);
    return { x: (p.x * 0.5 + 0.5) * w, y: (-p.y * 0.5 + 0.5) * h, visible: p.z < 1 && p.z > -1 };
  }
}
