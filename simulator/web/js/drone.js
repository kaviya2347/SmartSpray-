// Spray drone: 3D model, simple flight dynamics, spray particles.
import * as THREE from '../vendor/three.bundle.min.js';
import { toV } from './world3d.js';

const MIX_COLOR = { protectant_fungicide: 0xffa060, late_blight_fungicide: 0xff6670 };

export class Drone {
  constructor(cfg) {
    this.cfg = cfg;                               // mission_config from the backend
    this.group = new THREE.Group();
    this.body = new THREE.Group();
    this.group.add(this.body);
    this.props = [];
    this._build();
    this.reset({ x_m: 0, y_m: 0 });
  }

  reset(home) {
    this.pos = new THREE.Vector3(home.x_m, home.y_m, 0);   // field x, y, altitude
    this.vel = new THREE.Vector3();
    this.acc = new THREE.Vector3();
    this.heading = Math.PI / 2;                   // radians from east, CCW: start facing north
    this.yawRate = 0;
    this.motors = false;
    this.batteryWh = this.cfg.battery_wh;
    this.energyWh = 0;
    this.odometer = 0;
    this.tankMl = 0;
    this.tankTreatment = null;
    this.markersOnboard = 4;
    this.spraying = false;
    this._spin = 0;
    this.setMarkers(4);
    this.setTank(0, null);
  }

  _build() {
    const dark = new THREE.MeshStandardMaterial({ color: 0x22272b, roughness: 0.5, metalness: 0.3 });
    const accent = new THREE.MeshStandardMaterial({ color: 0x3f9b4f, roughness: 0.45, metalness: 0.2 });
    const b = this.body;
    const hull = new THREE.Mesh(new THREE.BoxGeometry(0.36, 0.11, 0.3), dark);
    const canopy = new THREE.Mesh(new THREE.SphereGeometry(0.16, 20, 12, 0, Math.PI * 2, 0, Math.PI / 2), accent);
    canopy.scale.set(1.25, 0.55, 1.0); canopy.position.y = 0.05;
    b.add(hull, canopy);
    const armGeo = new THREE.BoxGeometry(0.46, 0.03, 0.045);
    const motorGeo = new THREE.CylinderGeometry(0.04, 0.046, 0.06, 14);
    const bladeGeo = new THREE.BoxGeometry(0.38, 0.006, 0.034);
    const discMat = new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.1, depthWrite: false, side: THREE.DoubleSide });
    for (let i = 0; i < 4; i++) {
      const a = Math.PI / 4 + i * Math.PI / 2;
      const ax = Math.cos(a) * 0.42, az = Math.sin(a) * 0.42;
      const arm = new THREE.Mesh(armGeo, dark);
      arm.position.set(ax / 2, 0.01, az / 2); arm.rotation.y = -a;
      const motor = new THREE.Mesh(motorGeo, dark);
      motor.position.set(ax, 0.045, az);
      const prop = new THREE.Group();
      const b1 = new THREE.Mesh(bladeGeo, dark), b2 = new THREE.Mesh(bladeGeo, dark);
      b2.rotation.y = Math.PI / 2;
      const disc = new THREE.Mesh(new THREE.CircleGeometry(0.2, 28), discMat);
      disc.rotation.x = -Math.PI / 2;
      prop.add(b1, b2, disc);
      prop.position.set(ax, 0.085, az);
      prop.userData.dir = i % 2 ? 1 : -1;
      this.props.push(prop);
      b.add(arm, motor, prop);
      const led = new THREE.Mesh(new THREE.SphereGeometry(0.018, 8, 6), new THREE.MeshBasicMaterial({ color: ax > 0 ? (az < 0 ? 0x33ff66 : 0xff3344) : 0xffffff }));
      led.position.set(ax, 0.0, az);
      b.add(led);
    }
    // landing skids
    const legMat = new THREE.MeshStandardMaterial({ color: 0x9aa3a8, roughness: 0.6 });
    for (const s of [-1, 1]) {
      const skid = new THREE.Mesh(new THREE.CylinderGeometry(0.012, 0.012, 0.5, 6), legMat);
      skid.rotation.x = Math.PI / 2; skid.rotation.z = Math.PI / 2;
      skid.position.set(0, -0.26, s * 0.16);
      skid.rotation.set(0, 0, Math.PI / 2);
      const l1 = new THREE.Mesh(new THREE.CylinderGeometry(0.01, 0.01, 0.22, 6), legMat);
      l1.position.set(-0.13, -0.16, s * 0.14); l1.rotation.x = s * 0.25;
      const l2 = l1.clone(); l2.position.x = 0.13;
      b.add(skid, l1, l2);
    }
    // spray tank with visible liquid level
    const tank = new THREE.Mesh(new THREE.CylinderGeometry(0.1, 0.1, 0.17, 18),
      new THREE.MeshStandardMaterial({ color: 0xffffff, transparent: true, opacity: 0.35, roughness: 0.2 }));
    tank.position.y = -0.14;
    this.liquid = new THREE.Mesh(new THREE.CylinderGeometry(0.088, 0.088, 1, 18),
      new THREE.MeshStandardMaterial({ color: 0xffa060, roughness: 0.3, emissive: 0x331100 }));
    this.liquidBase = -0.14 - 0.08;
    b.add(tank, this.liquid);
    // boom + nozzles
    const boom = new THREE.Mesh(new THREE.BoxGeometry(0.03, 0.02, 0.62), dark);
    boom.position.set(0, -0.24, 0);
    b.add(boom);
    this.nozzles = [-0.28, 0.28].map((z) => {
      const n = new THREE.Mesh(new THREE.ConeGeometry(0.02, 0.05, 8), new THREE.MeshStandardMaterial({ color: 0x5cc8ff }));
      n.position.set(0, -0.27, z); n.rotation.x = Math.PI;
      b.add(n);
      return n;
    });
    // gimbal camera (front, looking down)
    const cam = new THREE.Mesh(new THREE.BoxGeometry(0.07, 0.06, 0.07), dark);
    cam.position.set(0.17, -0.07, 0);
    const lens = new THREE.Mesh(new THREE.CylinderGeometry(0.022, 0.022, 0.02, 12), new THREE.MeshStandardMaterial({ color: 0x0b1a2a, metalness: 0.8, roughness: 0.15 }));
    lens.position.set(0.17, -0.105, 0);
    b.add(cam, lens);
    // marker cartridge: plates still on board
    this.cartridge = new THREE.Group();
    for (let i = 0; i < 4; i++) {
      const plate = new THREE.Mesh(new THREE.BoxGeometry(0.2, 0.012, 0.2), new THREE.MeshStandardMaterial({ color: 0xf2f2f2 }));
      plate.position.set(-0.08, -0.07 - i * 0.016, 0);
      this.cartridge.add(plate);
    }
    b.add(this.cartridge);
    this.flashLight = new THREE.PointLight(0xffffff, 0, 6);
    this.flashLight.position.set(0.17, -0.2, 0);
    b.add(this.flashLight);
    this.group.traverse((o) => { if (o.isMesh) o.castShadow = true; });
  }

  setMarkers(n) {
    this.markersOnboard = n;
    this.cartridge.children.forEach((p, i) => { p.visible = i < n; });
  }

  setTank(ml, treatment) {
    this.tankMl = Math.max(0, ml);
    if (treatment !== undefined) this.tankTreatment = treatment;
    const level = Math.min(1, this.tankMl / this.cfg.tank_capacity_ml);
    this.liquid.visible = level > 0.005;
    this.liquid.scale.y = Math.max(0.001, level * 0.16);
    this.liquid.position.y = this.liquidBase + level * 0.08;
    this.liquid.material.color.setHex(MIX_COLOR[this.tankTreatment] ?? 0x90caf9);
  }

  nozzleWorld() { return this.nozzles.map((n) => n.getWorldPosition(new THREE.Vector3())); }
  forward() { return { x: Math.cos(this.heading), y: Math.sin(this.heading) }; }
  speed() { return Math.hypot(this.vel.x, this.vel.y); }

  // Velocity-command flight model: the autopilot and the pilot both ask for a
  // velocity; acceleration and climb are limited like a real multirotor.
  step(dt, cmd) {
    const c = this.cfg;
    const maxA = 3.0, maxVz = 1.8, maxAz = 2.0;
    const prev = this.vel.clone();
    if (!this.motors) { this.vel.set(0, 0, 0); this.acc.set(0, 0, 0); return; }
    for (const k of ['x', 'y']) {
      const dv = THREE.MathUtils.clamp(cmd[k] - this.vel[k], -maxA * dt, maxA * dt);
      this.vel[k] += dv;
    }
    const vzWanted = THREE.MathUtils.clamp(cmd.z, -maxVz, maxVz);
    this.vel.z += THREE.MathUtils.clamp(vzWanted - this.vel.z, -maxAz * dt, maxAz * dt);
    const yr = THREE.MathUtils.clamp(cmd.yaw ?? 0, -1.8, 1.8);
    this.yawRate += THREE.MathUtils.clamp(yr - this.yawRate, -6 * dt, 6 * dt);
    this.heading += this.yawRate * dt;
    const before = this.pos.clone();
    this.pos.addScaledVector(this.vel, dt);
    if (this.pos.z < 0) { this.pos.z = 0; this.vel.z = Math.max(0, this.vel.z); }
    this.odometer += Math.hypot(this.pos.x - before.x, this.pos.y - before.y, (this.pos.z - before.z));
    this.acc.copy(this.vel).sub(prev).divideScalar(Math.max(dt, 1e-4));
    // power: hover + extra for forward flight and climbing
    const v = this.speed();
    const p = c.hover_power_w + (c.cruise_power_w - c.hover_power_w) * Math.min(1.6, (v / c.cruise_speed_mps) ** 2)
      + Math.max(0, this.vel.z) * 25;
    const e = p * dt / 3600;
    this.energyWh += e;
    this.batteryWh = Math.max(0, this.batteryWh - e);
  }

  batteryPct() { return 100 * this.batteryWh / this.cfg.battery_wh; }

  // Visual update (every rendered frame).
  render(dt) {
    this.group.position.copy(toV(this.pos.x, this.pos.y, this.pos.z + 0.3));
    this.group.rotation.y = this.heading;
    // tilt into the direction of travel (body frame)
    const f = this.forward();
    const vf = this.vel.x * f.x + this.vel.y * f.y;
    const vr = this.vel.x * f.y - this.vel.y * f.x;
    const af = this.acc.x * f.x + this.acc.y * f.y;
    const ar = this.acc.x * f.y - this.acc.y * f.x;
    const pitch = THREE.MathUtils.clamp(-(vf * 0.06 + af * 0.05), -0.4, 0.4);
    const roll = THREE.MathUtils.clamp(vr * 0.06 + ar * 0.05, -0.4, 0.4);
    this.body.rotation.z += (pitch - this.body.rotation.z) * Math.min(1, dt * 8);
    this.body.rotation.x += (roll - this.body.rotation.x) * Math.min(1, dt * 8);
    const spin = this.motors ? 60 : 0;
    this._spin += spin * dt;
    for (const p of this.props) {
      p.rotation.y = this._spin * p.userData.dir;
      p.children[2].material.opacity = this.motors ? 0.14 : 0.0;
      p.children[0].visible = p.children[1].visible = !this.motors;
    }
    this.flashLight.intensity = Math.max(0, this.flashLight.intensity - dt * 30);
  }

  flash() { this.flashLight.intensity = 12; }
}

export class SprayParticles {
  constructor(scene, max = 2400) {
    this.max = max;
    this.pos = new Float32Array(max * 3);
    this.vel = new Float32Array(max * 3);
    this.life = new Float32Array(max);
    for (let i = 0; i < max; i++) this.pos[i * 3 + 1] = -50;
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(this.pos, 3));
    this.points = new THREE.Points(g, new THREE.PointsMaterial({
      color: 0xd8f1ff, size: 0.05, transparent: true, opacity: 0.75, depthWrite: false, sizeAttenuation: true,
    }));
    this.points.frustumCulled = false;
    scene.add(this.points);
    this.next = 0;
    this.carry = 0;
  }

  emit(origins, rate, dt, droneVel) {
    this.carry += rate * dt;
    const n = Math.floor(this.carry);
    this.carry -= n;
    for (let k = 0; k < n; k++) {
      const o = origins[k % origins.length];
      const i = this.next; this.next = (this.next + 1) % this.max;
      this.pos[i * 3] = o.x; this.pos[i * 3 + 1] = o.y; this.pos[i * 3 + 2] = o.z;
      this.vel[i * 3] = (Math.random() - 0.5) * 0.9 + droneVel.x * 0.4;
      this.vel[i * 3 + 1] = -2.4 - Math.random() * 1.2;
      this.vel[i * 3 + 2] = (Math.random() - 0.5) * 0.9 - droneVel.y * 0.4;
      this.life[i] = 1.4;
    }
  }

  update(dt) {
    for (let i = 0; i < this.max; i++) {
      if (this.life[i] <= 0) continue;
      this.life[i] -= dt;
      this.vel[i * 3 + 1] -= 3.5 * dt;
      this.vel[i * 3] *= 1 - 0.8 * dt;
      this.vel[i * 3 + 2] *= 1 - 0.8 * dt;
      this.pos[i * 3] += this.vel[i * 3] * dt;
      this.pos[i * 3 + 1] += this.vel[i * 3 + 1] * dt;
      this.pos[i * 3 + 2] += this.vel[i * 3 + 2] * dt;
      if (this.pos[i * 3 + 1] < 0.05 || this.life[i] <= 0) { this.life[i] = 0; this.pos[i * 3 + 1] = -50; }
    }
    this.points.geometry.attributes.position.needsUpdate = true;
  }
}
