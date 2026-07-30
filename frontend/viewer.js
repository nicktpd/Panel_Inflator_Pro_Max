// Three.js scene: orbit viewer with studio lighting, ground shadow,
// before/after mesh sets with a crossfade morph, and raycast part picking.

import * as THREE from 'three';
import { OrbitControls } from './vendor/OrbitControls.js';
import { GLTFLoader } from './vendor/GLTFLoader.js';

const PART_COLOR = 0xb9b2a7;      // neutral upholstery
const HW_COLOR = 0x6f7683;        // hardware steel-gray
const SELECT_EMISSIVE = 0x8a6a1f; // amber glow on selection

export class Viewer {
  constructor(canvas) {
    this.canvas = canvas;
    this.onSelect = null;          // callback(partId | null)
    this.onAfterLoaded = null;     // callback() when the after-mesh reloads
    this.pickEnabled = true;       // brush mode suppresses click-select
    this.selectedPartId = null;
    this.morphT = 1.0;             // 0 = flat original, 1 = pillowed
    this.beforeGroup = null;
    this.afterGroup = null;
    this.partClass = {};           // partId -> 'pillow' | 'passthrough'

    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;

    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x14161b);
    this.scene.fog = new THREE.Fog(0x14161b, 4000, 9000);

    this.camera = new THREE.PerspectiveCamera(40, 1, 1, 20000);
    this.camera.position.set(600, -700, 600);
    this.camera.up.set(0, 0, 1);   // panels are z-up

    this.controls = new OrbitControls(this.camera, canvas);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.08;

    // Neutral studio lighting: soft hemisphere + warm key with shadow + cool fill.
    this.scene.add(new THREE.HemisphereLight(0xdfe4ee, 0x30343c, 0.85));
    const key = new THREE.DirectionalLight(0xfff2df, 1.6);
    key.position.set(900, -600, 1400);
    key.castShadow = true;
    key.shadow.mapSize.set(2048, 2048);
    this.keyLight = key;
    this.scene.add(key);
    const fill = new THREE.DirectionalLight(0xa8c0e8, 0.45);
    fill.position.set(-700, 500, 500);
    this.scene.add(fill);

    // Ground: shadow catcher + faint grid, repositioned under each model.
    this.ground = new THREE.Mesh(
      new THREE.PlaneGeometry(1, 1),
      new THREE.ShadowMaterial({ opacity: 0.32 })
    );
    this.ground.receiveShadow = true;
    this.scene.add(this.ground);
    this.grid = null;

    this.raycaster = new THREE.Raycaster();
    this.loader = new GLTFLoader();

    this._downXY = null;
    canvas.addEventListener('pointerdown', (e) => { this._downXY = [e.clientX, e.clientY]; });
    canvas.addEventListener('pointerup', (e) => this._maybePick(e));
    window.addEventListener('resize', () => this._resize());
    this._resize();

    const tick = () => {
      requestAnimationFrame(tick);
      this.controls.update();
      this.renderer.render(this.scene, this.camera);
    };
    tick();
  }

  _resize() {
    const w = this.canvas.clientWidth || this.canvas.parentElement.clientWidth;
    const h = this.canvas.clientHeight || this.canvas.parentElement.clientHeight;
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
  }

  // ---- loading -----------------------------------------------------------

  async loadGLB(url) {
    const gltf = await this.loader.loadAsync(url);
    const group = gltf.scene;
    group.traverse((obj) => {
      if (!obj.isMesh) return;
      // Node names 'part_N' come from the backend; find the owning part.
      let p = obj; let partId = null;
      while (p) {
        const m = /^part_(\d+)$/.exec(p.name || '');
        if (m) { partId = parseInt(m[1], 10); break; }
        p = p.parent;
      }
      obj.userData.partId = partId;
      const cls = this.partClass[partId] || 'pillow';
      obj.material = new THREE.MeshStandardMaterial({
        color: cls === 'pillow' ? PART_COLOR : HW_COLOR,
        roughness: cls === 'pillow' ? 0.62 : 0.35,
        metalness: cls === 'pillow' ? 0.0 : 0.55,
      });
      obj.castShadow = true;
      obj.receiveShadow = false;
      if (!obj.geometry.attributes.normal) obj.geometry.computeVertexNormals();
    });
    return group;
  }

  setPartClasses(map) { this.partClass = map; }

  async setBefore(url) {
    if (this.beforeGroup) this._dispose(this.beforeGroup);
    this.beforeGroup = url ? await this.loadGLB(url) : null;
    if (this.beforeGroup) this.scene.add(this.beforeGroup);
    this._applyMorph();
    this._applySelection();
  }

  async setAfter(url) {
    const old = this.afterGroup;
    this.afterGroup = url ? await this.loadGLB(url) : null;
    if (this.afterGroup) this.scene.add(this.afterGroup);
    if (old) this._dispose(old);   // swap after load: no flicker
    this._applyMorph();
    this._applySelection();
    if (this.onAfterLoaded) this.onAfterLoaded();
  }

  /** All meshes of one part in the after (pillowed) group. */
  partMeshes(partId) {
    const out = [];
    if (this.afterGroup) {
      this.afterGroup.traverse((o) => {
        if (o.isMesh && o.userData.partId === partId) out.push(o);
      });
    }
    return out;
  }

  /** Raycast the after-group at client coords, optionally one part only. */
  raycastAt(clientX, clientY, partId = null) {
    if (!this.afterGroup) return null;
    const rect = this.canvas.getBoundingClientRect();
    const ndc = new THREE.Vector2(
      ((clientX - rect.left) / rect.width) * 2 - 1,
      -((clientY - rect.top) / rect.height) * 2 + 1
    );
    this.raycaster.setFromCamera(ndc, this.camera);
    const hits = this.raycaster.intersectObject(this.afterGroup, true);
    const hit = hits.find(
      (h) => h.object.isMesh && (partId === null || h.object.userData.partId === partId)
    );
    return hit || null;
  }

  setRotateEnabled(v) { this.controls.enableRotate = v; }

  clear() {
    if (this.beforeGroup) this._dispose(this.beforeGroup);
    if (this.afterGroup) this._dispose(this.afterGroup);
    this.beforeGroup = this.afterGroup = null;
    this.selectedPartId = null;
  }

  _dispose(group) {
    this.scene.remove(group);
    group.traverse((o) => {
      if (o.isMesh) { o.geometry.dispose(); o.material.dispose(); }
    });
  }

  // ---- before / after morph ----------------------------------------------

  setMorph(t) {
    this.morphT = t;
    this._applyMorph();
  }

  _applyMorph() {
    const t = this.morphT;
    const setOpacity = (group, vis, op) => {
      if (!group) return;
      group.visible = vis;
      group.traverse((o) => {
        if (!o.isMesh) return;
        o.material.transparent = op < 1;
        o.material.opacity = op;
        o.material.depthWrite = op > 0.5;
      });
    };
    setOpacity(this.afterGroup, t > 0.02, t);
    setOpacity(this.beforeGroup, t < 0.98, 1 - t);
  }

  // ---- selection -----------------------------------------------------------

  select(partId) {
    this.selectedPartId = partId;
    this._applySelection();
  }

  _applySelection() {
    for (const group of [this.beforeGroup, this.afterGroup]) {
      if (!group) continue;
      group.traverse((o) => {
        if (!o.isMesh) return;
        const sel = o.userData.partId === this.selectedPartId && this.selectedPartId !== null;
        o.material.emissive.setHex(sel ? SELECT_EMISSIVE : 0x000000);
        o.material.emissiveIntensity = sel ? 0.55 : 0;
      });
    }
  }

  _maybePick(e) {
    // Ignore drags (orbiting) — only a clean click selects.
    if (!this.pickEnabled) { this._downXY = null; return; }
    if (!this._downXY) return;
    const dx = e.clientX - this._downXY[0];
    const dy = e.clientY - this._downXY[1];
    this._downXY = null;
    if (dx * dx + dy * dy > 25) return;

    const rect = this.canvas.getBoundingClientRect();
    const ndc = new THREE.Vector2(
      ((e.clientX - rect.left) / rect.width) * 2 - 1,
      -((e.clientY - rect.top) / rect.height) * 2 + 1
    );
    this.raycaster.setFromCamera(ndc, this.camera);
    const active = this.morphT >= 0.5 ? this.afterGroup : this.beforeGroup;
    if (!active) return;
    const hits = this.raycaster.intersectObject(active, true);
    const hit = hits.find((h) => h.object.isMesh);
    const partId = hit ? hit.object.userData.partId : null;
    if (this.onSelect) this.onSelect(partId, hit || null);
  }

  // ---- camera --------------------------------------------------------------

  fit() {
    const group = this.afterGroup || this.beforeGroup;
    if (!group) return;
    const box = new THREE.Box3().setFromObject(group);
    const size = box.getSize(new THREE.Vector3());
    const center = box.getCenter(new THREE.Vector3());
    const radius = Math.max(size.x, size.y, size.z) * 0.75 || 500;

    this.controls.target.copy(center);
    this.camera.position.set(
      center.x + radius * 0.9,
      center.y - radius * 1.4,
      center.z + radius * 1.1
    );
    this.camera.near = radius / 100;
    this.camera.far = radius * 40;
    this.camera.updateProjectionMatrix();

    // Ground + grid sit just under the model, sized to it.
    const gsize = Math.max(size.x, size.y) * 4 || 2000;
    this.ground.geometry.dispose();
    this.ground.geometry = new THREE.PlaneGeometry(gsize, gsize);
    this.ground.position.set(center.x, center.y, box.min.z - 0.5);
    if (this.grid) { this.scene.remove(this.grid); this.grid.geometry.dispose(); }
    this.grid = new THREE.GridHelper(gsize, 40, 0x2a2e37, 0x20242c);
    this.grid.rotation.x = Math.PI / 2;   // GridHelper is y-up; rotate to z-up
    this.grid.position.copy(this.ground.position);
    this.grid.position.z -= 0.5;
    this.scene.add(this.grid);

    this.keyLight.position.set(center.x + radius, center.y - radius * 0.6, center.z + radius * 1.6);
    this.keyLight.target.position.copy(center);
    this.keyLight.target.updateMatrixWorld();
    const s = this.keyLight.shadow.camera;
    s.left = s.bottom = -radius * 2.2;
    s.right = s.top = radius * 2.2;
    s.far = radius * 8;
    s.updateProjectionMatrix();
  }
}
