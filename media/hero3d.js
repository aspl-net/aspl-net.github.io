// Hero scene: a prediction f(x) outside a convex polytope C(x) is projected onto it.
// Dykstra's algorithm (an exact Euclidean-projection solver) zig-zags onto the tight face,
// so y = Pi_C(f) and f - y lies on the face's outward normal. A quadratic loss on y is then
// drawn as level sets on the face; its descent direction -grad L splits into an in-face part,
// which the layer passes back, and a normal part, which it blocks. Training then moves y
// (and f with it) down the loss along the face. Cycles over faces; drag to rotate.
// Debug: ?t=8.2 freezes the timeline, ?seed=N fixes the random choices, ?debug runs the numeric checks.
(function () {
  const host = document.getElementById("hero3d");
  if (!host || !window.THREE) return;
  const qs = new URLSearchParams(location.search);
  const tq = qs.has("t") ? parseFloat(qs.get("t")) : NaN;
  let FIXT = Number.isFinite(tq) ? Math.min(Math.max(tq, 0), 240) : null;
  const PERF = qs.has("perf");
  const DEBUG = FIXT !== null || PERF || qs.has("debug");
  const dbg = window.__heroDebug = {};
  let seed = qs.has("seed") ? (parseInt(qs.get("seed"), 10) >>> 0) || 1 : null;
  const rand = seed === null ? Math.random : () => { seed = (seed * 1664525 + 1013904223) >>> 0; return seed / 4294967296; };
  // no WebGL at all: keep the SVG fallback and stay quiet
  { const c = document.createElement("canvas"); let ok = false; try { const gl = window.WebGLRenderingContext && (c.getContext("webgl2") || c.getContext("webgl")); ok = !!gl; if (gl) { const x = gl.getExtension("WEBGL_lose_context"); if (x) x.loseContext(); } } catch (e) {} if (!ok) return; }
  let renderer;
  try {
    renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, preserveDrawingBuffer: FIXT !== null });
  } catch (e) { return; }
  host.classList.add("has-3d");
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.setClearColor(0x000000, 0);
  host.appendChild(renderer.domElement);

  const V = (x, y, z) => new THREE.Vector3(x, y, z);
  const BLUE = 0x2563eb, LBLUE = 0x93c5fd, NAVY = 0x1e3a8a, ORANGE = 0xea580c;
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(32, 1, 0.1, 50);
  camera.position.set(0, 0.8, 5.45);
  camera.lookAt(0, -0.08, 0);
  // soft studio environment (a gradient dome with two light panels) for glassy reflections;
  // rebuilt after a lost WebGL context, since the prefiltered map lives only on the GPU
  const envScene = (() => {
    const env = new THREE.Scene();
    const dome = new THREE.SphereGeometry(10, 32, 16);
    const col = [];
    for (let i = 0; i < dome.attributes.position.count; i++) {
      const y = dome.attributes.position.getY(i) / 10;
      const c = new THREE.Color(0xeaf2fd).lerp(new THREE.Color(0x7aa6e6), Math.max(0, -y) * 0.9).lerp(new THREE.Color(0xffffff), Math.max(0, y));
      col.push(c.r, c.g, c.b);
    }
    dome.setAttribute("color", new THREE.Float32BufferAttribute(col, 3));
    env.add(new THREE.Mesh(dome, new THREE.MeshBasicMaterial({ vertexColors: true, side: THREE.BackSide })));
    [[V(4, 6, 5), 5], [V(-6, 2, -3), 3]].forEach(([p, s]) => {
      const m = new THREE.Mesh(new THREE.PlaneGeometry(s, s), new THREE.MeshBasicMaterial({ color: 0xdce7f7, side: THREE.DoubleSide }));
      m.position.copy(p); m.lookAt(0, 0, 0); env.add(m);
    });
    return env;
  })();
  let envRT = null;
  function buildEnv(fresh) {
    // after a lost context the old map's GL objects are gone: drop it rather than dispose it
    const pm = new THREE.PMREMGenerator(renderer);
    if (envRT && !fresh) envRT.dispose();
    envRT = pm.fromScene(envScene, 0.02);
    scene.environment = envRT.texture;
    pm.dispose();
    dbg.envBuilds = (dbg.envBuilds || 0) + 1;
  }
  buildEnv();
  scene.fog = new THREE.Fog(0xdde9fb, 4.7, 8.6); // far edges and glass fade into the page: depth cue
  scene.add(new THREE.HemisphereLight(0xffffff, 0xbcd3f2, 0.55));
  const sun = new THREE.DirectionalLight(0xffffff, 0.75);
  sun.position.set(3, 5, 4);
  scene.add(sun);
  const rim = new THREE.DirectionalLight(0xbfdbfe, 0.7);
  rim.position.set(-4, 2, -5);
  scene.add(rim);

  const world = new THREE.Group();
  scene.add(world);

  // ---------- feasible set: a bevelled irregular dodecahedron, built from its half-spaces ----------
  // The bevels sit about 32 degrees from the faces they join, so an exact projection solver
  // visibly zig-zags between a face and its bevel before it settles.
  const geo = (() => {
    const base = new THREE.DodecahedronGeometry(1.15, 0);
    base.scale(1.25, 0.95, 1.05);
    base.rotateZ(0.18);
    const bp = base.attributes.position, planes = [];
    for (let i = 0; i < bp.count; i += 3) {
      const a = V().fromBufferAttribute(bp, i), b = V().fromBufferAttribute(bp, i + 1), c = V().fromBufferAttribute(bp, i + 2);
      const n = V().subVectors(b, a).cross(V().subVectors(c, a)).normalize();
      if (n.dot(a) < 0) n.negate();
      if (!planes.some(P => P.n.dot(n) > 0.999)) planes.push({ n, d: n.dot(a), pts: [a, b, c] });
      else planes.find(P => P.n.dot(n) > 0.999).pts.push(a, b, c);
    }
    const BEVEL = 0.06, all = planes.map(P => ({ n: P.n, d: P.d }));
    for (let i = 0; i < planes.length; i++) for (let j = i + 1; j < planes.length; j++) {
      const sh = planes[i].pts.filter(p => Math.abs(planes[j].n.dot(p) - planes[j].d) < 1e-5);
      if (sh.length < 2) continue;
      const m = V().addVectors(planes[i].n, planes[j].n).normalize();
      all.push({ n: m, d: m.dot(sh[0]) - BEVEL });
    }
    // vertices = feasible intersections of three planes
    const vs = [];
    for (let i = 0; i < all.length; i++) for (let j = i + 1; j < all.length; j++) for (let k = j + 1; k < all.length; k++) {
      const M = new THREE.Matrix3().set(all[i].n.x, all[i].n.y, all[i].n.z, all[j].n.x, all[j].n.y, all[j].n.z, all[k].n.x, all[k].n.y, all[k].n.z);
      if (Math.abs(M.determinant()) < 1e-6) continue;
      const x = V(all[i].d, all[j].d, all[k].d).applyMatrix3(M.invert());
      if (all.every(P => P.n.dot(x) <= P.d + 1e-7) && !vs.some(v => v.distanceTo(x) < 1e-6)) vs.push(x);
    }
    const tri = [], polys = [];
    all.forEach(P => {
      const on = vs.filter(v => Math.abs(P.n.dot(v) - P.d) < 1e-6);
      if (on.length < 3) return;
      const c = on.reduce((s, v) => s.add(v), V()).multiplyScalar(1 / on.length);
      const t1 = V().subVectors(on[0], c).normalize(), t2 = V().crossVectors(P.n, t1);
      on.sort((p, q) => Math.atan2(V().subVectors(p, c).dot(t2), V().subVectors(p, c).dot(t1)) - Math.atan2(V().subVectors(q, c).dot(t2), V().subVectors(q, c).dot(t1)));
      for (let i = 0; i < on.length; i++) tri.push(c, on[i], on[(i + 1) % on.length]);
      polys.push({ n: P.n, d: P.d, verts: on });
    });
    const g = new THREE.BufferGeometry().setFromPoints(tri);
    g.computeVertexNormals();
    g.userData.polys = polys;
    return g;
  })();
  const glass = { color: 0x7aaaf0, roughness: 0.2, metalness: 0, clearcoat: 0.85, clearcoatRoughness: 0.08,
    envMapIntensity: 0.7, flatShading: true, transparent: true, depthWrite: false };
  const inner = new THREE.Mesh(geo, new THREE.MeshPhysicalMaterial(Object.assign({}, glass, { side: THREE.BackSide, opacity: 0.3, color: 0x5b90e0 })));
  const shell = new THREE.Mesh(geo, new THREE.MeshPhysicalMaterial(Object.assign({}, glass, { side: THREE.FrontSide, opacity: 0.28 })));
  // fresnel rim: glass brightens where it turns away from the viewer
  const rimMesh = new THREE.Mesh(geo, new THREE.ShaderMaterial({
    uniforms: { k: { value: 0.5 } },
    vertexShader: "varying vec3 vN; varying vec3 vV; void main(){ vec4 mv = modelViewMatrix * vec4(position, 1.0); vN = normalize(normalMatrix * normal); vV = normalize(-mv.xyz); gl_Position = projectionMatrix * mv; }",
    fragmentShader: "uniform float k; varying vec3 vN; varying vec3 vV; void main(){ float f = pow(1.0 - abs(dot(normalize(vN), normalize(vV))), 2.4); gl_FragColor = vec4(1.0, 1.0, 1.0, k * f); }",
    transparent: true, depthWrite: false, side: THREE.FrontSide
  }));
  inner.renderOrder = 0; shell.renderOrder = 2; rimMesh.renderOrder = 2.2;
  world.add(inner, shell, rimMesh);
  // one shared unit cylinder, scaled per rod, so per-cycle rods allocate no GPU buffers
  const rodGeo = new THREE.CylinderGeometry(1, 1, 1, 8);
  function rodMesh(a, b, r, m) {
    const d = V().subVectors(b, a), L = d.length();
    const o = new THREE.Mesh(rodGeo, m);
    o.position.copy(a).addScaledVector(d, 0.5);
    o.quaternion.setFromUnitVectors(V(0, 1, 0), d.normalize());
    o.scale.set(r, L, r);
    return o;
  }
  {
    // edges of the twelve main faces read lightly; the bevel seams stay fainter
    const eg = new THREE.EdgesGeometry(geo, 1).attributes.position;
    const em = new THREE.MeshBasicMaterial({ color: 0x1e40af, transparent: true, opacity: 0.42 });
    const emSoft = new THREE.MeshBasicMaterial({ color: 0x1e40af, transparent: true, opacity: 0.2 });
    const polys = geo.userData.polys;
    const area = P => P.verts.reduce((s, v, i) => s + V().subVectors(v, P.verts[0]).cross(V().subVectors(P.verts[(i + 1) % P.verts.length], P.verts[0])).length() / 2, 0);
    polys.forEach(P => { P.area = area(P); });
    const aMax = Math.max(...polys.map(P => P.area));
    for (let i = 0; i < eg.count; i += 2) {
      const a = V().fromBufferAttribute(eg, i), b = V().fromBufferAttribute(eg, i + 1), d = V().subVectors(b, a);
      const main = polys.some(P => P.area > 0.5 * aMax && Math.abs(P.n.dot(a) - P.d) < 1e-6 && Math.abs(P.n.dot(b) - P.d) < 1e-6);
      const r = rodMesh(a, b, main ? 0.0046 : 0.0034, main ? em : emSoft);
      r.renderOrder = 3;
      world.add(r);
    }
  }
  // soft contact shadow on an invisible floor
  let shadowTex = null;
  {
    const cv = document.createElement("canvas"); cv.width = cv.height = 128;
    const cx = cv.getContext("2d"), gr = cx.createRadialGradient(64, 64, 0, 64, 64, 64);
    gr.addColorStop(0, "rgba(30,58,138,0.55)"); gr.addColorStop(0.55, "rgba(30,58,138,0.16)"); gr.addColorStop(1, "rgba(30,58,138,0)");
    cx.fillStyle = gr; cx.fillRect(0, 0, 128, 128);
    shadowTex = new THREE.CanvasTexture(cv);
    const sh = new THREE.Mesh(new THREE.PlaneGeometry(3.4, 3.4), new THREE.MeshBasicMaterial({ map: shadowTex, transparent: true, depthWrite: false }));
    sh.rotation.x = -Math.PI / 2; sh.position.set(-0.32, -1.3, 0); sh.scale.set(0.9, 0.62, 1);
    scene.add(sh);
  }

  // planar faces as half-spaces n.x <= d with outward unit normal n
  const pos = geo.attributes.position;
  const allVerts = []; geo.userData.polys.forEach(P => P.verts.forEach(v => { if (!allVerts.some(w => w.distanceTo(v) < 1e-6)) allVerts.push(v.clone()); }));
  const faces = geo.userData.polys.map(P => ({ n: P.n.clone(), d: P.d, verts: P.verts.map(v => v.clone()) }));
  function distToEdge(F, a, b, p) {
    // in-plane distance from p to edge line ab, positive towards the face interior
    const e = V().subVectors(b, a).normalize();
    const inward = V().crossVectors(F.n, e).normalize();
    const s = V().subVectors(F.c, a).dot(inward) >= 0 ? 1 : -1;
    return s * V().subVectors(p, a).dot(inward);
  }
  function insideMargin(F, p) {
    return Math.min(...F.verts.map((v, i) => distToEdge(F, v, F.verts[(i + 1) % F.verts.length], p)));
  }
  faces.forEach(F => {
    F.c = F.verts.reduce((s, v) => s.add(v), V()).multiplyScalar(1 / F.verts.length);
    F.c.addScaledVector(F.n, F.d - F.n.dot(F.c));
    F.t1 = V().subVectors(F.verts[0], F.c);
    F.t1.addScaledVector(F.n, -F.t1.dot(F.n)).normalize();
    F.t2 = V().crossVectors(F.n, F.t1).normalize();
    const ang = p => Math.atan2(V().subVectors(p, F.c).dot(F.t2), V().subVectors(p, F.c).dot(F.t1));
    F.verts.sort((p, q) => ang(p) - ang(q));
    F.r = insideMargin(F, F.c);
  });
  { const rMax = Math.max(...faces.map(G => G.r)); faces.forEach(G => { G.big = G.r > 0.6 * rMax; }); }
  // Dykstra's algorithm onto the intersection of all half-spaces = exact Euclidean projection
  function dykstra(x0, order, cycles, record) {
    let x = x0.clone();
    const P = faces.map(() => V());
    const path = [x.clone()];
    for (let k = 0; k < cycles; k++) {
      for (const i of order) {
        const H = faces[i];
        const z = x.clone().add(P[i]);
        const v = H.n.dot(z) - H.d;
        const xn = v > 0 ? z.clone().addScaledVector(H.n, -v) : z.clone();
        P[i].copy(z).sub(xn);
        if (record && xn.distanceTo(x) > 1e-3) path.push(xn.clone());
        x = xn;
      }
    }
    return record ? path : x;
  }


  // ---------- helpers ----------
  const fadeMats = [];
  function mat(color, extra) {
    const m = new THREE.MeshPhysicalMaterial(Object.assign({ color, transparent: true, opacity: 0, roughness: 0.35, clearcoat: 0.6, clearcoatRoughness: 0.2, envMapIntensity: 0.3 }, extra || {}));
    fadeMats.push(m);
    return m;
  }
  function arrow(color, r, hr) {
    const m = mat(color, { emissive: color, emissiveIntensity: 0.4 });
    const g = new THREE.Group();
    const shaft = new THREE.Mesh(new THREE.CylinderGeometry(r || 0.018, r || 0.018, 1, 10), m);
    const head = new THREE.Mesh(new THREE.ConeGeometry(hr || 0.055, 0.14, 14), m);
    g.add(shaft, head);
    g.userData = { shaft, head, m };
    world.add(g);
    return g;
  }
  function placeArrow(g, origin, dir, len) {
    const L = Math.max(len, 1e-3);
    g.position.copy(origin);
    g.quaternion.setFromUnitVectors(V(0, 1, 0), dir.clone().normalize());
    const s = Math.min(1, L / 0.14);
    const sl = Math.max(L - 0.12 * s, 1e-3);
    g.userData.shaft.scale.set(1, sl, 1);
    g.userData.shaft.position.set(0, sl / 2, 0);
    g.userData.head.scale.setScalar(s);
    g.userData.head.position.set(0, L - 0.07 * s, 0);
  }

  // ---------- per-cycle objects ----------
  const faceMat = new THREE.MeshBasicMaterial({ color: 0xfbbf8a, transparent: true, opacity: 0, side: THREE.DoubleSide, depthWrite: false });
  const edgeMat = mat(ORANGE, { emissive: 0xc2410c, emissiveIntensity: 0.55 });
  let faceMesh = null, faceEdges = null;
  const fSphere = new THREE.Mesh(new THREE.SphereGeometry(0.075, 24, 16), mat(NAVY, { emissive: 0x1e3a8a, emissiveIntensity: 0.3 }));
  const ySphere = new THREE.Mesh(new THREE.SphereGeometry(0.07, 24, 16), mat(ORANGE, { emissive: 0xea580c, emissiveIntensity: 0.45 }));
  const glow = new THREE.Mesh(new THREE.SphereGeometry(0.16, 24, 16), new THREE.MeshBasicMaterial({ color: ORANGE, transparent: true, opacity: 0, depthWrite: false }));
  world.add(fSphere, ySphere, glow);
  // dashed segments (residual f - y, and the sides of the gradient split)
  const dashGeo = new THREE.CylinderGeometry(0.008, 0.008, 1, 6); dashGeo.translate(0, 0.5, 0);
  function dashPool(n, color) {
    const m = new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0, depthWrite: false });
    const list = [];
    for (let k = 0; k < n; k++) { const d = new THREE.Mesh(dashGeo, m); d.renderOrder = 4; d.visible = false; list.push(d); world.add(d); }
    return { m, list };
  }
  function dashLine(pool, from, a, b, DL, GAP, skip0) {
    const d = V().subVectors(b, a), L = d.length(); d.normalize();
    let k = from;
    for (let s = skip0 || 0; s + DL <= L + 1e-6 && k < pool.list.length; s += DL + GAP, k++) {
      const m = pool.list[k];
      m.visible = true; m.position.copy(a).addScaledVector(d, s);
      m.quaternion.setFromUnitVectors(V(0, 1, 0), d); m.scale.set(1, DL, 1);
    }
    return k;
  }
  const resDash = dashPool(18, NAVY);
  // solver path: one navy tube per Dykstra step, revealed in order, a dot at each iterate,
  // and a bead running along it
  const itMat = new THREE.MeshBasicMaterial({ color: NAVY, transparent: true, opacity: 0, depthWrite: false });
  const dotMat = new THREE.MeshBasicMaterial({ color: NAVY, transparent: true, opacity: 0, depthWrite: false });
  const MAXIT = 16;
  const segGeo = new THREE.CylinderGeometry(0.017, 0.017, 1, 8); segGeo.translate(0, 0.5, 0);
  const dotGeo = new THREE.SphereGeometry(0.027, 14, 10);
  const segs = [], dots = [];
  for (let k = 0; k < MAXIT; k++) {
    const sg = new THREE.Mesh(segGeo, itMat); sg.renderOrder = 4; segs.push(sg); world.add(sg);
    const d = new THREE.Mesh(dotGeo, dotMat); d.renderOrder = 4; dots.push(d); world.add(d);
  }
  const bead = new THREE.Mesh(new THREE.SphereGeometry(0.04, 18, 12), new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0 }));
  bead.renderOrder = 5; world.add(bead);
  // loss on the face: level sets and the minimizer
  const contourMat = new THREE.LineBasicMaterial({ color: NAVY, transparent: true, opacity: 0, depthWrite: false });
  let contours = null;
  const ringMat = new THREE.MeshBasicMaterial({ color: NAVY, transparent: true, opacity: 0, depthWrite: false });
  const ring = new THREE.Mesh(new THREE.TorusGeometry(0.05, 0.009, 8, 40), ringMat);
  const ringDot = new THREE.Mesh(new THREE.SphereGeometry(0.02, 12, 8), ringMat);
  ring.renderOrder = ringDot.renderOrder = 4; world.add(ring, ringDot);
  // descent direction -grad L at y and its split into the passed (in-face) and blocked (normal) parts
  const SLATE = 0x334155, RED = 0xdc2626;
  const gArr = arrow(SLATE), passArr = arrow(0x1d4ed8, 0.028, 0.07), blockArr = arrow(RED);
  // a red cross on the removed part, always facing the viewer
  const crossMat = new THREE.MeshBasicMaterial({ color: RED, transparent: true, opacity: 0, depthWrite: false, depthTest: false });
  const cross = new THREE.Group();
  [Math.PI / 4, -Math.PI / 4].forEach(a => {
    const b = new THREE.Mesh(rodGeo, crossMat); b.scale.set(0.014, 0.17, 0.014); b.rotation.z = a; cross.add(b);
  });
  cross.renderOrder = 6; cross.children.forEach(c => { c.renderOrder = 6; });
  world.add(cross);
  const qInv = new THREE.Quaternion();
  // gradient-descent trail on the face, older steps fade
  const NGD = 14;
  const trailSegs = [], trailDots = [];
  const trailGeo = new THREE.CylinderGeometry(0.017, 0.017, 1, 8); trailGeo.translate(0, 0.5, 0);
  for (let k = 0; k < NGD; k++) {
    const m = new THREE.MeshBasicMaterial({ color: 0xc2410c, transparent: true, opacity: 0, depthWrite: false });
    const sg = new THREE.Mesh(trailGeo, m); sg.renderOrder = 4; trailSegs.push(sg); world.add(sg);
    const d = new THREE.Mesh(new THREE.SphereGeometry(0.026, 12, 8), m); d.renderOrder = 4; trailDots.push(d); world.add(d);
  }

  // loss L(y) = 1/2 (y - t)^T A (y - t), A = A1 u u^T + A2 w w^T + A3 n n^T, t = ys + tn n.
  // u, w span the face, so on the face L is an elliptic bowl centred at ys; t lies outside C,
  // so -grad L also points out of the face, and that part is what the layer blocks.
  const A1 = 1, A2 = 3.2, A3 = 1, ETA = 0.21;
  let F, y0, h, its, faceLabelAt, freeB, loss = null;
  function gradL(y) {
    const d = V().subVectors(y, loss.t);
    return V().addScaledVector(loss.u, A1 * loss.u.dot(d)).addScaledVector(loss.w, A2 * loss.w.dot(d)).addScaledVector(F.n, A3 * F.n.dot(d));
  }
  function lossVal(y) {
    const d = V().subVectors(y, loss.t);
    return 0.5 * (A1 * loss.u.dot(d) ** 2 + A2 * loss.w.dot(d) ** 2 + A3 * F.n.dot(d) ** 2);
  }
  const projT = g => g.clone().addScaledVector(F.n, -g.dot(F.n));
  function gdPath() {
    // the training step through the layer: y <- y - eta * P_T grad L(y)  (P_T = Jacobian of the projection)
    const P = [y0.clone()];
    for (let k = 0; k < NGD; k++) P.push(P[k].clone().addScaledVector(projT(gradL(P[k])), -ETA));
    return P;
  }
  const labels = {};
  host.querySelectorAll("[data-label]").forEach(el => { labels[el.dataset.label] = el; });
  const leader = document.createElement("i");
  leader.className = "h3d-leader"; leader.setAttribute("aria-hidden", "true");
  host.appendChild(leader);

  // the chosen face is turned so its outward normal points here (world frame):
  // toward the camera, to the right and slightly up, so f(x) floats in open space.
  const NSTAR = V(0.7, 0.2, 0.69).normalize();
  let lastFace = -1;
  function pickFace(q) {
    const cand = [];
    faces.forEach((G, i) => {
      if (i === lastFace || !G.big) return;
      const c = G.n.clone().applyQuaternion(q).dot(NSTAR);
      if (c > 0.05 && c < 0.8) cand.push(i);
    });
    const pool = cand.length ? cand : faces.map((_, i) => i).filter(i => i !== lastFace && faces[i].big);
    lastFace = pool[Math.floor(rand() * pool.length)];
    return lastFace;
  }
  const sideMemo = {};
  function newCycle(q0) {
    const fi = pickFace(q0);
    F = faces[fi];
    const q = new THREE.Quaternion().setFromUnitVectors(F.n.clone().applyQuaternion(q0), NSTAR).multiply(q0);
    if (faceMesh) { world.remove(faceMesh); faceMesh.geometry.dispose(); }
    if (faceEdges) world.remove(faceEdges);
    const lift = F.n.clone().multiplyScalar(0.004);
    const pts = [F.c.clone().add(lift)].concat(F.verts.map(v => v.clone().add(lift)));
    const idx = [];
    for (let i = 1; i <= F.verts.length; i++) idx.push(0, i, i % F.verts.length + 1);
    const g = new THREE.BufferGeometry().setFromPoints(pts);
    g.setIndex(idx);
    faceMesh = new THREE.Mesh(g, faceMat);
    faceMesh.renderOrder = 2.5;
    world.add(faceMesh);
    faceEdges = new THREE.Group();
    F.verts.forEach((v, i) => faceEdges.add(rodMesh(v, F.verts[(i + 1) % F.verts.length], 0.02, edgeMat)));
    world.add(faceEdges);
    for (const k in sideMemo) delete sideMemo[k];

    // y0: interior point shifted toward one edge, so f also violates the neighbouring face
    // and the solver has to zig-zag between the two before settling on the tight face.
    const qi = q.clone().invert();
    const upL = V(0, 1, 0).applyQuaternion(qi);
    upL.addScaledVector(F.n, -upL.dot(F.n)).normalize();
    let ei = 0, eb = -1e9;
    F.verts.forEach((v, i) => {
      const m = v.clone().add(F.verts[(i + 1) % F.verts.length]).multiplyScalar(0.5);
      const s = V().subVectors(m, F.c).normalize().dot(upL) + 0.15 * rand();
      if (s > eb) { eb = s; ei = i; }
    });
    const ea = F.verts[ei], ebv = F.verts[(ei + 1) % F.verts.length];
    const mid = ea.clone().add(ebv).multiplyScalar(0.5);
    y0 = F.c.clone().lerp(mid, 0.72);
    y0.addScaledVector(F.n, F.d - F.n.dot(y0));
    const G = faces.reduce((best, H, j) => (j !== fi && Math.abs(H.n.dot(ea) - H.d) < 1e-4 && Math.abs(H.n.dot(ebv) - H.d) < 1e-4) ? H : best, null);
    const gi = faces.indexOf(G);
    const margin = G.d - G.n.dot(y0);
    h = Math.min(Math.max((margin + 0.36) / Math.max(G.n.dot(F.n), 0.2), 0.75), 0.95);
    const fpt = y0.clone().addScaledVector(F.n, h);
    const order = [gi, fi].concat(faces.map((_, j) => j).filter(j => j !== gi && j !== fi));
    its = dykstra(fpt, order, 12, true).slice(0, MAXIT);
    its.push(y0.clone());
    segs.forEach((sg, k) => {
      sg.userData.on = k + 1 < its.length;
      if (!sg.userData.on) return;
      const a = its[k], b = its[k + 1], d = V().subVectors(b, a);
      sg.position.copy(a); sg.quaternion.setFromUnitVectors(V(0, 1, 0), d.clone().normalize()); sg.userData.L = d.length();
    });
    // a dot per iterate, skipping ones that crowd the output or the previous dot
    let lastDot = its[0];
    dots.forEach((d, k) => {
      const p = its[k + 1];
      d.userData.on = k + 1 < its.length - 1 && p.distanceTo(y0) > 0.11 && p.distanceTo(lastDot) > 0.07;
      if (d.userData.on) { d.position.copy(p); lastDot = p; }
    });
    freeB = V().crossVectors(F.n, V().subVectors(F.c, y0).normalize()).normalize();
    if (freeB.clone().applyQuaternion(q).x < 0) freeB.negate();

    // loss: minimizer ys inside the face, bowl axes u, w; keep a candidate whose descent path
    // stays well inside the face, travels far enough to read, and visibly bends.
    const dir = V().subVectors(F.c, y0).normalize(), side = V().crossVectors(F.n, dir);
    let pick = null, fallback = null;
    for (let tries = 0; tries < 160 && !pick; tries++) {
      const rho = (0.2 + 0.4 * rand()) * F.r, ang = (2 * rand() - 1) * 1.1, phi = rand() * Math.PI;
      const ys = F.c.clone().addScaledVector(dir, rho * Math.cos(ang)).addScaledVector(side, rho * Math.sin(ang));
      const u = F.t1.clone().multiplyScalar(Math.cos(phi)).addScaledVector(F.t2, Math.sin(phi)).normalize();
      const w = V().crossVectors(F.n, u).normalize();
      loss = { ys, u, w, t: ys.clone() };
      const gT = projT(gradL(y0)).length();
      loss.t = ys.clone().addScaledVector(F.n, 1.1 * gT / A3);
      const P = gdPath();
      const minM = Math.min(...P.map(p => insideMargin(F, p)));
      const chord = V().subVectors(P[NGD], y0), cl = chord.length();
      let bend = 0;
      P.forEach(p => { const r = V().subVectors(p, y0); bend = Math.max(bend, r.addScaledVector(chord, -r.dot(chord) / (cl * cl)).length()); });
      // on screen the passed direction should point down, so the blocked one lands to the right
      const gw = projT(gradL(y0)).negate().applyQuaternion(q);
      const down = -gw.y / Math.hypot(gw.x, gw.y);
      const c = { ys, u, w, t: loss.t, P, minM, bend, cl, down };
      if (minM > 0.07 && (!fallback || down > fallback.down)) fallback = c;
      if (minM > 0.07 && cl > 0.42 && bend > 0.07 && down > 0.75) pick = c;
    }
    if (!fallback) fallback = { ys: F.c.clone(), u: F.t1.clone(), w: F.t2.clone(), t: F.c.clone().addScaledVector(F.n, 0.3), minM: 0, bend: 0, cl: 0, down: 0 };
    loss = pick || fallback;
    if (!loss.P) loss.P = gdPath();
    loss.gT0 = projT(gradL(y0)).length();
    // the output's direction of travel during training (its label sits across it)
    loss.travel = V().subVectors(loss.P[NGD], loss.P[0]).normalize().multiplyScalar(0.3);
    loss.GS = 0.48 / loss.gT0;
    loss.side = V().crossVectors(F.n, projT(gradL(y0)).normalize()).normalize();               // display scale: the in-face part is drawn 0.48 long
    loss.L = loss.P.map(p => lossVal(p));
    loss.cumT = [0];
    for (let k = 0; k < NGD; k++) loss.cumT.push(loss.cumT[k] + Math.pow(loss.P[k].distanceTo(loss.P[k + 1]), 0.55));
    // level sets of L on the face plane (true ellipses), clipped to the face polygon
    if (contours) { world.remove(contours); contours.geometry.dispose(); }
    const L0 = 0.5 * (A1 * loss.u.dot(V().subVectors(y0, loss.ys)) ** 2 + A2 * loss.w.dot(V().subVectors(y0, loss.ys)) ** 2);
    const cp = [];
    [0.06, 0.25, 0.55, 1.0, 1.6, 2.4].forEach(s => {
      const c = s * L0, ra = Math.sqrt(2 * c / A1), rb = Math.sqrt(2 * c / A2);
      let prev = null;
      for (let k = 0; k <= 160; k++) {
        const th = k / 160 * Math.PI * 2;
        const p = loss.ys.clone().addScaledVector(loss.u, ra * Math.cos(th)).addScaledVector(loss.w, rb * Math.sin(th));
        const ok = insideMargin(F, p) > 0.012;
        const pl = p.addScaledVector(F.n, 0.006);
        if (ok && prev) cp.push(prev, pl);
        prev = ok ? pl : null;
      }
    });
    contours = new THREE.LineSegments(new THREE.BufferGeometry().setFromPoints(cp), contourMat);
    contours.renderOrder = 4;
    world.add(contours);
    ring.position.copy(loss.ys).addScaledVector(F.n, 0.008);
    ring.quaternion.setFromUnitVectors(V(0, 0, 1), F.n);
    ringDot.position.copy(ring.position);
    trailSegs.forEach((sg, k) => {
      const a = loss.P[k], b = loss.P[k + 1], d = V().subVectors(b, a);
      sg.position.copy(a); sg.quaternion.setFromUnitVectors(V(0, 1, 0), d.clone().normalize()); sg.userData.L = d.length();
      trailDots[k].position.copy(a);
    });

    {
      // points the close-up must show: the wall, the estimate, the push and its two parts
      const g0 = gradL(y0).multiplyScalar(-loss.GS), gT0 = projT(g0);
      loss.key = F.verts.map(v => v.clone()).concat([fpt.clone(), y0.clone().add(g0), y0.clone().add(gT0), y0.clone().add(g0.clone().sub(gT0)), loss.ys.clone()]);
    }
    let far = F.verts[0], fd = -1;
    F.verts.forEach(v => { const d = v.distanceTo(fpt); if (d > fd) { fd = d; far = v; } });
    faceLabelAt = far.clone().addScaledVector(V().subVectors(far, F.c).normalize(), 0.12);

    // numerical checks (debug visits only: ?t=, ?perf, ?debug)
    if (!DEBUG) return q;
    const yStar = dykstra(fpt, order, 400, false);
    let projErrPath = 0, passDotN = 0, maxRise = -1e9;
    loss.P.forEach((p, k) => {
      projErrPath = Math.max(projErrPath, dykstra(p.clone().addScaledVector(F.n, h), order, 400, false).distanceTo(p));
      passDotN = Math.max(passDotN, Math.abs(projT(gradL(p)).dot(F.n)));
      if (k) maxRise = Math.max(maxRise, loss.L[k] - loss.L[k - 1]);
    });
    const gN = gradL(y0).dot(F.n);
    Object.assign(dbg, {
      face: fi, h: +h.toFixed(3), iterates: its.length, visibleSteps: its.filter((p, k) => k && p.distanceTo(its[k - 1]) > 0.02).length,
      projErr: yStar.distanceTo(y0), projErrPath, passDotN, gradNormalOutward: gN < 0,
      lossMonotone: maxRise < 0, maxRise, lossDrop: loss.L[NGD] / loss.L[0] ,
      pathMinMargin: loss.minM, pathBend: loss.bend, pathChord: loss.cl, picked: !!pick,
      onPlane: Math.max(...loss.P.map(p => Math.abs(F.n.dot(p) - F.d))),
      minimizerMargin: insideMargin(F, loss.ys), finalDist: loss.P[NGD].distanceTo(loss.ys)
    });
    return q;
  }

  // ---------- timeline (seconds within a cycle): five narrated steps, then a short hold ----------
  // the animation is scripted on its own clock (OLD); the viewer's clock (NEW) gives each step
  // time to read its sentence, and holds the wide shot before the loop restarts.
  const OLD = [0, 3.4, 6.6, 9.8, 13.2, 18.0, 18.8], NEW = [0, 3.6, 7.2, 10.6, 15.6, 20.6, 21.4];
  const CYCLE = NEW[6];
  function warp(t) {
    let k = 0; while (k < 5 && t >= NEW[k + 1]) k++;
    return OLD[k] + (t - NEW[k]) * (OLD[k + 1] - OLD[k]) / (NEW[k + 1] - NEW[k]);
  }
  const STEPS = [
    [NEW[0], NEW[1], "The neural network makes an estimate, outside the allowed region."],
    [NEW[1], NEW[2], "A standard solver zig-zags to the nearest allowed point; ASPL notes which wall it lands on."],
    [NEW[2], NEW[3], "Training pushes the output toward the best allowed answer (the ring), partly across the wall."],
    [NEW[3], NEW[4], "ASPL keeps the part along the wall and removes the part across it, with one small exact solve."],
    [NEW[4], NEW[6], "The output slides along the wall toward the best allowed answer; the estimate moves with it."],
  ];
  const stepsEl = document.getElementById("hero3d-steps");
  const stepText = stepsEl && stepsEl.querySelector(".h3d-step-text");
  const stepDots = stepsEl ? [...stepsEl.querySelectorAll(".h3d-dots i")] : [];
  // two stacked layers: the outgoing sentence fades out while the incoming one fades in
  const layers = stepText ? [0, 1].map(() => { const s = document.createElement("span"); stepText.appendChild(s); return s; }) : [];
  let shownStep = -1, layer = 0;
  const stepHTML = k => `<b>${k + 1}</b>${STEPS[k][2]}`;
  function reserveCaption() {
    // reserve the tallest sentence's height, so the buttons below never move
    if (!stepText || stepsEl.classList.contains("all")) return;
    const s = layers[1 - layer], keep = s.innerHTML, op = s.style.opacity;
    s.style.opacity = 0;
    let hMax = 0;
    STEPS.forEach((_, k) => { s.innerHTML = stepHTML(k); hMax = Math.max(hMax, s.offsetHeight); });
    s.innerHTML = keep; s.style.opacity = op;
    stepText.style.minHeight = hMax + "px";
  }
  function narrate(tc) {
    if (!stepText || stepsEl.classList.contains("all")) return;
    let k = STEPS.findIndex(s => tc >= s[0] && tc < s[1]);
    if (k < 0) k = STEPS.length - 1;
    if (k !== shownStep) {
      shownStep = k;
      layer = 1 - layer;
      layers[layer].innerHTML = stepHTML(k);
      layers[layer].style.opacity = 1;
      layers[1 - layer].style.opacity = 0;
      stepDots.forEach((d, i) => { d.className = i === k ? "on" : i < k ? "done" : ""; });
    }
    dbg.step = k + 1;
  }
  const ease = (a, b, t) => { const u = Math.min(Math.max((t - a) / (b - a), 0), 1); return u * u * (3 - 2 * u); };
  const lin = (a, b, t) => Math.min(Math.max((t - a) / (b - a), 0), 1);
  function setOpacity(m, o) { m.opacity = o; m.visible = o > 0.003; }
  const tmp = V(), tmp2 = V();
  const labelPrev = {};
  const OUTSIDE = { f: 1, blocked: 1, grad: 1 };
  let dtFrame = 0, lastT = null;
  function ptRectDist(px, py, R) {
    const dx = Math.max(R.x0 - px, 0, px - R.x1), dy = Math.max(R.y0 - py, 0, py - R.y1);
    return Math.hypot(dx, dy);
  }
  function segRectDist(ax, ay, bx, by, R) {
    // distance between segment ab and rectangle R (0 if they touch); allocation-free
    if (ptRectDist(ax, ay, R) === 0 || ptRectDist(bx, by, R) === 0) return 0;
    const ex = bx - ax, ey = by - ay, L2 = ex * ex + ey * ey || 1e-9;
    const side = (ux, uy, vx, vy) => {
      const d1 = (vx - ux) * (ay - uy) - (vy - uy) * (ax - ux), d2 = (vx - ux) * (by - uy) - (vy - uy) * (bx - ux);
      const d3 = ex * (uy - ay) - ey * (ux - ax), d4 = ex * (vy - ay) - ey * (vx - ax);
      return (d1 > 0) !== (d2 > 0) && (d3 > 0) !== (d4 > 0);
    };
    if (side(R.x0, R.y0, R.x1, R.y0) || side(R.x1, R.y0, R.x1, R.y1) || side(R.x1, R.y1, R.x0, R.y1) || side(R.x0, R.y1, R.x0, R.y0)) return 0;
    const cd = (cx, cy) => { const t = Math.min(1, Math.max(0, ((cx - ax) * ex + (cy - ay) * ey) / L2)); return Math.hypot(cx - ax - t * ex, cy - ay - t * ey); };
    return Math.min(cd(R.x0, R.y0), cd(R.x1, R.y0), cd(R.x1, R.y1), cd(R.x0, R.y1), ptRectDist(ax, ay, R), ptRectDist(bx, by, R));
  }
  function toScreen(p3, W, H) {
    tmp.copy(p3).applyMatrix4(world.matrixWorld).project(camera);
    return { x: (tmp.x * 0.5 + 0.5) * W, y: (-tmp.y * 0.5 + 0.5) * H };
  }
  function yAt(tc) {
    // y follows the gradient-descent iterates, one eased hop per step; a hop's duration
    // grows with its length, so long early hops do not flash past and late ones settle
    const u = lin(13.4, 16.6, tc) * loss.cumT[NGD];
    let k = 0; while (k < NGD - 1 && loss.cumT[k + 1] <= u) k++;
    const fr = Math.min((u - loss.cumT[k]) / (loss.cumT[k + 1] - loss.cumT[k]), 1);
    return { y: loss.P[k].clone().lerp(loss.P[k + 1], ease(0, 1, fr)), s: k + fr };
  }

  function update(tc, T, tv) {
    narrate(tv);
    const appear = ease(0.3, 0.7, tc), out = 1 - ease(17.3, 17.9, tc);
    const { y: yy, s: gs } = yAt(tc);
    const fpt = yy.clone().addScaledVector(F.n, h);
    fSphere.position.copy(fpt);
    // the estimate steps back while the push is split, then returns in step 5 and moves with
    // the output (it stays y + h n, so its nearest allowed point is always the output)
    const fBack = ease(13.4, 14.0, tc);
    setOpacity(fSphere.material, appear * out * Math.max(1 - ease(6.7, 7.6, tc), 0.6 * fBack));
    // step 2: solver iterates
    const u = lin(3.7, 5.6, tc);
    const drawK = (1 - Math.pow(1 - u, 1.35)) * (its.length - 1);
    segs.forEach((sg, k) => {
      const part = Math.min(Math.max(drawK - k, 0), 1);
      sg.visible = sg.userData.on && part > 0;
      if (sg.visible) sg.scale.set(1, sg.userData.L * part, 1);
    });
    const fadeIt = out * (1 - ease(5.9, 6.6, tc));
    setOpacity(itMat, 0.85 * fadeIt); setOpacity(dotMat, 0.9 * fadeIt);
    dots.forEach((d, k) => { d.visible = d.userData.on && drawK >= k + 1 && fadeIt > 0.003; });
    { const k = Math.min(Math.floor(drawK), its.length - 2), fr = drawK - k;
      bead.position.copy(its[k]).lerp(its[k + 1], Math.min(fr, 1));
      setOpacity(bead.material, u > 0 && u < 1 ? 0.95 * out : 0); }
    // the wall it touches, and the output on it
    const faceOn = ease(5.2, 5.7, tc) * out;
    faceMat.opacity = (0.5 + 0.2 * ease(6.8, 7.6, tc)) * faceOn;
    faceMat.depthWrite = tc > 7.6 && tc < 17.3;
    setOpacity(edgeMat, faceOn);
    ySphere.position.copy(yy);
    glow.position.copy(yy);
    const yOn = ease(5.5, 5.9, tc) * out;
    setOpacity(ySphere.material, yOn);
    glow.material.opacity = 0.22 * yOn * (0.75 + 0.25 * Math.sin(T * 4));
    glow.visible = yOn > 0;
    // straight, shortest line from the guess to the output (along the wall's normal)
    const resOn = out * Math.max(ease(5.8, 6.3, tc) * (1 - ease(6.7, 7.6, tc)), 0.8 * fBack);
    resDash.list.forEach(d => { d.visible = false; });
    if (resOn > 0.01) dashLine(resDash, 0, yy.clone().addScaledVector(F.n, 0.06), fpt.clone().addScaledVector(F.n, -0.07), 0.045, 0.035);
    resDash.m.opacity = 0.55 * resOn;
    // step 3: score rings on the wall around the best point
    const lossOn = ease(6.8, 7.5, tc) * out;
    contourMat.opacity = 0.34 * lossOn; contours.visible = lossOn > 0.01;
    setOpacity(ringMat, 0.9 * lossOn);
    { const k = lin(16.4, 17.2, tc), bump = k > 0 && k < 1 ? Math.sin(Math.PI * k) : 0; ring.scale.setScalar(1 + 0.9 * bump); }
    // the training push at the output, then (step 4) its split into kept and removed parts
    const g = gradL(yy).multiplyScalar(-loss.GS), gT = projT(g), gNv = g.clone().sub(gT);
    const gOn = ease(7.7, 8.3, tc), split = ease(9.9, 10.4, tc), xOn = ease(11.9, 12.2, tc), cut = ease(12.5, 13.0, tc);
    placeArrow(gArr, yy, g, g.length() * gOn + 1e-3);
    setOpacity(gArr.userData.m, Math.min(gOn * 1.5, 1) * (1 - 0.85 * split) * (1 - cut) * out);
    placeArrow(passArr, yy, gT.lengthSq() > 1e-12 ? gT : F.t1, gT.length() * split + 1e-3);
    passArr.userData.m.emissiveIntensity = 0.4 + 0.5 * split * (1 - ease(13.2, 13.8, tc)) * (0.5 + 0.5 * Math.sin(T * 5));
    setOpacity(passArr.userData.m, Math.min(split * 1.5, 1) * out * (1 - ease(16.6, 17.2, tc)));
    placeArrow(blockArr, yy, F.n, gNv.length() * split + 1e-3);
    setOpacity(blockArr.userData.m, Math.min(split * 1.5, 1) * (1 - cut) * out);
    // the removed part gets crossed out, then fades
    cross.position.copy(yy).addScaledVector(F.n, 0.5 * gNv.length());
    qInv.copy(world.quaternion).invert();
    cross.quaternion.copy(qInv).multiply(camera.quaternion);
    cross.scale.setScalar(0.6 + 0.4 * xOn);
    crossMat.opacity = xOn * (1 - cut) * out; cross.visible = crossMat.opacity > 0.003;
    // step 5: descent trail, older hops fade
    trailSegs.forEach((sg, k) => {
      const part = Math.min(Math.max(gs - k, 0), 1);
      const age = gs - (k + 1);
      const o = part > 0 ? 0.9 * Math.max(0.35, 1 - Math.max(age, 0) / 9) * out : 0;
      sg.material.opacity = o; sg.visible = o > 0.003; trailDots[k].visible = o > 0.003 && k > 0;
      if (sg.visible) sg.scale.set(1, sg.userData.L * ease(0, 1, part), 1);
    });

    // ---------- HTML labels: placed among candidate spots, clear of the wall's edges, the arrows and each other ----------
    const W = host.clientWidth, H = host.clientHeight;
    const want = [];
    function put(name, p3, show, dx, dy, prio, more) {
      // more: extra anchor points (screen), tried after the first at a small cost each
      const el = labels[name];
      if (!el) return;
      const a = toScreen(p3, W, H);
      want.push({ name, el, show, ax: a.x, ay: a.y, dx, dy, prio: prio || 0, w: el.offsetWidth + 4, h: el.offsetHeight, more: more || [] });
    }
    {
      // the estimate's label points away from the wall on screen, so it never has to cross the shape
      const a2 = toScreen(fpt, W, H), c2 = toScreen(F.c, W, H);
      let ux = a2.x - c2.x, uy = a2.y - c2.y; const L = Math.hypot(ux, uy) || 1; ux /= L; uy /= L;
      put("f", fpt, out * Math.max(appear * (1 - ease(6.4, 6.9, tc)), fBack * (1 - ease(16.4, 16.9, tc))), 24 * ux, 24 * uy - 6, 1);
    }
    {
      // the output keeps one name and one side for the whole cycle: across its direction of travel
      // (off the kept part and the trail), on the side away from the removed part
      const a2 = toScreen(yy, W, H), b2 = toScreen(tmp2.copy(yy).add(loss.travel), W, H), n2 = toScreen(tmp2.copy(yy).add(F.n), W, H);
      let ux = b2.x - a2.x, uy = b2.y - a2.y; const L = Math.hypot(ux, uy) || 1; ux /= L; uy /= L;
      let px = -uy, py = ux;
      const nx = n2.x - a2.x, ny = n2.y - a2.y, dn = (px * nx + py * ny) / (Math.hypot(nx, ny) || 1);
      if (!sideMemo.y && yOn > 0.5 && Math.abs(dn) > 0.25) sideMemo.y = dn > 0 ? -1 : 1;  // decided once the output is on screen
      const sg = sideMemo.y || (dn > 0 ? -1 : 1);
      // centre distance = the label's half-extent along that direction + a 16 px gap from the dot
      const lw = labels.y.offsetWidth + 4, lh = labels.y.offsetHeight, dd = Math.abs(px) * lw / 2 + Math.abs(py) * lh / 2 + 16;
      put("y", yy, yOn, dd * sg * px, dd * sg * py, 0);
    }
    put("face", faceLabelAt, faceOn * (1 - ease(7.6, 8.1, tc)), 0, 0, 3);
    put("best", loss.ys, lossOn * lin(0.34, 0.52, yy.distanceTo(loss.ys)), 0, 16, 2);
    // arrow labels sit beside the shaft, 65% of the way out: the push on the wall's side, kept and removed outside
    // (the output's own label holds the wall's side for the whole cycle)
    const wallC = toScreen(F.c, W, H);
    function beside(name, v, show, prio, toward) {
      const a2 = toScreen(yy, W, H), b2 = toScreen(yy.clone().add(v), W, H);
      let ux = b2.x - a2.x, uy = b2.y - a2.y; const L = Math.hypot(ux, uy) || 1; ux /= L; uy /= L;
      let px = -uy, py = ux;
      const mx = a2.x + 0.65 * (b2.x - a2.x), my = a2.y + 0.65 * (b2.y - a2.y);
      // the side is decided once per cycle, so the label never swaps sides while the camera moves
      if (!sideMemo[name] && show > 0.04) sideMemo[name] = px * (mx - wallC.x) + py * (my - wallC.y) < 0 ? -1 : 1;
      const sgn = sideMemo[name] || 1;
      px *= toward ? -sgn : sgn; py *= toward ? -sgn : sgn;
      const along = [0.45, 0.85, 0.3, 1.05].map(f => ({ x: a2.x + f * (b2.x - a2.x), y: a2.y + f * (b2.y - a2.y) }));
      put(name, yy.clone().addScaledVector(v, 0.65), show, 12 * px, 12 * py, prio, along);
      want[want.length - 1].near = [a2, b2];
    }
    // anchored to each arrow's full length, so the label does not chase the tip while the arrow grows
    beside("grad", g, gOn * (1 - ease(9.9, 10.3, tc)) * out, 4, true);
    beside("pass", gT, split * out * (1 - ease(13.4, 13.9, tc)) * Math.min(1, Math.max(0, (gT.length() - 0.1) / 0.15)), 5, false);
    beside("blocked", F.n.clone().multiplyScalar(Math.max(gNv.length(), 0.05)), split * (1 - ease(12.5, 12.9, tc)) * out, 6, false);
    put("C", V(0, -0.15, 0), appear * out * (1 - ease(3.0, 3.5, tc)), 0, 0, 7);
    // obstacles on screen: the wall's edges, the arrows, the dashed residual, the points
    const segsS = [], ptsS = [];
    if (faceOn > 0.1) { const wv = F.verts.map(v => toScreen(v, W, H)); wv.forEach((q, i) => segsS.push([q, wv[(i + 1) % wv.length], 6, 320, 1])); }
    const yS = toScreen(yy, W, H);
    // the shape's outline (convex hull of its projected corners): outside labels keep off it
    const hullS = [];
    if (FIXT !== null || want.some(a => OUTSIDE[a.name] && a.show > 0.04)) {
      const P = allVerts.map(v => toScreen(v, W, H)).sort((a, b) => a.x - b.x || a.y - b.y);
      const cr = (o, a, b) => (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x);
      const lo = [], up = [];
      P.forEach(p => { while (lo.length > 1 && cr(lo[lo.length - 2], lo[lo.length - 1], p) <= 0) lo.pop(); lo.push(p); });
      for (let i = P.length - 1; i >= 0; i--) { const p = P[i]; while (up.length > 1 && cr(up[up.length - 2], up[up.length - 1], p) <= 0) up.pop(); up.push(p); }
      const hull = lo.slice(0, -1).concat(up.slice(0, -1));
      hull.forEach((q, i) => hullS.push([q, hull[(i + 1) % hull.length], 5, 90]));
    }
    const addSeg = (p3, on) => { if (on > 0.15) segsS.push([yS, toScreen(p3, W, H), 3]); };
    addSeg(yy.clone().add(g), gArr.userData.m.opacity);
    addSeg(yy.clone().add(gT), passArr.userData.m.opacity);
    // from step 3 on, keep the kept part's path free even before it appears, so the output's label
    // does not settle there and then have to jump away when the push is split
    if (tc > 6.6) segsS.push([yS, toScreen(tmp2.copy(yy).add(gT), W, H), 8, 320]);
    addSeg(yy.clone().addScaledVector(F.n, gNv.length()), blockArr.userData.m.opacity);
    if (resOn > 0.15) segsS.push([yS, toScreen(fpt, W, H), 2]);
    if (itMat.opacity > 0.15) {
      // the solver path and its iterates
      const n = Math.min(Math.ceil(drawK), its.length - 1);
      let prevS = toScreen(its[0], W, H);
      for (let k = 0; k < n; k++) { const s = toScreen(its[k + 1], W, H); segsS.push([prevS, s, 5]); prevS = s; }
    }
    if (gs > 0.05) {
      // the descent trail behind the output
      let prevS = toScreen(loss.P[0], W, H);
      for (let k = 1; k <= Math.min(Math.ceil(gs), NGD); k++) { const s = toScreen(k < gs ? loss.P[k] : yy, W, H); segsS.push([prevS, s, 6]); prevS = s; }
    }
    if (yOn > 0.1) ptsS.push([yS, 14]);
    if (fSphere.material.opacity > 0.1) ptsS.push([toScreen(fpt, W, H), 11]);
    if (lossOn > 0.1) ptsS.push([toScreen(loss.ys, W, H), 8]);
    const vis = want.filter(a => a.show > 0.04).sort((a, b) => a.prio - b.prio);
    const placed = [];
    vis.forEach(a => {
      // preferred direction and gap (anchor to the label's near side); candidates vary both
      const d0 = Math.hypot(a.dx, a.dy), ang0 = d0 > 0 ? Math.atan2(a.dy, a.dx) : -Math.PI / 2;
      const extOf = (ux, uy) => Math.abs(ux) * a.w / 2 + Math.abs(uy) * a.h / 2;
      const gap0 = d0 > 0 ? Math.max(5, d0 - extOf(Math.cos(ang0), Math.sin(ang0))) : 5;
      const prev = labelPrev[a.name];
      let best = null;
      const anchors = [{ x: a.ax, y: a.ay }].concat(a.more);
      for (let ai = 0; ai < anchors.length; ai++) for (let ri = 0; ri < 4; ri++) for (let k = 0; k < 12; k++) {
        const AX = anchors[ai].x, AY = anchors[ai].y;
        const ang = ang0 + (k % 2 ? 1 : -1) * Math.ceil(k / 2) * Math.PI / 6;
        const r = gap0 + 8 * ri;
        const ux = Math.cos(ang), uy = Math.sin(ang), ext = extOf(ux, uy);
        const cx = AX + ux * (r + ext), cy = AY + uy * (r + ext);
        const R = { x0: cx - a.w / 2, x1: cx + a.w / 2, y0: cy - a.h / 2, y1: cy + a.h / 2 };
        let cost = Math.abs(ang - ang0) * 6 + (r - gap0) * 0.8 + ai * 6;
        if (prev && prev.vis) cost += Math.min(Math.hypot(cx - prev.tx, cy - prev.ty) * 1.1, 260);  // stay put unless the spot is really blocked
        const ob = Math.max(0, 2 - R.x0) + Math.max(0, R.x1 - (W - 2)) + Math.max(0, 2 - R.y0) + Math.max(0, R.y1 - (H - 2));
        cost += ob > 0 ? 500 + ob * 40 : 0;
        if (a.near) cost += Math.max(0, segRectDist(a.near[0].x, a.near[0].y, a.near[1].x, a.near[1].y, R) - 6) * 3;  // stay by your arrow
        if (best && cost >= best.cost) continue;
        placed.forEach(q => {
          const ox = Math.min(R.x1, q.x1) - Math.max(R.x0, q.x0) + 3, oy = Math.min(R.y1, q.y1) - Math.max(R.y0, q.y0) + 3;
          if (ox > 0 && oy > 0) cost += 700 + ox * oy * 2;
        });
        for (let i = 0; i < segsS.length && (!best || cost < best.cost); i++) { const sg = segsS[i]; if (sg[4] && a.name === "y") continue; const d = segRectDist(sg[0].x, sg[0].y, sg[1].x, sg[1].y, R); if (d < sg[2]) cost += (sg[3] || 120) + (sg[2] - d) * 20; }
        if (OUTSIDE[a.name]) for (let i = 0; i < hullS.length && (!best || cost < best.cost); i++) { const sg = hullS[i], d = segRectDist(sg[0].x, sg[0].y, sg[1].x, sg[1].y, R); if (d < sg[2]) cost += sg[3] + (sg[2] - d) * 10; }
        ptsS.forEach(([p, m]) => { const d = ptRectDist(p.x, p.y, R); if (d < m) cost += 120 + (m - d) * 20; });
        if (!best || cost < best.cost) best = { cost, cx, cy, R };
      }
      if (DEBUG && (a.name === "y" || a.name === "f")) {
        // why the output's label sits where it does: obstacle costs at the chosen and the preferred spot
        const parts = R => {
          const o = { placed: 0, segs: [], pts: 0 };
          placed.forEach(q => { const ox = Math.min(R.x1, q.x1) - Math.max(R.x0, q.x0) + 3, oy = Math.min(R.y1, q.y1) - Math.max(R.y0, q.y0) + 3; if (ox > 0 && oy > 0) o.placed += 700 + ox * oy * 2; });
          segsS.forEach((sg, i) => { if (sg[4]) return; const d = segRectDist(sg[0].x, sg[0].y, sg[1].x, sg[1].y, R); if (d < sg[2]) o.segs.push([i, sg[2], Math.round((sg[3] || 120) + (sg[2] - d) * 20)]); });
          ptsS.forEach(([p, m]) => { const d = ptRectDist(p.x, p.y, R); if (d < m) o.pts += Math.round(120 + (m - d) * 20); });
          return o;
        };
        const ux0 = Math.cos(ang0), uy0 = Math.sin(ang0), e0 = extOf(ux0, uy0), cx0 = a.ax + ux0 * (gap0 + e0), cy0 = a.ay + uy0 * (gap0 + e0);
        (dbg.why = dbg.why || {})[a.name] = { best: [Math.round(best.cx - a.ax), Math.round(best.cy - a.ay), Math.round(best.cost)], bestParts: parts(best.R),
          pref: [Math.round(cx0 - a.ax), Math.round(cy0 - a.ay)], prefParts: parts({ x0: cx0 - a.w / 2, x1: cx0 + a.w / 2, y0: cy0 - a.h / 2, y1: cy0 + a.h / 2 }), nSegs: segsS.length };
      }
      // last resort: keep the whole label inside the figure
      const ox = Math.min(0, W - 2 - best.R.x1) + Math.max(0, 2 - best.R.x0), oy = Math.min(0, H - 2 - best.R.y1) + Math.max(0, 2 - best.R.y0);
      best.cx += ox; best.cy += oy; best.R.x0 += ox; best.R.x1 += ox; best.R.y0 += oy; best.R.y1 += oy;
      a.tx = best.cx; a.ty = best.cy; placed.push(best.R);
    });
    const snap = FIXT !== null || dtFrame > 0.25;
    want.forEach(a => {
      const prev = labelPrev[a.name] || (labelPrev[a.name] = { x: 0, y: 0, tx: 0, ty: 0, vis: false });
      if (a.show > 0.04) {
        const k = snap || !prev.vis ? 1 : 1 - Math.exp(-dtFrame * (a.name === "y" ? 5 : 10));
        if (PERF && prev.vis && a.show > 0.3 && Math.hypot(a.tx - prev.tx, a.ty - prev.ty) > 30) (dbg.jumps = dbg.jumps || []).push([a.name, +tv.toFixed(2), Math.round(Math.hypot(a.tx - prev.tx, a.ty - prev.ty))]);
        prev.x += (a.tx - prev.x) * k; prev.y += (a.ty - prev.y) * k; prev.tx = a.tx; prev.ty = a.ty; prev.vis = true;
        a.el.style.transform = `translate(${prev.x.toFixed(1)}px, ${prev.y.toFixed(1)}px) translate(-50%, -50%)`;
      } else prev.vis = false;
      a.el.style.opacity = a.show;
    });
    {
      // a short leader line ties the output's label to its dot whenever the label sits apart from it
      const a = want.find(q => q.name === "y"), pv = labelPrev.y;
      let lo = 0;
      if (a && a.show > 0.04 && pv && pv.vis) {
        const R = { x0: pv.x - a.w / 2, x1: pv.x + a.w / 2, y0: pv.y - a.h / 2, y1: pv.y + a.h / 2 };
        const d = ptRectDist(yS.x, yS.y, R);
        if (d > 22) {
          const qx = Math.min(Math.max(yS.x, R.x0), R.x1), qy = Math.min(Math.max(yS.y, R.y0), R.y1);
          const sx = yS.x + (qx - yS.x) * 9 / d, sy = yS.y + (qy - yS.y) * 9 / d, L = Math.hypot(qx - sx, qy - sy);
          leader.style.width = L.toFixed(1) + "px";
          leader.style.transform = `translate(${sx.toFixed(1)}px, ${sy.toFixed(1)}px) rotate(${Math.atan2(qy - sy, qx - sx).toFixed(4)}rad)`;
          lo = Math.min(1, (d - 22) / 8) * a.show;
        }
        dbg.yLabelGap = Math.round(d);
      }
      leader.style.opacity = lo;
    }
    dbg.labelsShown = want.filter(a => a.show > 0.3).map(a => a.name);
    if (DEBUG) dbg.anchors = Object.fromEntries(want.map(q => [q.name, [Math.round(q.ax), Math.round(q.ay)]]));
    if (FIXT !== null) {
      // debug metrics: obliqueness of the wall, its screen polygon, content box, label boxes
      const fcW = F.c.clone().applyMatrix4(world.matrixWorld), nW = F.n.clone().transformDirection(world.matrixWorld);
      dbg.viewCos = V().subVectors(camera.position, fcW).normalize().dot(nW);
      dbg.wall = F.verts.map(v => toScreen(v, W, H));
      {
        const fS = toScreen(fpt, W, H); let inside = true, dmin = Infinity;
        const hp = hullS.map(h => h[0]);
        let area = 0; hp.forEach((q, i) => { const r = hp[(i + 1) % hp.length]; area += q.x * r.y - r.x * q.y; });
        hp.forEach((q, i) => {
          const r = hp[(i + 1) % hp.length], cr = (r.x - q.x) * (fS.y - q.y) - (r.y - q.y) * (fS.x - q.x);
          if ((area > 0 ? cr : -cr) < 0) inside = false;
          const ex = r.x - q.x, ey = r.y - q.y, t = Math.min(1, Math.max(0, ((fS.x - q.x) * ex + (fS.y - q.y) * ey) / (ex * ex + ey * ey)));
          dmin = Math.min(dmin, Math.hypot(fS.x - q.x - t * ex, fS.y - q.y - t * ey));
        });
        dbg.fOut = inside ? -dmin : dmin;
      }
      {
        // the split on screen: angle between kept and removed, and whether removed leaves the wall's outline
        const inPoly = (p, poly) => { let c = false; for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) { const a = poly[i], b = poly[j]; if ((a.y > p.y) !== (b.y > p.y) && p.x < (b.x - a.x) * (p.y - a.y) / (b.y - a.y) + a.x) c = !c; } return c; };
        const kS = toScreen(yy.clone().add(gT), W, H), rS = toScreen(yy.clone().addScaledVector(F.n, gNv.length()), W, H), gS = toScreen(yy.clone().add(g), W, H);
        const ax = kS.x - yS.x, ay = kS.y - yS.y, bx = rS.x - yS.x, by = rS.y - yS.y;
        dbg.splitAngle = Math.acos(Math.max(-1, Math.min(1, (ax * bx + ay * by) / (Math.hypot(ax, ay) * Math.hypot(bx, by) + 1e-9)))) * 180 / Math.PI;
        dbg.removedPx = Math.hypot(bx, by);
        dbg.removedTipOutside = !inPoly(rS, dbg.wall);
        dbg.pushTipOutside = !inPoly(gS, dbg.wall);
        // where the shape is cut by the box: length of each box edge that lies inside the shape's outline
        const hp = hullS.length ? hullS.map(h => h[0]) : null;
        if (hp) {
          const cut = { top: 0, bottom: 0, left: 0, right: 0 };
          for (let x = 0; x <= W; x += 2) { if (inPoly({ x, y: 0.5 }, hp)) cut.top += 2; if (inPoly({ x, y: H - 0.5 }, hp)) cut.bottom += 2; }
          for (let y = 0; y <= H; y += 2) { if (inPoly({ x: 0.5, y }, hp)) cut.left += 2; if (inPoly({ x: W - 0.5, y }, hp)) cut.right += 2; }
          dbg.cut = cut;
        }
      }
      const pts = geo.userData.polys.flatMap(P => P.verts).map(v => toScreen(v, W, H));
      if (fSphere.material.opacity > 0.05) pts.push(toScreen(fpt, W, H));
      pts.push(toScreen(yy, W, H));
      if (gOn > 0.05) pts.push(toScreen(yy.clone().add(g), W, H));
      const inb = pts.filter(q => q.x > -1 && q.x < W + 1 && q.y > -1 && q.y < H + 1);
      dbg.box = { x0: Math.min(...inb.map(q => q.x)), x1: Math.max(...inb.map(q => q.x)), y0: Math.min(...inb.map(q => q.y)), y1: Math.max(...inb.map(q => q.y)), W, H,
        clipped: pts.length - inb.length };
      const hb = host.getBoundingClientRect();
      dbg.labelBoxes = want.filter(a => a.show > 0.3).map(a => { const r = a.el.getBoundingClientRect(); return { name: a.name, x0: r.left - hb.left, y0: r.top - hb.top, x1: r.right - hb.left, y1: r.bottom - hb.top, color: getComputedStyle(a.el).color }; });
    }
  }

  // ---------- interaction: drag to rotate ----------
  let dragging = false, px = 0, py = 0, yawUser = 0, pitchUser = 0;
  const el = renderer.domElement;
  el.style.touchAction = "pan-y";
  el.addEventListener("pointerdown", e => { dragging = true; px = e.clientX; py = e.clientY; el.setPointerCapture(e.pointerId); });
  el.addEventListener("pointermove", e => {
    if (!dragging) return;
    yawUser += (e.clientX - px) * 0.008;
    pitchUser = Math.max(-0.8, Math.min(0.8, pitchUser + (e.clientY - py) * 0.006));
    px = e.clientX; py = e.clientY;
    requestDraw();
  });
  ["pointerup", "pointercancel", "pointerleave"].forEach(ev => el.addEventListener(ev, () => { dragging = false; }));

  // ---------- clock and loop ----------
  // the animation runs on its own clock, advanced by at most 0.1 s per frame: a hidden tab or a
  // stall resumes where it left off instead of replaying the missed time. Rendering stops while
  // the figure is off-screen, the tab is hidden, the viewer has paused it, or the context is lost.
  const prm = !!(window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches);
  const reduce = FIXT !== null || prm;
  const PRM_T = 14.3;  // reduced motion: a still of step 4, both parts of the push shown, the removed one crossed out
  if (prm && stepsEl) {
    stepsEl.classList.add("all");
    stepText.innerHTML = "<ol>" + STEPS.map(s => `<li>${s[2]}</li>`).join("") + "</ol>";
  }
  let clock = 0, lastNow = null, running = false, paused = false, onScreen = true, ctxLost = false, drawQueued = false;
  let cycleIdx = null;
  let qStart = new THREE.Quaternion().setFromEuler(new THREE.Euler(0.18, -0.35, 0)), qTarget = qStart.clone();
  const qBase = new THREE.Quaternion(), qUser = new THREE.Quaternion(), qSway = new THREE.Quaternion(), eul = new THREE.Euler();
  const CAM_POS = camera.position.clone(), CAM_LOOK = V(0, -0.08, 0), camTarget = CAM_LOOK.clone();
  // the close-up looks at the wall 40 degrees off its normal; while the push is split it tilts to
  // 60 degrees, so the part across the wall visibly stands out of it
  const TILT_A = 0.70, TILT_B = 1.05, FIT = 1.3, AIM_X = 0.28, AIM_Y = 0.12, keyC = V(), kp = V(), camDir = V(), camRight = V(), camUp = V(), aim = V();
  function draw() {
    if (ctxLost) return;
    dbg.frames = (dbg.frames || 0) + 1;
    const T = FIXT !== null ? FIXT : prm ? PRM_T : clock;
    const ci = Math.floor(T / CYCLE);
    if (ci !== cycleIdx) {
      // a frozen ?t= frame replays the earlier cycles' turns so it matches the live animation
      const from = cycleIdx === null ? 0 : cycleIdx + 1;
      for (let k = Math.max(from, ci - 12); k <= ci; k++) { qStart = qTarget.clone(); qTarget = newCycle(qStart); }
      cycleIdx = ci;
    }
    const tv = T - ci * CYCLE, tc = warp(tv);
    dtFrame = lastT === null || reduce ? 1 : Math.max(0, T - lastT); lastT = T;
    qBase.copy(qStart).slerp(qTarget, ease(0.0, 0.9, tc));
    qSway.setFromEuler(eul.set(0, prm ? 0 : 0.1 * Math.sin(T * 0.5), 0));
    qUser.setFromEuler(eul.set(pitchUser, yawUser, 0));
    world.quaternion.copy(qUser).multiply(qSway).multiply(qBase);
    world.updateMatrixWorld();
    // dolly in on the tight face while the loss and its gradient are shown, back out at the end
    const zoom = ease(3.6, 6.2, tc) * (1 - ease(16.6, 17.9, tc));
    const fc = F.c.clone().applyMatrix4(world.matrixWorld);
    const nW = F.n.clone().transformDirection(world.matrixWorld);
    // tilt within the plane spanned by n and (n x kept direction): the kept part keeps its full
    // length on screen and the removed part appears at a right angle to it
    const side = loss.side.clone().transformDirection(world.matrixWorld);
    if (side.x > 0) side.negate();
    const tilt = TILT_A + (TILT_B - TILT_A) * ease(6.9, 8.0, tc) * (1 - ease(13.3, 14.3, tc));
    camDir.copy(nW).multiplyScalar(Math.cos(tilt)).addScaledVector(side, Math.sin(tilt)).normalize();
    // fit: centre of the key points' bounding sphere, distance so the sphere fills the view
    keyC.set(0, 0, 0);
    loss.key.forEach(p => keyC.add(kp.copy(p).applyMatrix4(world.matrixWorld)));
    keyC.multiplyScalar(1 / loss.key.length);
    let R = 0; loss.key.forEach(p => { R = Math.max(R, kp.copy(p).applyMatrix4(world.matrixWorld).distanceTo(keyC)); });
    const halfFov = Math.min(camera.fov, 2 * Math.atan(Math.tan(camera.fov * Math.PI / 360) * camera.aspect) * 180 / Math.PI) * Math.PI / 360;
    const D = FIT * R / Math.sin(halfFov);
    // aim a little left of and below the key points on wide boxes: the shape's body fills the left,
    // the estimate and the arrows get the open space on the right
    camRight.crossVectors(kp.copy(camDir).negate(), V(0, 1, 0)).normalize();
    camUp.crossVectors(camRight, kp.copy(camDir).negate());
    const aimX = (window.innerWidth < 900 ? 0 : AIM_X) * Math.min(Math.max((camera.aspect - 1.33) / 0.45, 0), 1);
    aim.copy(keyC).addScaledVector(camRight, -aimX * R).addScaledVector(camUp, -AIM_Y * R);
    camTarget.copy(CAM_LOOK).lerp(aim, zoom);
    camera.position.copy(CAM_POS).lerp(kp.copy(aim).addScaledVector(camDir, D), zoom);
    camera.lookAt(camTarget);
    camera.updateMatrixWorld();
    const u0 = PERF ? performance.now() : 0;
    update(tc, T, tv);
    if (PERF) { dbg.perfN = (dbg.perfN || 0) + 1; dbg.perfMs = (dbg.perfMs || 0) + performance.now() - u0; }
    renderer.render(scene, camera);
    if (FIXT !== null) {
      // debug: bounding box of drawn pixels (alpha > 0.06), in CSS px
      const gl = renderer.getContext(), w = gl.drawingBufferWidth, hgt = gl.drawingBufferHeight, buf = new Uint8Array(w * hgt * 4);
      gl.readPixels(0, 0, w, hgt, gl.RGBA, gl.UNSIGNED_BYTE, buf);
      let x0 = w, x1 = -1, y0 = hgt, y1 = -1;
      for (let yy = 0; yy < hgt; yy += 2) for (let xx = 0; xx < w; xx += 2) if (buf[(yy * w + xx) * 4 + 3] > 15) { if (xx < x0) x0 = xx; if (xx > x1) x1 = xx; if (yy < y0) y0 = yy; if (yy > y1) y1 = yy; }
      const sc = host.clientWidth / w;
      dbg.alphaBox = { x0: x0 * sc, x1: x1 * sc, top: (hgt - 1 - y1) * sc, bottom: (hgt - 1 - y0) * sc, W: host.clientWidth, H: host.clientHeight };
    }
  }
  const wantRun = () => !reduce && !paused && onScreen && !document.hidden && !ctxLost;
  function tick(now) {
    if (!wantRun()) { running = false; return; }
    const dt = lastNow === null ? 0 : Math.min(Math.max((now - lastNow) / 1000, 0), 0.1);
    lastNow = now; clock += dt;
    draw();
    requestAnimationFrame(tick);
  }
  function kick() { if (!running && wantRun()) { running = true; lastNow = null; requestAnimationFrame(tick); } }
  function requestDraw() {
    if (running || drawQueued) return;
    drawQueued = true;
    requestAnimationFrame(() => { drawQueued = false; draw(); });
  }

  let lastW = 0, lastH = 0;
  function resize() {
    const W = host.clientWidth, H = host.clientHeight;
    if (!W || !H) return;
    if (W !== lastW || H !== lastH) {
      lastW = W; lastH = H;
      renderer.setSize(W, H, false);
      el.style.width = W + "px"; el.style.height = H + "px";
      camera.aspect = W / H;
      camera.updateProjectionMatrix();
      world.position.x = window.innerWidth < 900 ? -0.12 : -0.32;
    }
    reserveCaption();
    requestDraw();
  }
  if (window.ResizeObserver) new ResizeObserver(resize).observe(host);
  window.addEventListener("resize", resize);
  resize();
  if (document.fonts && document.fonts.ready) document.fonts.ready.then(resize);
  window.addEventListener("load", resize);

  if (window.IntersectionObserver) new IntersectionObserver(es => { onScreen = es[es.length - 1].isIntersecting; kick(); }).observe(host);
  document.addEventListener("visibilitychange", kick);

  // pause / play
  const pauseBtn = stepsEl && stepsEl.querySelector(".h3d-pause");
  if (pauseBtn) {
    if (reduce) pauseBtn.hidden = true;
    pauseBtn.addEventListener("click", () => {
      paused = !paused;
      pauseBtn.setAttribute("aria-pressed", paused ? "true" : "false");
      pauseBtn.classList.toggle("paused", paused);
      pauseBtn.title = paused ? "Play animation" : "Pause animation";
      if (!paused) kick();
    });
  }

  // a lost WebGL context: three.js restores buffers and textures itself, but the prefiltered
  // environment map was rendered on the GPU and has to be built again
  el.addEventListener("webglcontextlost", e => { e.preventDefault(); ctxLost = true; }, false);
  el.addEventListener("webglcontextrestored", () => {
    ctxLost = false;
    buildEnv(true);
    if (shadowTex) shadowTex.needsUpdate = true;
    scene.traverse(o => { if (o.material) [].concat(o.material).forEach(m => { m.needsUpdate = true; }); });
    kick(); requestDraw();
  }, false);

  if (FIXT !== null) dbg.render = t => { FIXT = Math.min(Math.max(+t || 0, 0), 240); draw(); return dbg; };
  if (DEBUG) dbg.advance = sec => { for (let s = 0; s < sec; s += 0.1) { clock += 0.1; draw(); } return dbg.info(); };
  dbg.info = () => ({ geometries: renderer.info.memory.geometries, textures: renderer.info.memory.textures, programs: renderer.info.programs ? renderer.info.programs.length : null, clock, running });
  if (reduce) requestDraw(); else kick();
})();
