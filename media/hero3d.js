// Hero scene: a prediction f(x) outside a convex polytope C(x) is projected onto it.
// Dykstra's algorithm (a true Euclidean-projection solver) zig-zags onto the tight face;
// y = Pi_C(f) has free directions along the face and a blocked direction along its
// outward normal, and f - y always lies on that normal. Training then moves f and y
// together while y stays inside the face. Cycles over faces; drag to rotate.
// Debug: ?t=4.2 freezes the timeline, ?seed=N fixes the random choices.
(function () {
  const host = document.getElementById("hero3d");
  if (!host || !window.THREE) return;
  const qs = new URLSearchParams(location.search);
  const FIXT = qs.has("t") ? parseFloat(qs.get("t")) : null;
  let seed = qs.has("seed") ? (parseInt(qs.get("seed"), 10) >>> 0) || 1 : null;
  const rand = seed === null ? Math.random : () => { seed = (seed * 1664525 + 1013904223) >>> 0; return seed / 4294967296; };
  let renderer;
  try {
    renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
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
  // soft studio environment (a gradient dome with two light panels) for glassy reflections
  {
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
      const m = new THREE.Mesh(new THREE.PlaneGeometry(s, s), new THREE.MeshBasicMaterial({ color: 0xffffff, side: THREE.DoubleSide }));
      m.position.copy(p); m.lookAt(0, 0, 0); env.add(m);
    });
    const pm = new THREE.PMREMGenerator(renderer);
    scene.environment = pm.fromScene(env, 0.02).texture;
    pm.dispose();
  }
  scene.fog = new THREE.Fog(0xdde9fb, 4.7, 8.6); // far edges and glass fade into the page: depth cue
  scene.add(new THREE.HemisphereLight(0xffffff, 0xbcd3f2, 0.55));
  const sun = new THREE.DirectionalLight(0xffffff, 0.6);
  sun.position.set(3, 5, 4);
  scene.add(sun);
  const rim = new THREE.DirectionalLight(0xbfdbfe, 0.7);
  rim.position.set(-4, 2, -5);
  scene.add(rim);

  const world = new THREE.Group();
  scene.add(world);

  // ---------- feasible set: an irregular dodecahedron (convex polytope) ----------
  const geo = new THREE.DodecahedronGeometry(1.15, 0);
  geo.scale(1.25, 0.95, 1.05);
  geo.rotateZ(0.18);
  geo.computeVertexNormals();
  const glass = { color: 0x6fa2ec, roughness: 0.14, metalness: 0, clearcoat: 1, clearcoatRoughness: 0.08,
    envMapIntensity: 0.75, flatShading: true, transparent: true, depthWrite: false };
  const inner = new THREE.Mesh(geo, new THREE.MeshPhysicalMaterial(Object.assign({}, glass, { side: THREE.BackSide, opacity: 0.36, color: 0x4a84dc })));
  const shell = new THREE.Mesh(geo, new THREE.MeshPhysicalMaterial(Object.assign({}, glass, { side: THREE.FrontSide, opacity: 0.34 })));
  inner.renderOrder = 0; shell.renderOrder = 2;
  world.add(inner, shell);
  {
    const eg = new THREE.EdgesGeometry(geo, 1).attributes.position;
    const em = new THREE.MeshBasicMaterial({ color: NAVY, transparent: true, opacity: 0.8 });
    for (let i = 0; i < eg.count; i += 2) {
      const a = V().fromBufferAttribute(eg, i), b = V().fromBufferAttribute(eg, i + 1), d = V().subVectors(b, a);
      const r = new THREE.Mesh(new THREE.CylinderGeometry(0.0075, 0.0075, d.length(), 6), em);
      r.position.copy(a).addScaledVector(d, 0.5);
      r.quaternion.setFromUnitVectors(V(0, 1, 0), d.normalize());
      r.renderOrder = 3;
      world.add(r);
    }
  }
  // soft contact shadow on an invisible floor
  {
    const cv = document.createElement("canvas"); cv.width = cv.height = 128;
    const cx = cv.getContext("2d"), gr = cx.createRadialGradient(64, 64, 0, 64, 64, 64);
    gr.addColorStop(0, "rgba(30,58,138,0.55)"); gr.addColorStop(0.55, "rgba(30,58,138,0.16)"); gr.addColorStop(1, "rgba(30,58,138,0)");
    cx.fillStyle = gr; cx.fillRect(0, 0, 128, 128);
    const sh = new THREE.Mesh(new THREE.PlaneGeometry(3.4, 3.4), new THREE.MeshBasicMaterial({ map: new THREE.CanvasTexture(cv), transparent: true, depthWrite: false }));
    sh.rotation.x = -Math.PI / 2; sh.position.set(-0.32, -1.3, 0); sh.scale.set(0.9, 0.62, 1);
    scene.add(sh);
  }

  // planar faces as half-spaces n.x <= d with outward unit normal n
  const pos = geo.attributes.position;
  const faces = [];
  for (let i = 0; i < pos.count; i += 3) {
    const a = V().fromBufferAttribute(pos, i), b = V().fromBufferAttribute(pos, i + 1), c = V().fromBufferAttribute(pos, i + 2);
    const n = V().subVectors(b, a).cross(V().subVectors(c, a)).normalize();
    if (n.dot(a) < 0) n.negate();
    let f = faces.find(F => F.n.dot(n) > 0.999 && Math.abs(F.n.dot(a) - F.d) < 1e-4);
    if (!f) { f = { n, d: n.dot(a), verts: [] }; faces.push(f); }
    [a, b, c].forEach(v => { if (!f.verts.some(w => w.distanceTo(v) < 1e-5)) f.verts.push(v); });
  }
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

  const hullVerts = [];
  faces.forEach(G => G.verts.forEach(v => { if (!hullVerts.some(w => w.distanceTo(v) < 1e-5)) hullVerts.push(v); }));

  // ---------- helpers ----------
  const fadeMats = [];
  function mat(color, extra) {
    const m = new THREE.MeshPhysicalMaterial(Object.assign({ color, transparent: true, opacity: 0, roughness: 0.35, clearcoat: 0.6, clearcoatRoughness: 0.2, envMapIntensity: 0.3 }, extra || {}));
    fadeMats.push(m);
    return m;
  }
  function rod(a, b, r, m) {
    const d = V().subVectors(b, a), L = d.length();
    const o = new THREE.Mesh(new THREE.CylinderGeometry(r, r, L, 10), m);
    o.position.copy(a).addScaledVector(d, 0.5);
    o.quaternion.setFromUnitVectors(V(0, 1, 0), d.clone().normalize());
    return o;
  }
  function arrow(color) {
    const m = mat(color, { emissive: color, emissiveIntensity: 0.4 });
    const g = new THREE.Group();
    const shaft = new THREE.Mesh(new THREE.CylinderGeometry(0.018, 0.018, 1, 10), m);
    const head = new THREE.Mesh(new THREE.ConeGeometry(0.055, 0.14, 14), m);
    g.add(shaft, head);
    g.userData = { shaft, head, m };
    world.add(g);
    return g;
  }
  function placeArrow(g, origin, dir, len) {
    const L = Math.max(len, 1e-3);
    g.position.copy(origin);
    g.quaternion.setFromUnitVectors(V(0, 1, 0), dir.clone().normalize());
    g.userData.shaft.scale.set(1, Math.max(L - 0.12, 1e-3), 1);
    g.userData.shaft.position.set(0, Math.max(L - 0.12, 1e-3) / 2, 0);
    g.userData.head.position.set(0, L - 0.06, 0);
  }

  // ---------- per-cycle objects ----------
  const faceMat = new THREE.MeshBasicMaterial({ color: 0xfdba74, transparent: true, opacity: 0, side: THREE.DoubleSide, depthWrite: false });
  const edgeMat = mat(ORANGE, { emissive: 0xc2410c, emissiveIntensity: 0.55 });
  let faceMesh = null, faceEdges = null;
  const fSphere = new THREE.Mesh(new THREE.SphereGeometry(0.075, 24, 16), mat(BLUE, { emissive: 0x1d4ed8, emissiveIntensity: 0.35 }));
  const ySphere = new THREE.Mesh(new THREE.SphereGeometry(0.07, 24, 16), mat(ORANGE, { emissive: 0xea580c, emissiveIntensity: 0.45 }));
  const glow = new THREE.Mesh(new THREE.SphereGeometry(0.16, 24, 16), new THREE.MeshBasicMaterial({ color: ORANGE, transparent: true, opacity: 0, depthWrite: false }));
  world.add(fSphere, ySphere, glow);
  // residual f(x) - y as dashes along the outward normal, and a right-angle mark at y
  const dashMat = new THREE.MeshBasicMaterial({ color: NAVY, transparent: true, opacity: 0, depthWrite: false });
  const dashGeo = new THREE.CylinderGeometry(0.008, 0.008, 1, 6); dashGeo.translate(0, 0.5, 0);
  const dashes = [];
  for (let k = 0; k < 18; k++) { const d = new THREE.Mesh(dashGeo, dashMat); d.renderOrder = 4; dashes.push(d); world.add(d); }
  const cornerMat = new THREE.MeshBasicMaterial({ color: NAVY, transparent: true, opacity: 0, depthWrite: false });
  const cornerGeo = new THREE.CylinderGeometry(0.011, 0.011, 1, 6); cornerGeo.translate(0, 0.5, 0);
  const corner = [0, 1].map(() => { const m = new THREE.Mesh(cornerGeo, cornerMat); m.renderOrder = 4; world.add(m); return m; });
  // solver path: one tube per Dykstra step, revealed in order, with a bead running along it
  const itMat = mat(0x3b82f6, { emissive: 0x2563eb, emissiveIntensity: 0.5 });
  const dotMat = mat(0xe0edff, { emissive: 0x93c5fd, emissiveIntensity: 0.6 });
  const MAXIT = 14;
  const segGeo = new THREE.CylinderGeometry(0.011, 0.011, 1, 8); segGeo.translate(0, 0.5, 0);
  const segs = [], dots = [];
  for (let k = 0; k < MAXIT; k++) {
    const sg = new THREE.Mesh(segGeo, itMat); sg.renderOrder = 4; segs.push(sg); world.add(sg);
    const d = new THREE.Mesh(new THREE.SphereGeometry(0.026, 14, 10), dotMat); d.renderOrder = 4; dots.push(d); world.add(d);
  }
  const bead = new THREE.Mesh(new THREE.SphereGeometry(0.04, 18, 12), new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0 }));
  bead.renderOrder = 5; world.add(bead);
  const free1 = arrow(BLUE), free2 = arrow(BLUE), blocked = arrow(ORANGE);

  let F, y0, h, its, faceLabelAt, freeA, freeB, trainU, trainW, trainA, trainB;
  const labels = {};
  host.querySelectorAll("[data-label]").forEach(el => { labels[el.dataset.label] = el; });

  // the chosen face is turned so its outward normal points here (world frame):
  // toward the camera, to the right and slightly up, so f(x) floats in open space.
  const NSTAR = V(0.63, 0.17, 0.76).normalize();
  let lastFace = -1;
  function pickFace(q) {
    // a face that needs a moderate turn to reach NSTAR, never the previous one
    const cand = [];
    faces.forEach((G, i) => {
      if (i === lastFace) return;
      const c = G.n.clone().applyQuaternion(q).dot(NSTAR);
      if (c > 0.05 && c < 0.8) cand.push(i);
    });
    const pool = cand.length ? cand : faces.map((_, i) => i).filter(i => i !== lastFace);
    lastFace = pool[Math.floor(rand() * pool.length)];
    return lastFace;
  }
  const dbg = window.__heroDebug = {};
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
    world.add(faceMesh);
    faceEdges = new THREE.Group();
    F.verts.forEach((v, i) => faceEdges.add(rod(v, F.verts[(i + 1) % F.verts.length], 0.022, edgeMat)));
    world.add(faceEdges);

    // y0: interior point shifted toward one edge, so f also violates the neighbouring face
    // and the solver has to zig-zag between the two before settling on the tight face.
    const qi = q.clone().invert();
    // the shared edge toward screen-up: the zig-zag then opens across the f -> y line
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
    dots.forEach((d, k) => { d.userData.on = k + 1 < its.length - 1; if (d.userData.on) d.position.copy(its[k + 1]); });
    // free directions: the in-face directions that read best on screen (screen-right and its in-face normal)
    freeA = V().subVectors(F.c, y0).normalize();
    freeB = V().crossVectors(F.n, freeA).normalize();
    if (freeB.clone().applyQuaternion(q).x < 0) freeB.negate();

    // training path: a smooth closed loop that starts at y0 and sweeps through the face
    // interior; shrunk until every point keeps a margin from the face boundary
    trainU = V().subVectors(F.c, y0); trainA = 0.5 * trainU.length(); trainU.normalize();
    trainW = V().crossVectors(F.n, trainU).normalize();
    trainB = 0.6 * F.r;
    for (let tries = 0; tries < 40; tries++) {
      let ok = true;
      for (let s = 0; s < 96 && ok; s++) if (insideMargin(F, trainPoint(s / 96)) < 0.06) ok = false;
      if (ok) break;
      trainB *= 0.9;
    }
    let far = F.verts[0], fd = -1;
    F.verts.forEach(v => { const d = v.distanceTo(fpt); if (d > fd) { fd = d; far = v; } });
    faceLabelAt = far.clone().addScaledVector(V().subVectors(far, F.c).normalize(), 0.12);

    // numerical checks
    const yStar = dykstra(fpt, order, 400, false);
    let minIn = 1e9; for (let s = 0; s < 200; s++) minIn = Math.min(minIn, insideMargin(F, trainPoint(s / 200)));
    let projErrTrain = 0;
    for (let s = 0; s < 8; s++) {
      const yt = trainPoint(s / 8), ft = yt.clone().addScaledVector(F.n, h);
      projErrTrain = Math.max(projErrTrain, dykstra(ft, order, 400, false).distanceTo(yt));
    }
    Object.assign(dbg, {
      face: fi, h: +h.toFixed(3), iterates: its.length,
      projErr: yStar.distanceTo(y0), projErrTrain,
      y0Margin: insideMargin(F, y0), trainMinMargin: minIn,
      freeDotN: Math.max(Math.abs(freeA.dot(F.n)), Math.abs(freeB.dot(F.n))),
      outward: F.n.dot(F.c) > 0, onPlane: Math.abs(F.n.dot(y0) - F.d)
    });
    return q;
  }
  function trainPoint(s) {
    const th = s * Math.PI * 2;
    const p = y0.clone().addScaledVector(trainU, trainA * (1 - Math.cos(th))).addScaledVector(trainW, trainB * Math.sin(th));
    return p.addScaledVector(F.n, F.d - F.n.dot(p));
  }

  // ---------- timeline ----------
  const CYCLE = 10.0;
  const ease = (a, b, t) => { const u = Math.min(Math.max((t - a) / (b - a), 0), 1); return u * u * (3 - 2 * u); };
  function setOpacity(m, o) { m.opacity = o; m.visible = o > 0.003; }
  const tmp = V();

  function toScreen(p3, W, H) {
    tmp.copy(p3).applyMatrix4(world.matrixWorld).project(camera);
    return { x: (tmp.x * 0.5 + 0.5) * W, y: (-tmp.y * 0.5 + 0.5) * H };
  }
  function update(tc, T) {
    tc -= 0.8;
    const appear = ease(0.0, 0.5, tc), out = 1 - ease(CYCLE - 1.6, CYCLE - 0.8, tc);
    const slide = ease(5.0, CYCLE - 2.0, tc);
    const yy = trainPoint(slide);
    const fpt = yy.clone().addScaledVector(F.n, h);
    fSphere.position.copy(fpt);
    setOpacity(fSphere.material, appear * out);
    // iterates
    const u = Math.min(Math.max((tc - 0.6) / 2.4, 0), 1);
    const drawK = (1 - Math.pow(1 - u, 1.6)) * (its.length - 1);
    segs.forEach((sg, k) => {
      const part = Math.min(Math.max(drawK - k, 0), 1);
      sg.visible = sg.userData.on && part > 0;
      if (sg.visible) sg.scale.set(1, sg.userData.L * part, 1);
    });
    dots.forEach((d, k) => { d.visible = d.userData.on && drawK >= k + 1; });
    const fadeIt = out * (1 - ease(4.4, 5.4, tc));
    setOpacity(itMat, 0.9 * fadeIt); setOpacity(dotMat, fadeIt);
    { const k = Math.min(Math.floor(drawK), its.length - 2), fr = drawK - k;
      bead.position.copy(its[k]).lerp(its[k + 1], Math.min(fr, 1));
      setOpacity(bead.material, u > 0 && u < 1 ? 0.95 * out : 0); }
    // tight face
    const faceOn = ease(2.6, 3.3, tc) * out;
    faceMat.opacity = 0.62 * faceOn;
    setOpacity(edgeMat, faceOn);
    // y
    ySphere.position.copy(yy);
    glow.position.copy(yy);
    const yOn = ease(3.0, 3.4, tc) * out;
    setOpacity(ySphere.material, yOn);
    glow.material.opacity = 0.22 * yOn * (0.75 + 0.25 * Math.sin(T * 4));
    glow.visible = yOn > 0;
    // residual f - y, always along the outward normal (optimality: f - y is normal to the face)
    const resOn = ease(4.6, 5.2, tc) * out;
    const DL = 0.045, GAP = 0.035, nD = Math.min(dashes.length, Math.floor(h / (DL + GAP)));
    dashes.forEach((d, k) => {
      d.visible = resOn > 0.01 && k < nD;
      if (!d.visible) return;
      d.position.copy(yy).addScaledVector(F.n, 0.06 + k * (DL + GAP));
      d.quaternion.setFromUnitVectors(V(0, 1, 0), F.n);
      d.scale.set(1, DL, 1);
    });
    dashMat.opacity = 0.7 * resOn;
    const cs = 0.12;
    corner[0].position.copy(yy).addScaledVector(F.n, cs); corner[0].quaternion.setFromUnitVectors(V(0, 1, 0), freeB.clone().negate()); corner[0].scale.set(1, cs, 1);
    corner[1].position.copy(yy).addScaledVector(freeB, -cs); corner[1].quaternion.setFromUnitVectors(V(0, 1, 0), F.n); corner[1].scale.set(1, cs, 1);
    corner.forEach(c => { c.visible = resOn > 0.01; });
    cornerMat.opacity = 0.85 * resOn;
    // free / blocked directions at y
    const arr = ease(3.6, 4.5, tc);
    placeArrow(free1, yy, freeA, 0.42 * arr + 1e-3);
    placeArrow(free2, yy, freeB, 0.36 * arr + 1e-3);
    placeArrow(blocked, yy, F.n, 0.45 * arr + 1e-3);
    [free1, free2, blocked].forEach(a => setOpacity(a.userData.m, Math.min(arr * 1.5, 1) * out));

    // ---------- HTML labels, relaxed so they never overlap ----------
    const W = host.clientWidth, H = host.clientHeight;
    const want = [];
    function put(name, p3, show, dx, dy) {
      const el = labels[name];
      if (!el) return;
      tmp.copy(p3).applyMatrix4(world.matrixWorld).project(camera);
      want.push({ name, el, show, x: (tmp.x * 0.5 + 0.5) * W + dx, y: (-tmp.y * 0.5 + 0.5) * H + dy,
        w: el.offsetWidth + 6, h: el.offsetHeight + 2 });
    }
    put("f", fpt, appear * out, 0, -20);
    put("y", yy, yOn, -18, 10);
    put("face", faceLabelAt, faceOn, 0, 0);
    put("free", yy.clone().addScaledVector(freeA, 0.56), Math.min(arr * 1.5, 1) * out, 0, 0);
    // blocked label beside the normal arrow, on the side away from the free arrows
    {
      const a2 = toScreen(yy, W, H), b2 = toScreen(yy.clone().addScaledVector(F.n, 0.3), W, H), c2 = toScreen(yy.clone().addScaledVector(freeB, 0.3), W, H);
      let px = -(b2.y - a2.y), py = b2.x - a2.x; const L = Math.hypot(px, py) || 1; px /= L; py /= L;
      if (py > 0) { px = -px; py = -py; }
      put("blocked", yy.clone().addScaledVector(F.n, 0.3), Math.min(arr * 1.5, 1) * out, 34 * px, 34 * py);
    }
    {
      // C(x) sits just outside the lowest-left vertex of the set as it appears on screen
      let best = null, bs = -1e9;
      for (const v of hullVerts) { const p = toScreen(v, W, H), sc = p.y - 0.6 * p.x; if (sc > bs) { bs = sc; best = v; } }
      put("C", best, 1, -16, 10);
    }
    const vis = want.filter(a => a.show > 0.05);
    for (let it = 0; it < 30; it++) {
      for (let i = 0; i < vis.length; i++) for (let j = i + 1; j < vis.length; j++) {
        const a = vis[i], b = vis[j];
        const ox = (a.w + b.w) / 2 - Math.abs(a.x - b.x), oy = (a.h + b.h) / 2 - Math.abs(a.y - b.y);
        if (ox > 0 && oy > 0) {
          if (ox < oy) { const s = (a.x < b.x ? -1 : 1) * ox / 2; a.x += s; b.x -= s; }
          else { const s = (a.y < b.y ? -1 : 1) * oy / 2; a.y += s; b.y -= s; }
        }
      }
      vis.forEach(a => { a.x = Math.min(Math.max(a.x, a.w / 2), W - a.w / 2); a.y = Math.min(Math.max(a.y, a.h / 2), H - a.h / 2); });
    }
    want.forEach(a => {
      a.el.style.transform = `translate(${a.x}px, ${a.y}px) translate(-50%, -50%)`;
      a.el.style.opacity = a.show;
    });
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
  });
  ["pointerup", "pointercancel", "pointerleave"].forEach(ev => el.addEventListener(ev, () => { dragging = false; }));

  function resize() {
    const W = host.clientWidth, H = host.clientHeight;
    renderer.setSize(W, H, false);
    el.style.width = W + "px"; el.style.height = H + "px";
    camera.aspect = W / H;
    camera.updateProjectionMatrix();
  }
  window.addEventListener("resize", resize);
  resize();

  const prm = !!(window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches);
  const reduce = FIXT !== null || prm;
  const t0 = performance.now();
  let cycleIdx = null;
  let qStart = new THREE.Quaternion().setFromEuler(new THREE.Euler(0.18, -0.35, 0)), qTarget = qStart.clone();
  const qBase = new THREE.Quaternion(), qUser = new THREE.Quaternion(), qSway = new THREE.Quaternion();
  world.position.x = -0.32;
  function frame(now) {
    const T = FIXT !== null ? FIXT : prm ? 6.5 : Math.max(0, (now - t0) / 1000);
    const ci = Math.floor(T / CYCLE);
    if (ci !== cycleIdx) {
      // replay the turns of skipped cycles so a frozen ?t= frame matches the live animation
      for (let k = cycleIdx === null ? 0 : cycleIdx + 1; k <= ci; k++) { qStart = qTarget.clone(); qTarget = newCycle(qStart); }
      cycleIdx = ci;
    }
    const tc = T - ci * CYCLE;
    qBase.copy(qStart).slerp(qTarget, ease(0.0, 1.3, tc));
    qSway.setFromEuler(new THREE.Euler(0, prm ? 0 : 0.1 * Math.sin(T * 0.5), 0));
    qUser.setFromEuler(new THREE.Euler(pitchUser, yawUser, 0));
    world.quaternion.copy(qUser).multiply(qSway).multiply(qBase);
    world.updateMatrixWorld();
    update(tc, T);
    renderer.render(scene, camera);
    if (!reduce) requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
  // static modes (reduced motion, frozen ?t=) draw on demand once layout and fonts settle
  const redraw = () => { if (reduce) requestAnimationFrame(frame); };
  if (reduce) {
    window.addEventListener("resize", redraw);
    el.addEventListener("pointermove", () => { if (dragging) redraw(); });
    if (document.fonts && document.fonts.ready) document.fonts.ready.then(redraw);
    window.addEventListener("load", redraw);
  }
})();
