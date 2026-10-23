// Hero scene: a prediction f(x) outside a convex polytope C(x) is projected onto it.
// Dykstra's algorithm (an exact Euclidean-projection solver) zig-zags onto the tight face,
// so y = Pi_C(f) and f - y lies on the face's outward normal. A quadratic loss on y is then
// drawn as level sets on the face; its descent direction -grad L splits into an in-face part,
// which the layer passes back, and a normal part, which it blocks. Training then moves y
// (and f with it) down the loss along the face. Cycles over faces; drag to rotate.
// Debug: ?t=8.2 freezes the timeline, ?seed=N fixes the random choices.
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
      const m = new THREE.Mesh(new THREE.PlaneGeometry(s, s), new THREE.MeshBasicMaterial({ color: 0xdce7f7, side: THREE.DoubleSide }));
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
  const glass = { color: 0x6fa2ec, roughness: 0.34, metalness: 0, clearcoat: 0.55, clearcoatRoughness: 0.18,
    envMapIntensity: 0.55, flatShading: true, transparent: true, depthWrite: false };
  const inner = new THREE.Mesh(geo, new THREE.MeshPhysicalMaterial(Object.assign({}, glass, { side: THREE.BackSide, opacity: 0.36, color: 0x4a84dc })));
  const shell = new THREE.Mesh(geo, new THREE.MeshPhysicalMaterial(Object.assign({}, glass, { side: THREE.FrontSide, opacity: 0.34 })));
  inner.renderOrder = 0; shell.renderOrder = 2;
  world.add(inner, shell);
  {
    // edges of the twelve main faces read strongly; the bevel seams stay faint
    const eg = new THREE.EdgesGeometry(geo, 1).attributes.position;
    const em = new THREE.MeshBasicMaterial({ color: NAVY, transparent: true, opacity: 0.7 });
    const emSoft = new THREE.MeshBasicMaterial({ color: NAVY, transparent: true, opacity: 0.5 });
    const polys = geo.userData.polys;
    const area = P => P.verts.reduce((s, v, i) => s + V().subVectors(v, P.verts[0]).cross(V().subVectors(P.verts[(i + 1) % P.verts.length], P.verts[0])).length() / 2, 0);
    polys.forEach(P => { P.area = area(P); });
    const aMax = Math.max(...polys.map(P => P.area));
    for (let i = 0; i < eg.count; i += 2) {
      const a = V().fromBufferAttribute(eg, i), b = V().fromBufferAttribute(eg, i + 1), d = V().subVectors(b, a);
      const main = polys.some(P => P.area > 0.5 * aMax && Math.abs(P.n.dot(a) - P.d) < 1e-6 && Math.abs(P.n.dot(b) - P.d) < 1e-6);
      const r = new THREE.Mesh(new THREE.CylinderGeometry(main ? 0.0068 : 0.0052, main ? 0.0068 : 0.0052, d.length(), 6), main ? em : emSoft);
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
    const s = Math.min(1, L / 0.14);
    const sl = Math.max(L - 0.12 * s, 1e-3);
    g.userData.shaft.scale.set(1, sl, 1);
    g.userData.shaft.position.set(0, sl / 2, 0);
    g.userData.head.scale.setScalar(s);
    g.userData.head.position.set(0, L - 0.07 * s, 0);
  }

  // ---------- per-cycle objects ----------
  const faceMat = new THREE.MeshBasicMaterial({ color: 0xfdba74, transparent: true, opacity: 0, side: THREE.DoubleSide, depthWrite: false });
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
  const resDash = dashPool(18, NAVY), splitDash = dashPool(28, NAVY);
  const cornerMat = new THREE.MeshBasicMaterial({ color: NAVY, transparent: true, opacity: 0, depthWrite: false });
  const cornerGeo = new THREE.CylinderGeometry(0.011, 0.011, 1, 6); cornerGeo.translate(0, 0.5, 0);
  const corner = [0, 1].map(() => { const m = new THREE.Mesh(cornerGeo, cornerMat); m.renderOrder = 4; world.add(m); return m; });
  // solver path: one tube per Dykstra step, revealed in order, with a bead running along it
  const itMat = mat(0x3b82f6, { emissive: 0x2563eb, emissiveIntensity: 0.5 });
  const dotMat = mat(0xe0edff, { emissive: 0x93c5fd, emissiveIntensity: 0.6 });
  const MAXIT = 16;
  const segGeo = new THREE.CylinderGeometry(0.011, 0.011, 1, 8); segGeo.translate(0, 0.5, 0);
  const segs = [], dots = [];
  for (let k = 0; k < MAXIT; k++) {
    const sg = new THREE.Mesh(segGeo, itMat); sg.renderOrder = 4; segs.push(sg); world.add(sg);
    const d = new THREE.Mesh(new THREE.SphereGeometry(0.026, 14, 10), dotMat); d.renderOrder = 4; dots.push(d); world.add(d);
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
  const SLATE = 0x334155, GREY = 0x94a3b8;
  const gArr = arrow(SLATE), passArr = arrow(BLUE), blockArr = arrow(0x64748b);
  const blockCol0 = new THREE.Color(0x64748b), blockCol1 = new THREE.Color(0xcbd5e1);
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

  // the chosen face is turned so its outward normal points here (world frame):
  // toward the camera, to the right and slightly up, so f(x) floats in open space.
  const NSTAR = V(0.7, 0.18, 0.69).normalize();
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
    faceMesh.renderOrder = 2.5;
    world.add(faceMesh);
    faceEdges = new THREE.Group();
    F.verts.forEach((v, i) => faceEdges.add(rod(v, F.verts[(i + 1) % F.verts.length], 0.022, edgeMat)));
    world.add(faceEdges);

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
    dots.forEach((d, k) => { d.userData.on = k + 1 < its.length - 1; if (d.userData.on) d.position.copy(its[k + 1]); });
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

    let far = F.verts[0], fd = -1;
    F.verts.forEach(v => { const d = v.distanceTo(fpt); if (d > fd) { fd = d; far = v; } });
    faceLabelAt = far.clone().addScaledVector(V().subVectors(far, F.c).normalize(), 0.12);

    // numerical checks
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

  // ---------- timeline (seconds within a cycle): five narrated steps ----------
  const CYCLE = 18.0;
  const STEPS = [
    [0.0, 3.4, "The neural network makes an estimate, outside the allowed region."],
    [3.4, 6.6, "ASPL searches, step by step, for the nearest allowed point."],
    [6.6, 9.8, "Training pushes the output, partly out of the allowed region."],
    [9.8, 13.2, "ASPL keeps the part along the wall and removes the part that leaves the region."],
    [13.2, CYCLE, "The output slides along the wall to a better answer, and stays allowed."],
  ];
  const stepsEl = document.getElementById("hero3d-steps");
  const stepText = stepsEl && stepsEl.querySelector(".h3d-step-text");
  const stepDots = stepsEl ? [...stepsEl.querySelectorAll(".h3d-dots i")] : [];
  let shownStep = -1;
  function narrate(tc) {
    if (!stepText || stepsEl.classList.contains("all")) return;
    let k = STEPS.findIndex(s => tc >= s[0] && tc < s[1]);
    if (k < 0) k = STEPS.length - 1;
    if (k !== shownStep) {
      shownStep = k;
      stepText.innerHTML = `<b>${k + 1}</b>${STEPS[k][2]}`;
      stepDots.forEach((d, i) => { d.className = i === k ? "on" : i < k ? "done" : ""; });
    }
    const [a, b] = STEPS[k];
    stepText.style.opacity = Math.min(1, (tc - a) / 0.35, (b - tc) / 0.3, 1);
    dbg.step = k + 1;
  }
  const ease = (a, b, t) => { const u = Math.min(Math.max((t - a) / (b - a), 0), 1); return u * u * (3 - 2 * u); };
  const lin = (a, b, t) => Math.min(Math.max((t - a) / (b - a), 0), 1);
  function setOpacity(m, o) { m.opacity = o; m.visible = o > 0.003; }
  const tmp = V();
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

  function update(tc, T) {
    narrate(tc);
    const appear = ease(0.6, 1.1, tc), out = 1 - ease(17.3, 17.9, tc);
    const { y: yy, s: gs } = yAt(tc);
    const fpt = yy.clone().addScaledVector(F.n, h);
    fSphere.position.copy(fpt);
    // the guess stays in view but steps back once the output is the subject
    const guessBack = (1 - 0.55 * ease(6.6, 7.4, tc)) * (1 - ease(13.2, 13.9, tc));
    setOpacity(fSphere.material, appear * out * guessBack);
    // step 2: solver iterates
    const u = lin(3.7, 5.6, tc);
    const drawK = (1 - Math.pow(1 - u, 1.35)) * (its.length - 1);
    segs.forEach((sg, k) => {
      const part = Math.min(Math.max(drawK - k, 0), 1);
      sg.visible = sg.userData.on && part > 0;
      if (sg.visible) sg.scale.set(1, sg.userData.L * part, 1);
    });
    dots.forEach(d => { d.visible = false; });
    const fadeIt = out * (1 - ease(5.9, 6.6, tc));
    setOpacity(itMat, 0.9 * fadeIt); setOpacity(dotMat, fadeIt);
    { const k = Math.min(Math.floor(drawK), its.length - 2), fr = drawK - k;
      bead.position.copy(its[k]).lerp(its[k + 1], Math.min(fr, 1));
      setOpacity(bead.material, u > 0 && u < 1 ? 0.95 * out : 0); }
    // the wall it touches, and the output on it
    const faceOn = ease(5.2, 5.7, tc) * out;
    faceMat.opacity = (0.62 + 0.28 * ease(6.8, 7.6, tc)) * faceOn;
    faceMat.depthWrite = tc > 7.6 && tc < 17.3;
    setOpacity(edgeMat, faceOn);
    ySphere.position.copy(yy);
    glow.position.copy(yy);
    const yOn = ease(5.5, 5.9, tc) * out;
    setOpacity(ySphere.material, yOn);
    glow.material.opacity = 0.22 * yOn * (0.75 + 0.25 * Math.sin(T * 4));
    glow.visible = yOn > 0;
    // straight, shortest line from the guess to the output (along the wall's normal)
    const resOn = ease(5.8, 6.3, tc) * out * (1 - 0.45 * ease(6.6, 7.4, tc)) * (1 - ease(13.2, 13.9, tc));
    resDash.list.forEach(d => { d.visible = false; });
    if (resOn > 0.01) dashLine(resDash, 0, yy.clone().addScaledVector(F.n, 0.06), fpt.clone().addScaledVector(F.n, -0.07), 0.045, 0.035);
    resDash.m.opacity = 0.55 * resOn;
    corner.forEach(c => { c.visible = false; });
    // step 3: score rings on the wall around the best point
    const lossOn = ease(6.8, 7.5, tc) * out;
    contourMat.opacity = 0.34 * lossOn; contours.visible = lossOn > 0.01;
    setOpacity(ringMat, 0.9 * lossOn);
    { const k = lin(16.4, 17.2, tc), bump = k > 0 && k < 1 ? Math.sin(Math.PI * k) : 0; ring.scale.setScalar(1 + 0.9 * bump); }
    // the training push at the output, then (step 4) its split into kept and removed parts
    const g = gradL(yy).multiplyScalar(-loss.GS), gT = projT(g), gNv = g.clone().sub(gT);
    const gOn = ease(7.7, 8.3, tc), split = ease(10.0, 10.8, tc), cut = ease(12.0, 12.8, tc);
    placeArrow(gArr, yy, g, g.length() * gOn + 1e-3);
    setOpacity(gArr.userData.m, Math.min(gOn * 1.5, 1) * (1 - 0.9 * split) * (1 - cut) * out);
    placeArrow(passArr, yy, gT.lengthSq() > 1e-12 ? gT : F.t1, gT.length() * split + 1e-3);
    passArr.userData.m.emissiveIntensity = 0.4 + 0.6 * split * (1 - ease(13.2, 13.8, tc)) * (0.5 + 0.5 * Math.sin(T * 5));
    setOpacity(passArr.userData.m, Math.min(split * 1.5, 1) * out * (1 - ease(16.6, 17.2, tc)));
    placeArrow(blockArr, yy, F.n, gNv.length() * split * (1 - cut) + 1e-3);
    blockArr.userData.m.color.copy(blockCol0).lerp(blockCol1, ease(11.1, 11.9, tc));
    setOpacity(blockArr.userData.m, Math.min(split * 1.5, 1) * (1 - cut) * out);
    splitDash.list.forEach(d => { d.visible = false; });
    const sdOn = split * (1 - cut) * out;
    if (sdOn > 0.01) {
      const tip = yy.clone().add(g), k = dashLine(splitDash, 0, yy.clone().add(gT), tip, 0.035, 0.03);
      dashLine(splitDash, k, yy.clone().add(gNv), tip, 0.035, 0.03);
    }
    splitDash.m.opacity = 0.4 * sdOn;
    // step 5: descent trail, older hops fade
    trailSegs.forEach((sg, k) => {
      const part = Math.min(Math.max(gs - k, 0), 1);
      const age = gs - (k + 1);
      const o = part > 0 ? 0.9 * Math.max(0.35, 1 - Math.max(age, 0) / 9) * out : 0;
      sg.material.opacity = o; sg.visible = o > 0.003; trailDots[k].visible = o > 0.003 && k > 0;
      if (sg.visible) sg.scale.set(1, sg.userData.L * ease(0, 1, part), 1);
    });

    // ---------- HTML labels: at most four at a time, relaxed so they never overlap ----------
    const W = host.clientWidth, H = host.clientHeight;
    const want = [];
    function put(name, p3, show, dx, dy) {
      const el = labels[name];
      if (!el) return;
      tmp.copy(p3).applyMatrix4(world.matrixWorld).project(camera);
      want.push({ name, el, show, x: (tmp.x * 0.5 + 0.5) * W + dx, y: (-tmp.y * 0.5 + 0.5) * H + dy,
        w: el.offsetWidth + 6, h: el.offsetHeight + 2 });
    }
    put("f", fpt, appear * out * (1 - ease(6.4, 6.9, tc)), 0, -20);
    { const t = tc < 6.6 ? "nearest allowed point" : "output"; if (labels.y && labels.y.textContent !== t) labels.y.textContent = t; }
    {
      // the output's label sits across the push direction, clear of arrows and trail
      const a2 = toScreen(yy, W, H), b2 = toScreen(yy.clone().add(gT.lengthSq() > 1e-10 ? gT.clone().setLength(0.2) : V()), W, H);
      let ux = b2.x - a2.x, uy = b2.y - a2.y; const L = Math.hypot(ux, uy);
      if (L > 3 && tc > 7.6) {
        ux /= L; uy /= L;
        let px = -uy, py = ux; if (px > 0) { px = -px; py = -py; }
        put("y", yy, yOn, 30 * px, 22 * py);
      } else put("y", yy, yOn, -(labels.y.offsetWidth / 2 + 14), 14);
    }
    put("face", faceLabelAt, faceOn * (1 - ease(7.6, 8.1, tc)), 0, 0);
    put("best", loss.ys, lossOn * Math.max(1 - ease(9.6, 10.0, tc), ease(13.6, 14.1, tc) * (1 - ease(16.0, 16.4, tc))), 0, 16);
    put("grad", yy.clone().add(g.clone().multiplyScalar(gOn)).add(g.clone().setLength(0.12)), gOn * (1 - ease(9.9, 10.3, tc)) * out, 0, 0);
    put("pass", yy.clone().add(gT).add(gT.clone().setLength(0.13)), split * out * (1 - ease(13.4, 13.9, tc)) * Math.min(1, Math.max(0, (gT.length() - 0.1) / 0.15)), 0, 4);
    {
      const a2 = toScreen(yy, W, H), b2 = toScreen(yy.clone().addScaledVector(F.n, 0.2), W, H);
      let px = -(b2.y - a2.y), py = b2.x - a2.x; const L = Math.hypot(px, py) || 1; px /= L; py /= L;
      if (py > 0) { px = -px; py = -py; }
      put("blocked", yy.clone().addScaledVector(F.n, Math.max(gNv.length(), 0.2) * 0.7), split * (1 - ease(12.2, 12.7, tc)) * out, 34 * px, 30 * py);
    }
    put("C", V(0, -0.15, 0), appear * out * (1 - ease(3.0, 3.5, tc)), 0, 0);
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
      vis.forEach(a => { a.x = Math.min(Math.max(a.x, a.w / 2 + 2), W - a.w / 2 - 2); a.y = Math.min(Math.max(a.y, a.h / 2 + 2), H - a.h / 2 - 2); });
    }
    want.forEach(a => {
      a.el.style.transform = `translate(${a.x}px, ${a.y}px) translate(-50%, -50%)`;
      a.el.style.opacity = a.show;
    });
    dbg.labelsShown = want.filter(a => a.show > 0.3).map(a => a.name);
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
  if (prm && stepsEl) {
    stepsEl.classList.add("all");
    stepsEl.querySelector(".h3d-step-text").innerHTML = "<ol>" + STEPS.map(s => `<li>${s[2]}</li>`).join("") + "</ol>";
  }
  const t0 = performance.now();
  let cycleIdx = null;
  let qStart = new THREE.Quaternion().setFromEuler(new THREE.Euler(0.18, -0.35, 0)), qTarget = qStart.clone();
  const qBase = new THREE.Quaternion(), qUser = new THREE.Quaternion(), qSway = new THREE.Quaternion();
  world.position.x = -0.32;
  const CAM_POS = camera.position.clone(), CAM_LOOK = V(0, -0.08, 0), camTarget = CAM_LOOK.clone();
  function frame(now) {
    const T = FIXT !== null ? FIXT : prm ? 10.9 : Math.max(0, (now - t0) / 1000);
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
    // dolly in on the tight face while the loss and its gradient are shown, back out at the end
    const zoom = ease(3.6, 6.2, tc) * (1 - ease(16.6, 17.9, tc));
    // the close-up looks at the face 40 degrees off its normal, so the outward (blocked) part
    // never points straight at the viewer
    const fc = F.c.clone().applyMatrix4(world.matrixWorld);
    const nW = F.n.clone().transformDirection(world.matrixWorld);
    // tilt within the plane spanned by n and (n x passed direction): the passed part then keeps its
    // full length on screen and the blocked part appears at a right angle to it
    const side = loss.side.clone().transformDirection(world.matrixWorld);
    if (side.x > 0) side.negate();
    const camDir = nW.clone().multiplyScalar(Math.cos(0.92)).addScaledVector(side, Math.sin(0.92)).normalize();
    camTarget.copy(CAM_LOOK).lerp(fc.clone().addScaledVector(nW, 0.42), 0.9 * zoom);
    camera.position.copy(CAM_POS).lerp(fc.clone().addScaledVector(camDir, 3.4), zoom);
    camera.lookAt(camTarget);
    camera.updateMatrixWorld();
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
