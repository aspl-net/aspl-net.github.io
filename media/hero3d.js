// Hero scene: a prediction f(x) outside a 3-D feasible set C(x) is projected onto it.
// Solver iterates zig-zag onto the closest face, the face lights up (tight face), the
// projection y shows its free (blue) and blocked (orange) directions, then training
// moves y along the face. Cycles over faces; drag to rotate.
(function () {
  const host = document.getElementById("hero3d");
  if (!host || !window.THREE) return;
  let renderer;
  try {
    renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  } catch (e) { return; }
  host.classList.add("has-3d");
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.setClearColor(0x000000, 0);
  host.appendChild(renderer.domElement);

  const BLUE = 0x2563eb, LBLUE = 0x93c5fd, NAVY = 0x1e3a8a, ORANGE = 0xea580c;
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(32, 1, 0.1, 50);
  camera.position.set(0, 0.8, 5.5);
  camera.lookAt(0, 0, 0);
  scene.add(new THREE.AmbientLight(0xffffff, 0.75));
  const sun = new THREE.DirectionalLight(0xffffff, 0.55);
  sun.position.set(3, 5, 4);
  scene.add(sun);

  const world = new THREE.Group();
  scene.add(world);

  // ---------- feasible set: an irregular dodecahedron ----------
  const geo = new THREE.DodecahedronGeometry(1.15, 0);
  geo.scale(1.25, 0.95, 1.05);
  geo.rotateZ(0.18);
  geo.computeVertexNormals();
  const body = new THREE.Mesh(geo, new THREE.MeshPhongMaterial({
    color: 0x7fb2f5, transparent: true, opacity: 0.42, flatShading: true,
    shininess: 40, depthWrite: false, side: THREE.DoubleSide
  }));
  world.add(body);
  world.add(new THREE.LineSegments(new THREE.EdgesGeometry(geo, 1),
    new THREE.LineBasicMaterial({ color: NAVY, transparent: true, opacity: 0.75 })));

  // group triangles into planar faces (normal, centroid, ordered vertices)
  const pos = geo.attributes.position;
  const faces = [];
  for (let i = 0; i < pos.count; i += 3) {
    const a = new THREE.Vector3().fromBufferAttribute(pos, i);
    const b = new THREE.Vector3().fromBufferAttribute(pos, i + 1);
    const c = new THREE.Vector3().fromBufferAttribute(pos, i + 2);
    const n = new THREE.Vector3().subVectors(b, a).cross(new THREE.Vector3().subVectors(c, a)).normalize();
    let f = faces.find(F => F.n.dot(n) > 0.999 && Math.abs(F.n.dot(a) - F.d) < 1e-4);
    if (!f) { f = { n, d: n.dot(a), verts: [] }; faces.push(f); }
    [a, b, c].forEach(v => { if (!f.verts.some(w => w.distanceTo(v) < 1e-5)) f.verts.push(v); });
  }
  faces.forEach(F => {
    F.c = F.verts.reduce((s, v) => s.add(v), new THREE.Vector3()).multiplyScalar(1 / F.verts.length);
    F.t1 = new THREE.Vector3().subVectors(F.verts[0], F.c).normalize();
    F.t2 = new THREE.Vector3().crossVectors(F.n, F.t1).normalize();
    F.verts.sort((p, q) => {
      const ap = Math.atan2(new THREE.Vector3().subVectors(p, F.c).dot(F.t2), new THREE.Vector3().subVectors(p, F.c).dot(F.t1));
      const aq = Math.atan2(new THREE.Vector3().subVectors(q, F.c).dot(F.t2), new THREE.Vector3().subVectors(q, F.c).dot(F.t1));
      return ap - aq;
    });
    F.r = Math.min(...F.verts.map(v => v.distanceTo(F.c))) * Math.cos(Math.PI / 5); // inradius
  });

  let CLabelAt = new THREE.Vector3();
  { let lo = null; for (let i = 0; i < pos.count; i++) { const v = new THREE.Vector3().fromBufferAttribute(pos, i); if (!lo || v.y - 0.3 * v.x < lo.y - 0.3 * lo.x) lo = v; } CLabelAt = lo.clone().add(new THREE.Vector3(-0.25, -0.22, 0)); }

  // ---------- helpers ----------
  const fadeMats = [];
  function mat(color, extra) {
    const m = new THREE.MeshPhongMaterial(Object.assign({ color, transparent: true, opacity: 0, shininess: 60 }, extra || {}));
    fadeMats.push(m);
    return m;
  }
  function rod(a, b, r, m) {
    const d = new THREE.Vector3().subVectors(b, a), L = d.length();
    const g = new THREE.CylinderGeometry(r, r, L, 10);
    const o = new THREE.Mesh(g, m);
    o.position.copy(a).addScaledVector(d, 0.5);
    o.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), d.clone().normalize());
    return o;
  }
  function arrow(color) {
    const m = mat(color);
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
    g.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir.clone().normalize());
    g.userData.shaft.scale.set(1, Math.max(L - 0.12, 1e-3), 1);
    g.userData.shaft.position.set(0, Math.max(L - 0.12, 1e-3) / 2, 0);
    g.userData.head.position.set(0, L - 0.06, 0);
  }

  // ---------- per-cycle objects ----------
  const faceMat = new THREE.MeshBasicMaterial({ color: ORANGE, transparent: true, opacity: 0, side: THREE.DoubleSide, depthWrite: false });
  const edgeMat = mat(ORANGE, { emissive: 0x7c2d12 });
  let faceMesh = null, faceEdges = null;
  const fSphere = new THREE.Mesh(new THREE.SphereGeometry(0.075, 24, 16), mat(BLUE, { emissive: 0x0b2a6b }));
  const ySphere = new THREE.Mesh(new THREE.SphereGeometry(0.07, 24, 16), mat(ORANGE, { emissive: 0x7c2d12 }));
  const glow = new THREE.Mesh(new THREE.SphereGeometry(0.16, 24, 16), new THREE.MeshBasicMaterial({ color: ORANGE, transparent: true, opacity: 0, depthWrite: false }));
  world.add(fSphere, ySphere, glow);
  const itMat = mat(LBLUE, { emissive: 0x1e40af });
  const itSpheres = [];
  for (let k = 0; k < 8; k++) { const s = new THREE.Mesh(new THREE.SphereGeometry(0.032, 14, 10), itMat); itSpheres.push(s); world.add(s); }
  const itLine = new THREE.Line(new THREE.BufferGeometry(), new THREE.LineBasicMaterial({ color: 0x60a5fa, transparent: true, opacity: 0.95 }));
  world.add(itLine);
  const free1 = arrow(BLUE), free2 = arrow(BLUE), blocked = arrow(ORANGE);

  let F, y0, fpt, its, faceLabelAt;
  const labels = {};
  host.querySelectorAll("[data-label]").forEach(el => { labels[el.dataset.label] = el; });

  let lastFace = -1;
  function pickFace() {
    // the face that best faces the camera now, not the previous one
    const view = new THREE.Vector3().subVectors(camera.position, new THREE.Vector3()).normalize();
    const q = world.quaternion;
    let best = -1, bestS = -2;
    faces.forEach((G, i) => {
      if (i === lastFace) return;
      const nW = G.n.clone().applyQuaternion(q);
      // oblique faces read best: the normal (blocked direction) is visible, the face is not edge-on
      const s = -Math.abs(nW.dot(view) - 0.5) - 0.35 * Math.abs(nW.y) - (Math.abs(nW.y) > 0.55 ? 2 : 0) - (nW.dot(view) < 0.15 ? 2 : 0) + 0.12 * Math.random();
      if (s > bestS) { bestS = s; best = i; }
    });
    lastFace = best;
    return faces[best];
  }

  function newCycle() {
    F = pickFace();
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

    const ang = Math.random() * Math.PI * 2, rr = 0.25 * F.r;
    y0 = F.c.clone().addScaledVector(F.t1, rr * Math.cos(ang)).addScaledVector(F.t2, rr * Math.sin(ang));
    const h = 0.9 + 0.2 * Math.random();
    // f(x) starts off the face and above it on screen, so the zig-zag stays in view
    const upL = new THREE.Vector3(0, 1, 0).applyQuaternion(world.quaternion.clone().invert());
    const tUp = upL.clone().addScaledVector(F.n, -upL.dot(F.n)).normalize();
    const tSide = new THREE.Vector3().crossVectors(F.n, tUp).normalize();
    const side = Math.random() < 0.5 ? 1 : -1;
    fpt = y0.clone().addScaledVector(F.n, h).addScaledVector(tUp, 0.7 * F.r).addScaledVector(tSide, side * 0.5 * F.r);
    its = [];
    for (let k = 0; k < 8; k++) {
      const r = Math.pow(0.55, k);
      const p = y0.clone().addScaledVector(new THREE.Vector3().subVectors(fpt, y0), r)
        .addScaledVector(F.t2, (k % 2 ? 1 : -1) * 0.22 * F.r * r * (k > 0 ? 1 : 0));
      its.push(p);
    }
    its.push(y0.clone());
    itLine.geometry.dispose();
    itLine.geometry = new THREE.BufferGeometry().setFromPoints(its);
    itLine.geometry.setDrawRange(0, 0);
    fSphere.position.copy(fpt);
    // label just outside the face, at the vertex farthest from where f(x) comes in
    let far = F.verts[0], fd = -1;
    F.verts.forEach(v => { const d = v.distanceTo(fpt); if (d > fd) { fd = d; far = v; } });
    faceLabelAt = far.clone().addScaledVector(new THREE.Vector3().subVectors(far, F.c).normalize(), 0.28);
    itSpheres.forEach((s, k) => s.position.copy(its[k + 1] || y0));
  }

  // ---------- timeline ----------
  const CYCLE = 10.0;
  const ease = (a, b, t) => Math.min(Math.max((t - a) / (b - a), 0), 1);
  function setOpacity(m, o) { m.opacity = o; m.visible = o > 0.003; }
  const tmp = new THREE.Vector3();

  function update(tc, T) {
    const appear = ease(0.0, 0.5, tc), out = 1 - ease(CYCLE - 0.8, CYCLE, tc);
    // f(x)
    setOpacity(fSphere.material, appear * out);
    // iterates
    const drawK = ease(0.6, 3.0, tc) * (its.length - 1);
    itLine.geometry.setDrawRange(0, Math.floor(drawK) + 1);
    itLine.material.opacity = 0.95 * out * (1 - 0.6 * ease(4.0, 5.0, tc));
    itLine.visible = drawK > 0;
    itSpheres.forEach((s, k) => { s.visible = drawK >= k + 1; });
    setOpacity(itMat, out * (1 - 0.6 * ease(4.0, 5.0, tc)));
    // tight face
    const faceOn = ease(2.6, 3.3, tc) * out;
    faceMat.opacity = 0.42 * faceOn;
    setOpacity(edgeMat, faceOn);
    // y and its slide along the face (training)
    const slide = ease(5.0, CYCLE - 1.2, tc);
    const th = slide * Math.PI * 2;
    const yy = y0.clone().addScaledVector(F.t1, 0.22 * F.r * Math.sin(th)).addScaledVector(F.t2, 0.18 * F.r * Math.sin(2 * th));
    ySphere.position.copy(yy);
    glow.position.copy(yy);
    const yOn = ease(3.0, 3.4, tc) * out;
    setOpacity(ySphere.material, yOn);
    glow.material.opacity = 0.22 * yOn * (0.75 + 0.25 * Math.sin(T * 4));
    glow.visible = yOn > 0;
    // free / blocked directions at y
    const arr = ease(3.6, 4.5, tc);
    placeArrow(free1, yy, F.t1, 0.55 * arr + 1e-3);
    placeArrow(free2, yy, F.t2, 0.45 * arr + 1e-3);
    placeArrow(blocked, yy, F.n, 0.6 * arr + 1e-3);
    [free1, free2, blocked].forEach(a => setOpacity(a.userData.m, Math.min(arr * 1.5, 1) * out));

    // ---------- HTML labels ----------
    const W = host.clientWidth, H = host.clientHeight;
    function put(name, p3, show, dx, dy) {
      const el = labels[name];
      if (!el) return;
      tmp.copy(p3).applyMatrix4(world.matrixWorld).project(camera);
      el.style.transform = `translate(${(tmp.x * 0.5 + 0.5) * W + (dx || 0)}px, ${(-tmp.y * 0.5 + 0.5) * H + (dy || 0)}px) translate(-50%, -50%)`;
      el.style.opacity = show;
    }
    put("f", fpt, appear * out, 0, -22);
    put("y", yy, yOn, -16, 14);
    put("face", faceLabelAt, faceOn, 0, 0);
    put("free", yy.clone().addScaledVector(F.t1, 0.7), Math.min(arr * 1.5, 1) * out, 0, 0);
    put("blocked", yy.clone().addScaledVector(F.n, 0.78), Math.min(arr * 1.5, 1) * out, 0, 0);
    put("C", CLabelAt, 1, 0, 0);
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

  const reduce = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const t0 = performance.now();
  let cycleIdx = null;
  function frame(now) {
    const T = reduce ? 6.5 : Math.max(0, (now - t0) / 1000);
    const ci = Math.floor(T / CYCLE);
    const sway = reduce ? 0 : 0.32 * Math.sin(T * 0.22);
    world.rotation.set(0.18 + pitchUser, -0.35 + sway + yawUser, 0);
    world.updateMatrixWorld();
    if (ci !== cycleIdx) { cycleIdx = ci; newCycle(); }
    update(T - ci * CYCLE, T);
    renderer.render(scene, camera);
    if (!reduce) requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
})();
