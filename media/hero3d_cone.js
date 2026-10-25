// Hero scene (cone preview): the allowed region C(x) is a glass second-order (Lorentz, "ice-cream")
// cone closed by one slanted flat cut. The neural network estimate f lies outside it; a solver finds
// y = Pi_C(f), so f - y is the wall's outward normal. A quadratic loss pulls y toward the best allowed
// answer. Cycles alternate between the two kinds of wall. On the flat cut (Corollary cor:polyhedral)
// the push splits two ways: the part along the wall passes, the part across it stops. On the cone's
// side (Corollary cor:soc) it splits three ways: along the cone's straight line through the apex it
// passes, around the axis it shrinks by kappa = (radius at y)/(radius at f), outward it stops.
// Training moves f by -eta J^T grad L and y glides along the wall.
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



  // ---------- feasible set: a glass cone ----------
  // C = K ∩ {a.p <= b}: the second-order (Lorentz) cone K = {sqrt(x^2 + z^2) <= y - YA}, apex at
  // height YA, opening upward at 45 degrees, closed by one slanted flat cut. Each piece has a
  // closed-form projection.
  const YA = -0.85, APEX = V(0, YA, 0);
  const CUTN = V(0.2, 1, 0.08).normalize(), CUTB = CUTN.dot(V(0, 0.3, 0));
  const UP = V(0, 1, 0), DOWN = V(0, -1, 0), S2 = Math.SQRT1_2;
  // a point of the cone's surface: angle phi around the axis, radius r (= height above the apex)
  const onCone = (phi, r, lift) => {
    const p = V(r * Math.cos(phi), YA + r, r * Math.sin(phi));
    return lift ? p.addScaledVector(V(Math.cos(phi) * S2, -S2, Math.sin(phi) * S2), lift) : p;
  };
  const coneN = p => { const r = Math.hypot(p.x, p.z); return V(S2 * p.x / r, -S2, S2 * p.z / r); };       // outward normal
  const coneRuling = p => { const r = Math.hypot(p.x, p.z); return V(S2 * p.x / r, S2, S2 * p.z / r); };  // straight line away from the apex
  const coneAround = p => { const r = Math.hypot(p.x, p.z); return V(-p.z / r, 0, p.x / r); };
  const rimR = phi => (CUTB - CUTN.dot(APEX)) / (CUTN.x * Math.cos(phi) + CUTN.y + CUTN.z * Math.sin(phi));
  // clearance of a point on the cone's side (from the cut, and from the apex) and of a point on the
  // flat cut (its distance to the cone's surface, (t - r)/sqrt 2 for this 45 degree cone)
  const coneClear = p => Math.min(CUTB - CUTN.dot(p), p.distanceTo(APEX));
  const flatClear = p => (p.y - YA - Math.hypot(p.x, p.z)) * S2;
  const NT = 192;
  const rimLoop = [];
  for (let i = 0; i < NT; i++) { const ph = -Math.PI + 2 * Math.PI * i / NT; rimLoop.push(onCone(ph, rimR(ph))); }
  const rimC = rimLoop.reduce((s, p) => s.add(p), V()).multiplyScalar(1 / NT);
  const geo = (() => {
    const pos = [], nrm = [];
    const tri = (a, b, c, na, nb, nc) => { pos.push(a.x, a.y, a.z, b.x, b.y, b.z, c.x, c.y, c.z); nrm.push(na.x, na.y, na.z, nb.x, nb.y, nb.z, nc.x, nc.y, nc.z); };
    const NV = 12;
    for (let i = 0; i < NT; i++) {
      const p0 = -Math.PI + 2 * Math.PI * i / NT, p1 = -Math.PI + 2 * Math.PI * (i + 1) / NT;
      const r0 = rimR(p0), r1 = rimR(p1);
      const n0 = V(S2 * Math.cos(p0), -S2, S2 * Math.sin(p0)), n1 = V(S2 * Math.cos(p1), -S2, S2 * Math.sin(p1));
      for (let j = 0; j < NV; j++) {
        const a = onCone(p0, r0 * j / NV), a1 = onCone(p0, r0 * (j + 1) / NV), b = onCone(p1, r1 * j / NV), b1 = onCone(p1, r1 * (j + 1) / NV);
        tri(a, a1, b, n0, n0, n1); tri(a1, b1, b, n0, n1, n1);   // wound so the normal points out
      }
    }
    for (let i = 0; i < NT; i++) tri(rimC, rimLoop[(i + 1) % NT], rimLoop[i], CUTN, CUTN, CUTN);
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
    g.setAttribute("normal", new THREE.Float32BufferAttribute(nrm, 3));
    return g;
  })();
  const glass = { color: 0x7aaaf0, roughness: 0.2, metalness: 0, clearcoat: 0.85, clearcoatRoughness: 0.08,
    envMapIntensity: 0.7, transparent: true, depthWrite: false };
  const inner = new THREE.Mesh(geo, new THREE.MeshPhysicalMaterial(Object.assign({}, glass, { side: THREE.BackSide, opacity: 0.3, color: 0x5b90e0 })));
  const shell = new THREE.Mesh(geo, new THREE.MeshPhysicalMaterial(Object.assign({}, glass, { side: THREE.FrontSide, opacity: 0.26 })));
  // fresnel rim: glass brightens where it turns away from the viewer (this is what makes the curve read)
  const rimMesh = new THREE.Mesh(geo, new THREE.ShaderMaterial({
    uniforms: { k: { value: 0.55 } },
    vertexShader: "varying vec3 vN; varying vec3 vV; void main(){ vec4 mv = modelViewMatrix * vec4(position, 1.0); vN = normalize(normalMatrix * normal); vV = normalize(-mv.xyz); gl_Position = projectionMatrix * mv; }",
    fragmentShader: "uniform float k; varying vec3 vN; varying vec3 vV; void main(){ float f = pow(1.0 - abs(dot(normalize(vN), normalize(vV))), 2.4); gl_FragColor = vec4(1.0, 1.0, 1.0, k * f); }",
    transparent: true, depthWrite: false, side: THREE.FrontSide
  }));
  inner.renderOrder = 0; shell.renderOrder = 2; rimMesh.renderOrder = 2.2;
  world.add(inner, shell, rimMesh);
  // one shared unit cylinder, scaled per rod, so per-cycle rods allocate no GPU buffers
  const rodGeo = new THREE.CylinderGeometry(1, 1, 1, 8);
  function placeRod(o, a, b, r) {
    const d = V().subVectors(b, a), L = d.length();
    o.position.copy(a).addScaledVector(d, 0.5);
    if (L > 1e-9) o.quaternion.setFromUnitVectors(UP, d.multiplyScalar(1 / L));
    o.scale.set(r, Math.max(L, 1e-4), r);
  }
  {
    // the rim, where the flat cut meets the cone
    const em = new THREE.MeshBasicMaterial({ color: 0x1e40af, transparent: true, opacity: 0.42 });
    const t = new THREE.Mesh(new THREE.TubeGeometry(new THREE.CatmullRomCurve3(rimLoop, true, "centripetal"), NT, 0.0046, 6, true), em);
    t.renderOrder = 3; world.add(t);
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
    sh.rotation.x = -Math.PI / 2; sh.position.set(-0.2, YA - 0.2, 0); sh.scale.set(0.55, 0.42, 1);
    scene.add(sh);
  }
  // outline points for the label placer (the shape's screen hull) and the camera's fit
  const allVerts = rimLoop.filter((_, i) => i % 4 === 0).concat([APEX.clone()]);

  // ---------- exact projection ----------
  // onto the cone, the closed form of Bauschke and Combettes: inside -> itself; in the polar cone ->
  // the apex; otherwise (t + r)/(2r) (w, r) with w = (x, z), t = height above the apex, r = |w|
  const coneProj = p => {
    const t = p.y - YA, r = Math.hypot(p.x, p.z);
    if (r <= t) return p.clone();
    if (r <= -t) return APEX.clone();
    const s = (t + r) / (2 * r);
    return V(s * p.x, YA + s * r, s * p.z);
  };
  const flatProj = p => { const v = CUTN.dot(p) - CUTB; return v > 0 ? p.clone().addScaledVector(CUTN, -v) : p.clone(); };
  const BLOCKS = [flatProj, coneProj];
  const feasible = (p, tol) => Math.hypot(p.x, p.z) <= p.y - YA + tol && CUTN.dot(p) <= CUTB + tol;
  // Dykstra's algorithm over the two pieces converges to the exact Euclidean projection
  function dykstra(x0, cycles, record) {
    let x = x0.clone();
    const P = BLOCKS.map(() => V());
    const path = [x.clone()];
    for (let k = 0; k < cycles; k++) for (let i = 0; i < BLOCKS.length; i++) {
      const z = x.clone().add(P[i]), xn = BLOCKS[i](z);
      P[i].copy(z).sub(xn);
      if (record && xn.distanceTo(x) > 1e-3) path.push(xn.clone());
      x = xn;
    }
    return record ? path : x;
  }
  // closed form: a piece's own projection is the projection onto C whenever it lands in C (C is convex)
  const projC = p => {
    let q = coneProj(p); if (feasible(q, 1e-12)) return q;
    q = flatProj(p); if (feasible(q, 1e-12)) return q;
    return dykstra(p, 4000, false);
  };
  // the projection's Jacobian at an estimate f (it is symmetric), applied to v.
  // Output on the cone's side (Corollary cor:soc): along the straight line through the apex x1,
  // around the axis x kappa = (radius at the output)/(radius at f), outward x0.
  // Output on the flat cut (Corollary cor:polyhedral): the projector onto the cut's plane.
  function jacCone(f, v) {
    const y = coneProj(f), l = coneRuling(y), e = coneAround(y), kap = Math.hypot(y.x, y.z) / Math.hypot(f.x, f.z);
    return l.multiplyScalar(l.dot(v)).addScaledVector(e, kap * e.dot(v));
  }
  const jacFlat = v => v.clone().addScaledVector(CUTN, -CUTN.dot(v));

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
  // dashed segments (the residual f - y along the wall's outward normal)
  const dashGeo = new THREE.CylinderGeometry(0.008, 0.008, 1, 6); dashGeo.translate(0, 0.5, 0);
  function dashPool(n, color) {
    const m = new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0, depthWrite: false });
    const list = [];
    for (let k = 0; k < n; k++) { const d = new THREE.Mesh(dashGeo, m); d.renderOrder = 4; d.visible = false; list.push(d); world.add(d); }
    return { m, list };
  }
  function dashLine(pool, from, a, b, DL, GAP) {
    const d = V().subVectors(b, a), L = d.length(); d.normalize();
    let k = from;
    for (let s = 0; s + DL <= L + 1e-6 && k < pool.list.length; s += DL + GAP, k++) {
      const m = pool.list[k];
      m.visible = true; m.position.copy(a).addScaledVector(d, s);
      m.quaternion.setFromUnitVectors(UP, d); m.scale.set(1, DL, 1);
    }
    return k;
  }
  const resDash = dashPool(18, NAVY);
  // solver path: one navy tube per step, revealed in order, a dot at each iterate, and a bead
  const itMat = new THREE.MeshBasicMaterial({ color: NAVY, transparent: true, opacity: 0, depthWrite: false });
  const dotMat = new THREE.MeshBasicMaterial({ color: NAVY, transparent: true, opacity: 0, depthWrite: false });
  const MAXIT = 12;
  const segGeo = new THREE.CylinderGeometry(0.017, 0.017, 1, 8); segGeo.translate(0, 0.5, 0);
  const dotGeo = new THREE.SphereGeometry(0.027, 14, 10);
  const segs = [], dots = [];
  for (let k = 0; k < MAXIT; k++) {
    const sg = new THREE.Mesh(segGeo, itMat); sg.renderOrder = 4; segs.push(sg); world.add(sg);
    const d = new THREE.Mesh(dotGeo, dotMat); d.renderOrder = 4; dots.push(d); world.add(d);
  }
  const bead = new THREE.Mesh(new THREE.SphereGeometry(0.04, 18, 12), new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0 }));
  bead.renderOrder = 5; world.add(bead);
  // loss on the wall: true level sets, and the best allowed answer
  const contourMat = new THREE.LineBasicMaterial({ color: NAVY, transparent: true, opacity: 0, depthWrite: false });
  let contours = null;
  const ringMat = new THREE.MeshBasicMaterial({ color: NAVY, transparent: true, opacity: 0, depthWrite: false });
  const ring = new THREE.Mesh(new THREE.TorusGeometry(0.05, 0.009, 8, 40), ringMat);
  const ringDot = new THREE.Mesh(new THREE.SphereGeometry(0.02, 12, 8), ringMat);
  ring.renderOrder = ringDot.renderOrder = 4; world.add(ring, ringDot);
  // the training push at the output and its parts
  const SLATE = 0x334155, RED = 0xdc2626, AMBER = 0xd97706;
  const gArr = arrow(SLATE), passArr = arrow(0x1d4ed8, 0.03, 0.075), blockArr = arrow(RED, 0.023, 0.062);
  // around the axis: an arrow that follows the cone's horizontal circle, drawn from shared rods
  const arcMat = mat(AMBER, { emissive: AMBER, emissiveIntensity: 0.4 });
  const NARC = 14, arcRods = [], ghostRods = [];
  const ghostMat = new THREE.MeshBasicMaterial({ color: 0xf59e0b, transparent: true, opacity: 0, depthWrite: false });
  for (let k = 0; k < NARC; k++) {
    const a = new THREE.Mesh(rodGeo, arcMat); a.renderOrder = 4; arcRods.push(a); world.add(a);
    const g = new THREE.Mesh(rodGeo, ghostMat); g.renderOrder = 3.5; ghostRods.push(g); world.add(g);
  }
  const arcHead = new THREE.Mesh(new THREE.ConeGeometry(0.06, 0.14, 14), arcMat); arcHead.renderOrder = 4; world.add(arcHead);
  const arcEnd = V();
  // a point at arc length len from yp along the circle through yp around the axis, lifted off the glass
  function arcPt(yp, sgn, len, out) {
    const r = Math.hypot(yp.x, yp.z), th = Math.atan2(yp.z, yp.x) + sgn * len / r, rl = r + 0.016;
    return out.set(rl * Math.cos(th), yp.y - 0.016, rl * Math.sin(th));
  }
  function placeArc(yp, sgn, L, ghostL) {
    const hl = Math.min(0.13, Math.max(L, 1e-4) * 0.5), s = hl / 0.14, body = Math.max(L - hl, 1e-4);
    const pa = V(), pb = V();
    arcRods.forEach((o, k) => { arcPt(yp, sgn, body * k / NARC, pa); arcPt(yp, sgn, body * (k + 1) / NARC, pb); placeRod(o, pa, pb, 0.019); });
    arcPt(yp, sgn, body, arcEnd);
    const tan = coneAround(arcEnd).multiplyScalar(sgn);
    arcHead.position.copy(arcEnd).addScaledVector(tan, 0.07 * s);
    arcHead.quaternion.setFromUnitVectors(UP, tan);
    arcHead.scale.setScalar(Math.max(s, 1e-3));
    arcPt(yp, sgn, L, arcEnd);
    ghostRods.forEach((o, k) => { arcPt(yp, sgn, ghostL * k / NARC, pa); arcPt(yp, sgn, ghostL * (k + 1) / NARC, pb); placeRod(o, pa, pb, 0.009); });
  }
  // a red cross on the outward part, always facing the viewer
  const crossMat = new THREE.MeshBasicMaterial({ color: RED, transparent: true, opacity: 0, depthWrite: false, depthTest: false });
  const cross = new THREE.Group();
  [Math.PI / 4, -Math.PI / 4].forEach(a => {
    const b = new THREE.Mesh(rodGeo, crossMat); b.scale.set(0.014, 0.17, 0.014); b.rotation.z = a; cross.add(b);
  });
  cross.renderOrder = 6; cross.children.forEach(c => { c.renderOrder = 6; });
  world.add(cross);
  const qInv = new THREE.Quaternion();
  // the output's trail along the wall, older steps fade
  const NGD = 16;
  const trailSegs = [], trailDots = [];
  const trailGeo = new THREE.CylinderGeometry(0.017, 0.017, 1, 8); trailGeo.translate(0, 0.5, 0);
  for (let k = 0; k < NGD; k++) {
    const m = new THREE.MeshBasicMaterial({ color: 0xc2410c, transparent: true, opacity: 0, depthWrite: false });
    const sg = new THREE.Mesh(trailGeo, m); sg.renderOrder = 4; trailSegs.push(sg); world.add(sg);
    const d = new THREE.Mesh(new THREE.SphereGeometry(0.026, 12, 8), m); d.renderOrder = 4; trailDots.push(d); world.add(d);
  }

  // level sets by marching squares over a patch's parameters (a, b) in [0,1]^2; P maps them to a
  // point of the wall (or null outside it); returns line-segment end points
  function marching(P, val, NA, NB, levels) {
    const G = [], Vv = [];
    for (let i = 0; i <= NA; i++) for (let j = 0; j <= NB; j++) { const p = P(i / NA, j / NB); G.push(p); Vv.push(p ? val(p) : NaN); }
    const id = (i, j) => i * (NB + 1) + j, out = [];
    levels.forEach(c => {
      for (let i = 0; i < NA; i++) for (let j = 0; j < NB; j++) {
        const k = [id(i, j), id(i + 1, j), id(i + 1, j + 1), id(i, j + 1)];
        if (k.some(q => !G[q])) continue;
        const pts = [];
        for (let e = 0; e < 4; e++) {
          const a = k[e], b = k[(e + 1) % 4];
          if ((Vv[a] < c) !== (Vv[b] < c)) pts.push(G[a].clone().lerp(G[b], (c - Vv[a]) / (Vv[b] - Vv[a])));
        }
        if (pts.length >= 2) out.push(pts[0], pts[1]);
        if (pts.length === 4) out.push(pts[2], pts[3]);
      }
    });
    return out;
  }

  // loss L(y) = 1/2 |y - t|^2 with t outside C straight out from a point ys of the wall, so the best
  // allowed answer is ys. Cycles alternate: the answer lands on the flat cut (cycle 1, 3, ...) or on
  // the cone's curved side (cycle 2, 4, ...).
  const ETA = 0.5, DT = 0.75;
  let F, y0, f0, h, its, faceLabelAt, loss = null, kind = "flat";
  const labels = {};
  host.querySelectorAll("[data-label]").forEach(el => { labels[el.dataset.label] = el; });
  const leader = document.createElement("i");
  leader.className = "h3d-leader"; leader.setAttribute("aria-hidden", "true");
  host.appendChild(leader);
  const PHIN = Math.atan2(0.69, 0.7);  // the output's outward normal turns to face right and toward the viewer
  const sideMemo = {};
  const hz = v => Math.atan2(v.z, v.x);
  function placeFlat() {
    // two points of the cut, well inside the cone, a readable distance apart
    const ex = V(-CUTN.y, CUTN.x, 0).normalize(), ez = V().crossVectors(CUTN, ex).normalize();
    const onCut = (a, b) => rimC.clone().addScaledVector(ex, a).addScaledVector(ez, b);
    for (let tries = 0; tries < 800; tries++) {
      const Y0 = onCut((2 * rand() - 1) * 0.9, (2 * rand() - 1) * 0.9), YS = onCut((2 * rand() - 1) * 0.9, (2 * rand() - 1) * 0.9);
      const d = Y0.distanceTo(YS);
      if (flatClear(Y0) < 0.18 || flatClear(YS) < 0.15 || d < 0.5 || d > 0.8) continue;
      const hh = 0.45 + 0.15 * rand(), Fp = Y0.clone().addScaledVector(CUTN, hh), t = YS.clone().addScaledVector(CUTN, DT);
      // f <- f - eta J^T (y - t), J = projector onto the cut's plane; y = the flat piece's projection
      const Pf = [Fp.clone()], Py = [Y0.clone()], L = [0.5 * Y0.distanceToSquared(t)];
      let ok = true;
      for (let k = 0; k < NGD && ok; k++) {
        const fN = Pf[k].clone().sub(jacFlat(V().subVectors(Py[k], t)).multiplyScalar(ETA)), yN = flatProj(fN);
        const LN = 0.5 * yN.distanceToSquared(t);
        if (flatClear(yN) < 0.1 || LN > L[k] + 1e-12) ok = false;
        Pf.push(fN); Py.push(yN); L.push(LN);
      }
      if (!ok || Py[NGD].distanceTo(YS) > 0.1) continue;
      return { hh, n0: CUTN.clone(), Y0, Fp, YS, t, Pf, Py, L };
    }
    return null;
  }
  function placeCurved() {
    for (let tries = 0; tries < 800; tries++) {
      // the output on the cone's side, away from the apex and the rim; the shrink factor kappa
      // = r / (r + h / sqrt 2) lands near 0.55-0.75
      const ph0 = (2 * rand() - 1) * Math.PI, r0 = 0.5 + 0.35 * rand(), hh = 0.45 + 0.15 * rand();
      const Y0 = onCone(ph0, r0);
      if (coneClear(Y0) < 0.14 || r0 > rimR(ph0) - 0.18) continue;
      const kap = r0 / (r0 + hh * S2);
      if (kap < 0.55 || kap > 0.75) continue;
      const u0 = coneN(Y0), Fp = Y0.clone().addScaledVector(u0, hh);
      const phs = ph0 + (rand() < 0.5 ? -1 : 1) * (0.45 + 0.35 * rand()), rs = r0 + (2 * rand() - 1) * 0.25;
      const YS = onCone(phs, rs);
      if (coneClear(YS) < 0.15 || rs > rimR(phs) - 0.16 || rs < 0.35) continue;
      const t = YS.clone().addScaledVector(coneN(YS), DT);
      // the push must have a readable part in each of the three directions
      const g = V().subVectors(t, Y0), gl = g.length();
      if (g.dot(u0) < 0.3 * gl || Math.abs(g.dot(coneRuling(Y0))) < 0.3 * gl || Math.abs(g.dot(coneAround(Y0))) < 0.3 * gl) continue;
      // training through the layer: f <- f - eta J^T (y - t), y = Pi(f) on the cone's side
      const Pf = [Fp.clone()], Py = [Y0.clone()], L = [0.5 * Y0.distanceToSquared(t)];
      let ok = true;
      for (let k = 0; k < NGD && ok; k++) {
        const fN = Pf[k].clone().sub(jacCone(Pf[k], V().subVectors(Py[k], t)).multiplyScalar(ETA)), yN = coneProj(fN);
        const LN = 0.5 * yN.distanceToSquared(t);
        if (coneClear(yN) < 0.1 || Math.hypot(yN.x, yN.z) < 0.3 || !feasible(yN, 1e-12) || LN > L[k] + 1e-12) ok = false;
        Pf.push(fN); Py.push(yN); L.push(LN);
      }
      if (!ok || Py[NGD].distanceTo(YS) > 0.1) continue;
      return { hh, n0: u0, Y0, Fp, YS, t, Pf, Py, L, ph0, phs };
    }
    return null;
  }
  function newCycle(q0, ci) {
    kind = ci % 2 === 0 ? "flat" : "curved";
    const pick = kind === "flat" ? placeFlat() : placeCurved();
    if (!pick) throw new Error("hero: no placement");
    h = pick.hh; y0 = pick.Y0; f0 = pick.Fp;
    // yaw so the output's outward normal faces right and toward the viewer; a slight tilt shows the top
    const q = new THREE.Quaternion().setFromEuler(new THREE.Euler(0.16, hz(pick.n0) - PHIN, 0, "XYZ"));
    if (labels.face) labels.face.textContent = kind === "flat" ? "flat wall" : "curved wall";

    // the wall where all of this happens: the whole flat cut, or a patch of the cone's side
    if (faceMesh) { world.remove(faceMesh); faceMesh.geometry.dispose(); }
    if (faceEdges) { world.remove(faceEdges); faceEdges.geometry.dispose(); }
    let border, patchP, liftP;
    if (kind === "flat") {
      const ins = p => rimC.clone().lerp(p, 0.965).addScaledVector(CUTN, 0.005);
      border = rimLoop.filter((_, i) => i % 2 === 0).map(ins);
      const pos = [];
      for (let i = 0; i < border.length; i++) { const a = border[i], b = border[(i + 1) % border.length], c = rimC.clone().addScaledVector(CUTN, 0.005); pos.push(c.x, c.y, c.z, a.x, a.y, a.z, b.x, b.y, b.z); }
      const g = new THREE.BufferGeometry(); g.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
      faceMesh = new THREE.Mesh(g, faceMat);
      const ex = V(-CUTN.y, CUTN.x, 0).normalize(), ez = V().crossVectors(CUTN, ex).normalize();
      patchP = (a, b) => { const p = rimC.clone().addScaledVector(ex, (2 * a - 1) * 1.6).addScaledVector(ez, (2 * b - 1) * 1.6); return flatClear(p) > 0.03 ? p : null; };
      liftP = p => p.clone().addScaledVector(CUTN, 0.007);
    } else {
      const pA = Math.min(pick.ph0, pick.phs) - 0.36, pB = Math.max(pick.ph0, pick.phs) + 0.36;
      const rA = Math.max(0.26, Math.min(Math.hypot(y0.x, y0.z), Math.hypot(pick.YS.x, pick.YS.z)) - 0.28);
      const rTop = ph => Math.min(Math.max(Math.hypot(y0.x, y0.z), Math.hypot(pick.YS.x, pick.YS.z)) + 0.26, rimR(ph) - 0.04);
      const NP = 36, NQ = 10, pos = [];
      const P = (i, j) => { const ph = pA + (pB - pA) * i / NP; return onCone(ph, rA + (rTop(ph) - rA) * j / NQ, 0.004); };
      for (let i = 0; i < NP; i++) for (let j = 0; j < NQ; j++) {
        const a = P(i, j), a1 = P(i, j + 1), b = P(i + 1, j), b1 = P(i + 1, j + 1);
        pos.push(a.x, a.y, a.z, a1.x, a1.y, a1.z, b.x, b.y, b.z, a1.x, a1.y, a1.z, b1.x, b1.y, b1.z, b.x, b.y, b.z);
      }
      const g = new THREE.BufferGeometry(); g.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
      faceMesh = new THREE.Mesh(g, faceMat);
      border = [];
      for (let i = 0; i <= 24; i++) border.push(onCone(pA + (pB - pA) * i / 24, rA, 0.006));
      for (let j = 1; j <= 6; j++) border.push(onCone(pB, rA + (rTop(pB) - rA) * j / 6, 0.006));
      for (let i = 23; i >= 0; i--) { const ph = pA + (pB - pA) * i / 24; border.push(onCone(ph, rTop(ph), 0.006)); }
      for (let j = 5; j >= 1; j--) border.push(onCone(pA, rA + (rTop(pA) - rA) * j / 6, 0.006));
      patchP = (a, b) => { const ph = pA + (pB - pA) * a, r = rA + (rTop(ph) - rA) * b; return onCone(ph, r); };
      liftP = p => p.clone().addScaledVector(coneN(p), 0.007);
    }
    faceMesh.renderOrder = 2.5; world.add(faceMesh);
    faceEdges = new THREE.Mesh(new THREE.TubeGeometry(new THREE.CatmullRomCurve3(border, true, "centripetal"), 120, 0.013, 6, true), edgeMat);
    faceEdges.renderOrder = 3; world.add(faceEdges);
    const bc = border.reduce((s, p) => s.add(p), V()).multiplyScalar(1 / border.length);
    F = { n: pick.n0.clone(), c: bc, verts: border.filter((_, i) => i % 3 === 0) };
    for (const k in sideMemo) delete sideMemo[k];

    // solver steps from the estimate onto the wall
    its = dykstra(f0, 6, true).slice(0, MAXIT);
    its.push(y0.clone());
    segs.forEach((sg, k) => {
      sg.userData.on = k + 1 < its.length;
      if (!sg.userData.on) return;
      const a = its[k], b = its[k + 1], d = V().subVectors(b, a);
      sg.position.copy(a); sg.quaternion.setFromUnitVectors(UP, d.clone().normalize()); sg.userData.L = d.length();
    });
    let lastDot = its[0];
    dots.forEach((d, k) => {
      const p = its[k + 1];
      d.userData.on = k + 1 < its.length - 1 && p.distanceTo(y0) > 0.11 && p.distanceTo(lastDot) > 0.07;
      if (d.userData.on) { d.position.copy(p); lastDot = p; }
    });

    // loss, its level sets on the wall, and the training path
    loss = { t: pick.t, ys: pick.YS, P: pick.Py, Pf: pick.Pf, L: pick.L };
    loss.hP = loss.P.map((p, k) => loss.Pf[k].distanceTo(p));
    loss.cumT = [0];
    for (let k = 0; k < NGD; k++) loss.cumT.push(loss.cumT[k] + Math.pow(loss.P[k].distanceTo(loss.P[k + 1]), 0.55));
    loss.travel = V().subVectors(loss.P[NGD], loss.P[0]).normalize().multiplyScalar(0.3);
    const g0 = V().subVectors(loss.t, y0);
    loss.GS = 0.85 / g0.length();
    if (contours) { world.remove(contours); contours.geometry.dispose(); }
    {
      const val = p => 0.5 * p.distanceToSquared(loss.t), Emin = val(loss.ys), E0 = val(y0);
      const lv = [0.05, 0.2, 0.45, 0.8, 1.25, 1.85, 2.6].map(s => Emin + s * (E0 - Emin));
      const cp = marching(patchP, val, kind === "flat" ? 64 : 48, kind === "flat" ? 64 : 24, lv).map(liftP);
      contours = new THREE.LineSegments(new THREE.BufferGeometry().setFromPoints(cp), contourMat);
      contours.renderOrder = 4; world.add(contours);
    }
    const nS = kind === "flat" ? CUTN : coneN(loss.ys);
    ring.position.copy(loss.ys).addScaledVector(nS, 0.009);
    ring.quaternion.setFromUnitVectors(V(0, 0, 1), nS);
    ringDot.position.copy(ring.position);
    const tp = loss.P.map(p => p.clone().addScaledVector(kind === "flat" ? CUTN : coneN(p), 0.012));
    trailSegs.forEach((sg, k) => {
      const a = tp[k], b = tp[k + 1], d = V().subVectors(b, a);
      sg.position.copy(a); sg.quaternion.setFromUnitVectors(UP, d.clone().normalize()); sg.userData.L = d.length();
      trailDots[k].position.copy(a);
    });
    {
      // what the close-up must show: the wall, the estimate, the push and its parts, the ring;
      // and the camera's two side directions (it looks along their diagonal with the normal)
      const g = g0.clone().multiplyScalar(loss.GS), u0 = pick.n0, gOut = Math.max(g.dot(u0), 0);
      let tips, A, B, wA, wB;
      if (kind === "flat") {
        const gT = jacFlat(g), s = gT.length() > 1e-9 ? gT.clone().normalize() : V(1, 0, 0);
        tips = [y0.clone().add(gT)];
        A = V().crossVectors(CUTN, s).normalize(); B = s; wA = 0.9; wB = 0.0;
      } else {
        const l = coneRuling(y0), e = coneAround(y0), sg = g.dot(e) >= 0 ? 1 : -1;
        tips = [y0.clone().addScaledVector(l, g.dot(l)), arcPt(y0, sg, Math.abs(g.dot(e)), V())];
        A = e.clone().multiplyScalar(sg); B = l; wA = 0.9; wB = 0.45;
      }
      loss.camA = A; loss.camB = B; loss.wA = wA; loss.wB = wB;
      const ring4 = [A.clone().multiplyScalar(0.2), A.clone().multiplyScalar(-0.2), V().crossVectors(u0, A).normalize().multiplyScalar(0.18), V().crossVectors(u0, A).normalize().multiplyScalar(-0.18)].map(d => y0.clone().add(d));
      loss.key = F.verts.filter((_, i) => i % 2 === 0).concat([f0.clone(), y0.clone().add(g), y0.clone().addScaledVector(u0, gOut), loss.ys.clone()], tips);
      loss.keyPush = [y0.clone(), y0.clone().add(g)].concat(ring4);
      loss.keySplit = [y0.clone(), y0.clone().addScaledVector(u0, gOut)].concat(tips, ring4);
    }
    let far = F.verts[0], fd = -1;
    F.verts.forEach(v => { const d = v.distanceTo(f0); if (d > fd) { fd = d; far = v; } });
    faceLabelAt = kind === "flat" ? far.clone() : far.clone().addScaledVector(V().subVectors(far, F.c).normalize(), 0.1);

    // numerical checks (debug visits only: ?t=, ?perf, ?debug)
    if (!DEBUG) return q;
    const yStar = dykstra(f0, 4000, false);
    let projErrPath = 0, maxRise = -1e9, onWallErr = 0, minClear = 1e9, estOut = 1e9;
    const clear = kind === "flat" ? flatClear : coneClear;
    loss.P.forEach((p, k) => {
      projErrPath = Math.max(projErrPath, dykstra(loss.Pf[k], 4000, false).distanceTo(p));
      onWallErr = Math.max(onWallErr, kind === "flat" ? Math.abs(CUTN.dot(p) - CUTB) : Math.abs(Math.hypot(p.x, p.z) - (p.y - YA)));
      minClear = Math.min(minClear, clear(p));
      estOut = Math.min(estOut, loss.Pf[k].distanceTo(p));
      if (k) maxRise = Math.max(maxRise, loss.L[k] - loss.L[k - 1]);
    });
    // the Jacobian against central finite differences of the exact projection, at the estimate
    const J = v => kind === "flat" ? jacFlat(v) : jacCone(f0, v);
    let jacErr = 0;
    const eps = 1e-6;
    [V(1, 0, 0), V(0, 1, 0), V(0, 0, 1)].forEach(e => {
      const fd = projC(f0.clone().addScaledVector(e, eps)).sub(projC(f0.clone().addScaledVector(e, -eps))).multiplyScalar(1 / (2 * eps));
      jacErr = Math.max(jacErr, fd.distanceTo(J(e)));
    });
    const u0 = pick.n0, out0 = J(u0).length();
    const extra = kind === "flat" ? (() => { const a = V().crossVectors(CUTN, V(0, 0, 1)).normalize(); return { factorAlong: a.dot(J(a)) }; })()
      : (() => { const l = coneRuling(y0), e = coneAround(y0); return { factorRuling: l.dot(J(l)), factorAround: e.dot(J(e)), factorAroundExpected: Math.hypot(y0.x, y0.z) / Math.hypot(f0.x, f0.z) }; })();
    ['factorAlong', 'factorRuling', 'factorAround', 'factorAroundExpected'].forEach(k => { delete dbg[k]; });
    Object.assign(dbg, extra, {
      kind, h: +h.toFixed(3), hops: its.length - 1, projErr: yStar.distanceTo(y0), projErrPath, jacErr, factorOut: out0,
      lossMonotone: maxRise < 0, maxRise, lossDrop: loss.L[NGD] / loss.L[0], onWallErr, minClear, estimateDistMin: estOut,
      startClear: clear(y0), bestClear: clear(loss.ys), finalDist: loss.P[NGD].distanceTo(loss.ys)
    });
    return q;
  }
  // ---------- timeline (seconds within a cycle): five narrated steps, then a short hold ----------
  // the animation is scripted on its own clock (OLD); the viewer's clock (NEW) gives each step
  // time to read its sentence, and holds the wide shot before the loop restarts.
  const OLD = [0, 3.4, 6.6, 9.8, 13.2, 18.0, 18.8], NEW = [0, 3.5, 6.5, 9.5, 14.5, 20.0, 20.8];
  const CYCLE = NEW[6];
  function warp(t) {
    let k = 0; while (k < 5 && t >= NEW[k + 1]) k++;
    return OLD[k] + (t - NEW[k]) * (OLD[k + 1] - OLD[k]) / (NEW[k + 1] - NEW[k]);
  }
  const STEPS = [
    [NEW[0], NEW[1], "The neural network makes an estimate, outside the allowed region."],
    [NEW[1], NEW[2], "A solver finds the nearest allowed point; ASPL reads which wall it touches."],
    [NEW[2], NEW[3], "Training pushes the output toward the best allowed answer (the ring)."],
    [NEW[3], NEW[4], ""],
    [NEW[4], NEW[6], "The output glides along the wall to the best allowed answer, always allowed."],
  ];
  // step 4 depends on the wall: flat (cycles 1, 3, ...) or curved (cycles 2, 4, ...)
  const STEP4 = {
    flat: "On a flat wall, the part along it passes and the part across it stops.",
    curved: "On a curved wall: straight part passes, curved part shrinks, outward part stops.",
  };
  const stepText4 = k => k === 3 ? STEP4[kind] : STEPS[k][2];
  const stepsEl = document.getElementById("hero3d-steps");
  const stepText = stepsEl && stepsEl.querySelector(".h3d-step-text");
  const stepDots = stepsEl ? [...stepsEl.querySelectorAll(".h3d-dots i")] : [];
  // two stacked layers: the outgoing sentence fades out while the incoming one fades in
  const layers = stepText ? [0, 1].map(() => { const s = document.createElement("span"); stepText.appendChild(s); return s; }) : [];
  let shownStep = -1, shownKind = null, layer = 0;
  const stepHTML = k => `<b>${k + 1}</b>${stepText4(k)}`;
  function reserveCaption() {
    // reserve the tallest sentence's height, so the buttons below never move
    if (!stepText || stepsEl.classList.contains("all")) return;
    const s = layers[1 - layer], keep = s.innerHTML, op = s.style.opacity;
    s.style.opacity = 0;
    let hMax = 0;
    const k0 = kind;
    ["flat", "curved"].forEach(w => { kind = w; STEPS.forEach((_, k) => { s.innerHTML = stepHTML(k); hMax = Math.max(hMax, s.offsetHeight); }); });
    kind = k0;
    s.innerHTML = keep; s.style.opacity = op;
    stepText.style.minHeight = hMax + "px";
  }
  function narrate(tc) {
    if (!stepText || stepsEl.classList.contains("all")) return;
    let k = STEPS.findIndex(s => tc >= s[0] && tc < s[1]);
    if (k < 0) k = STEPS.length - 1;
    if (k !== shownStep || kind !== shownKind) {
      shownStep = k; shownKind = kind;
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
  const OUTSIDE = { f: 1, blocked: 1 };
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
    // y follows the training iterates, one eased hop per step, on the wall's surface (on the cone,
    // angle and radius are interpolated, so it never cuts through the glass); the estimate moves with it
    const u = lin(13.4, 16.6, tc) * loss.cumT[NGD];
    let k = 0; while (k < NGD - 1 && loss.cumT[k + 1] <= u) k++;
    const fr = Math.min((u - loss.cumT[k]) / (loss.cumT[k + 1] - loss.cumT[k]), 1), e = ease(0, 1, fr);
    const hh = loss.hP[k] + (loss.hP[k + 1] - loss.hP[k]) * e;
    let y;
    if (kind === "flat") y = loss.P[k].clone().lerp(loss.P[k + 1], e);
    else {
      const a = loss.P[k], b = loss.P[k + 1];
      let pa = hz(a), pb = hz(b); if (pb - pa > Math.PI) pb -= 2 * Math.PI; if (pa - pb > Math.PI) pb += 2 * Math.PI;
      y = onCone(pa + (pb - pa) * e, Math.hypot(a.x, a.z) + (Math.hypot(b.x, b.z) - Math.hypot(a.x, a.z)) * e);
    }
    return { y, f: y.clone().addScaledVector(kind === "flat" ? CUTN : coneN(y), hh), s: k + fr };
  }

  function update(tc, T, tv) {
    narrate(tv);
    const appear = ease(0.3, 0.7, tc), out = 1 - ease(17.3, 17.9, tc);
    const { y: yy, f: fpt, s: gs } = yAt(tc);
    const flat = kind === "flat";
    const uY = flat ? CUTN.clone() : coneN(yy);
    fSphere.position.copy(fpt);
    // the estimate steps back while the push is split, then returns in step 5 and moves with
    // the output (its nearest allowed point is always the output)
    const fBack = ease(13.4, 14.0, tc);
    setOpacity(fSphere.material, appear * out * Math.max(1 - ease(6.7, 7.6, tc), 0.6 * fBack));
    // step 2: the solver's steps
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
    faceMat.opacity = (0.42 + 0.16 * ease(6.8, 7.6, tc)) * faceOn;
    faceMat.depthWrite = tc > 7.6 && tc < 17.3;
    setOpacity(edgeMat, faceOn);
    ySphere.position.copy(yy);
    glow.position.copy(yy);
    const yOn = ease(5.5, 5.9, tc) * out;
    setOpacity(ySphere.material, yOn);
    glow.material.opacity = 0.22 * yOn * (0.75 + 0.25 * Math.sin(T * 4));
    glow.visible = yOn > 0;
    // straight, shortest line from the estimate to the output (along the wall's outward normal)
    const resOn = out * Math.max(ease(5.8, 6.3, tc) * (1 - ease(6.7, 7.6, tc)), 0.8 * fBack);
    resDash.list.forEach(d => { d.visible = false; });
    if (resOn > 0.01) dashLine(resDash, 0, yy.clone().addScaledVector(uY, 0.06), fpt.clone().addScaledVector(uY, -0.07), 0.045, 0.035);
    resDash.m.opacity = 0.55 * resOn;
    // step 3: level sets on the wall around the best allowed answer
    const lossOn = ease(6.8, 7.5, tc) * out;
    contourMat.opacity = 0.34 * lossOn; contours.visible = lossOn > 0.01;
    setOpacity(ringMat, 0.9 * lossOn);
    { const k = lin(16.4, 17.2, tc), bump = k > 0 && k < 1 ? Math.sin(Math.PI * k) : 0; ring.scale.setScalar(1 + 0.9 * bump); }
    // the training push at the output, then (step 4) its parts through the layer.
    // flat wall: the part along it passes, the part across it stops.
    // cone side: along the straight line through the apex it passes, around the axis it shrinks by
    // kappa = (radius at the output) / (radius at the estimate), outward it stops.
    const g = V().subVectors(loss.t, yy).multiplyScalar(loss.GS);
    const gOut = Math.max(g.dot(uY), 0);
    let passV, gAr = 0, kap = 1;
    if (flat) passV = jacFlat(g);
    else {
      const l = coneRuling(yy);
      passV = l.multiplyScalar(g.dot(l));
      gAr = g.dot(coneAround(yy));
      kap = Math.hypot(yy.x, yy.z) / Math.hypot(fpt.x, fpt.z);
    }
    const gOn = ease(7.7, 8.3, tc), split = ease(9.9, 10.4, tc), shrink = ease(10.7, 11.7, tc), xOn = ease(11.9, 12.2, tc), cut = ease(12.5, 13.0, tc);
    const partsOff = 1 - ease(13.4, 13.9, tc);
    placeArrow(gArr, yy, g, g.length() * gOn + 1e-3);
    setOpacity(gArr.userData.m, Math.min(gOn * 1.5, 1) * (1 - 0.85 * split) * (1 - cut) * out);
    placeArrow(passArr, yy, passV.lengthSq() > 1e-12 ? passV : UP, passV.length() * split + 1e-3);
    passArr.userData.m.emissiveIntensity = 0.4 + 0.5 * split * (1 - ease(12.6, 13.2, tc)) * (0.5 + 0.5 * Math.sin(T * 5));
    setOpacity(passArr.userData.m, Math.min(split * 1.5, 1) * out * partsOff);
    const arcFull = Math.abs(gAr) * split, arcNow = arcFull * (1 - (1 - kap) * shrink);
    const arcOn = flat ? 0 : Math.min(split * 1.5, 1) * out * partsOff;
    if (!flat) placeArc(yy, gAr >= 0 ? 1 : -1, arcNow + 1e-3, arcFull);
    arcMat.emissiveIntensity = 0.4 + 0.45 * Math.sin(Math.PI * shrink);
    setOpacity(arcMat, arcOn);
    ghostMat.opacity = flat ? 0 : 0.5 * Math.min(shrink * 2, 1) * out * partsOff; ghostRods.forEach(o => { o.visible = ghostMat.opacity > 0.003; });
    placeArrow(blockArr, yy, uY, gOut * split + 1e-3);
    setOpacity(blockArr.userData.m, Math.min(split * 1.5, 1) * (1 - cut) * out);
    // the outward part gets crossed out, then fades
    cross.position.copy(yy).addScaledVector(uY, 0.55 * gOut);
    qInv.copy(world.quaternion).invert();
    cross.quaternion.copy(qInv).multiply(camera.quaternion);
    cross.scale.setScalar(0.6 + 0.4 * xOn);
    crossMat.opacity = xOn * (1 - cut) * out; cross.visible = crossMat.opacity > 0.003;
    // step 5: the trail along the wall, older hops fade
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
      // the output keeps one name and one side for the whole cycle: across its direction of travel,
      // on the side away from the outward part
      const a2 = toScreen(yy, W, H), b2 = toScreen(tmp2.copy(yy).add(loss.travel), W, H), n2 = toScreen(tmp2.copy(yy).add(uY), W, H);
      let ux = b2.x - a2.x, uy = b2.y - a2.y; const L = Math.hypot(ux, uy) || 1; ux /= L; uy /= L;
      const px = -uy, py = ux;
      const nx = n2.x - a2.x, ny = n2.y - a2.y, dn = (px * nx + py * ny) / (Math.hypot(nx, ny) || 1);
      if (!sideMemo.y && yOn > 0.5 && Math.abs(dn) > 0.25) sideMemo.y = dn > 0 ? -1 : 1;
      const sg = sideMemo.y || (dn > 0 ? -1 : 1);
      const lw = labels.y.offsetWidth + 4, lh = labels.y.offsetHeight, dd = Math.abs(px) * lw / 2 + Math.abs(py) * lh / 2 + 16;
      put("y", yy, yOn, dd * sg * px, dd * sg * py, 0);
    }
    put("face", faceLabelAt, faceOn * (1 - ease(7.6, 8.1, tc)), 0, 0, 3);
    // the ring's label: in step 3 only (step 4 needs the room for its parts; in step 5 the caption names it)
    put("best", loss.ys, lossOn * (1 - ease(9.6, 9.9, tc)), 0, 16, 2);
    const wallC = toScreen(F.c, W, H);
    function beside(name, v, show, prio, toward) {
      const a2 = toScreen(yy, W, H), b2 = toScreen(yy.clone().add(v), W, H);
      let ux = b2.x - a2.x, uy = b2.y - a2.y; const L = Math.hypot(ux, uy) || 1; ux /= L; uy /= L;
      let px = -uy, py = ux;
      const mx = a2.x + 0.65 * (b2.x - a2.x), my = a2.y + 0.65 * (b2.y - a2.y);
      if (!sideMemo[name] && show > 0.04) sideMemo[name] = px * (mx - wallC.x) + py * (my - wallC.y) < 0 ? -1 : 1;
      const sgn = sideMemo[name] || 1;
      px *= toward ? -sgn : sgn; py *= toward ? -sgn : sgn;
      const along = [0.45, 0.85, 0.3, 1.05].map(f => ({ x: a2.x + f * (b2.x - a2.x), y: a2.y + f * (b2.y - a2.y) }));
      put(name, yy.clone().addScaledVector(v, 0.65), show, 12 * px, 12 * py, prio, along);
      want[want.length - 1].near = [a2, b2];
    }
    const partsShow = split * out * partsOff;
    beside("grad", g, gOn * (1 - ease(9.7, 9.95, tc)) * out, 4, true);
    beside("pass", passV.lengthSq() > 1e-6 ? passV : V(0, 0.05, 0), partsShow, 5, false);
    if (!flat) beside("around", V().subVectors(arcEnd, yy), partsShow, 6, false);
    else if (labels.around) labels.around.style.opacity = 0;
    beside("blocked", uY.clone().multiplyScalar(Math.max(gOut, 0.05)), split * (1 - ease(12.5, 12.9, tc)) * out, 7, false);
    put("C", V(0, -0.2, 0), appear * out * (1 - ease(3.0, 3.5, tc)), 0, 0, 8);
    // obstacles on screen: the patch's border, the arrows, the dashed residual, the points
    const segsS = [], ptsS = [];
    if (faceOn > 0.1) { const wv = F.verts.map(v => toScreen(v, W, H)); wv.forEach((q, i) => segsS.push([q, wv[(i + 1) % wv.length], 4, 40, 1])); }
    const yS = toScreen(yy, W, H);
    // the shape's outline (convex hull of its projected rim and apex): outside labels keep off it
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
    addSeg(yy.clone().add(passV), passArr.userData.m.opacity);
    if (arcMat.opacity > 0.15) {
      // the arc as a short polyline
      const sg = gAr >= 0 ? 1 : -1;
      let prevS = yS;
      for (let k = 1; k <= 3; k++) { const s = toScreen(arcPt(yy, sg, arcFull * k / 3, tmp2), W, H); segsS.push([prevS, s, 3]); prevS = s; }
    }
    // from step 3 on, keep the parts' paths free before they appear, so labels do not have to jump
    if (tc > 6.6) {
      segsS.push([yS, toScreen(tmp2.copy(yy).add(passV), W, H), 8, 320]);
      if (!flat) segsS.push([yS, toScreen(arcPt(yy, gAr >= 0 ? 1 : -1, Math.abs(gAr), tmp2), W, H), 8, 320]);
    }
    addSeg(yy.clone().addScaledVector(uY, gOut), blockArr.userData.m.opacity);
    if (resOn > 0.15) segsS.push([yS, toScreen(fpt, W, H), 2]);
    if (itMat.opacity > 0.15) {
      const n = Math.min(Math.ceil(drawK), its.length - 1);
      let prevS = toScreen(its[0], W, H);
      for (let k = 0; k < n; k++) { const s = toScreen(its[k + 1], W, H); segsS.push([prevS, s, 5]); prevS = s; }
    }
    if (gs > 0.05) {
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
      // debug metrics: obliqueness of the wall, its screen border, the split, content box, label boxes
      const fcW = yy.clone().applyMatrix4(world.matrixWorld), nW = uY.clone().transformDirection(world.matrixWorld);
      dbg.viewCos = V().subVectors(camera.position, fcW).normalize().dot(nW);
      dbg.wall = F.verts.map(v => toScreen(v, W, H));
      {
        const fS = toScreen(fpt, W, H); let inside = true, dmin = Infinity;
        const hp = hullS.map(hh => hh[0]);
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
        // the split on screen: pixel lengths of the three parts and the angles between them
        const pS = toScreen(yy.clone().add(passV), W, H), aS = flat ? yS : toScreen(arcEnd, W, H), oS = toScreen(yy.clone().addScaledVector(uY, gOut), W, H);
        const ang = (p, q) => { const ax = p.x - yS.x, ay = p.y - yS.y, bx = q.x - yS.x, by = q.y - yS.y; return Math.acos(Math.max(-1, Math.min(1, (ax * bx + ay * by) / (Math.hypot(ax, ay) * Math.hypot(bx, by) + 1e-9)))) * 180 / Math.PI; };
        dbg.split = { kind, passPx: Math.hypot(pS.x - yS.x, pS.y - yS.y), aroundPx: Math.hypot(aS.x - yS.x, aS.y - yS.y), outPx: Math.hypot(oS.x - yS.x, oS.y - yS.y),
          angPassAround: ang(pS, aS), angPassOut: ang(pS, oS), angAroundOut: ang(aS, oS), shrinkTo: kap };
        const inPoly = (p, poly) => { let c = false; for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) { const a = poly[i], b = poly[j]; if ((a.y > p.y) !== (b.y > p.y) && p.x < (b.x - a.x) * (p.y - a.y) / (b.y - a.y) + a.x) c = !c; } return c; };
        const hp = hullS.length ? hullS.map(hh => hh[0]) : null;
        if (hp) {
          const cutc = { top: 0, bottom: 0, left: 0, right: 0 };
          for (let x = 0; x <= W; x += 2) { if (inPoly({ x, y: 0.5 }, hp)) cutc.top += 2; if (inPoly({ x, y: H - 0.5 }, hp)) cutc.bottom += 2; }
          for (let y = 0; y <= H; y += 2) { if (inPoly({ x: 0.5, y }, hp)) cutc.left += 2; if (inPoly({ x: W - 0.5, y }, hp)) cutc.right += 2; }
          dbg.cut = cutc;
        }
      }
      const pts = allVerts.map(v => toScreen(v, W, H));
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
  const PRM_T = 13.2;  // reduced motion: a still of step 4, the three parts shown, the curved one shrunk, the outward one crossed out
  if (prm && stepsEl) {
    stepsEl.classList.add("all");
    stepText.innerHTML = "<ol>" + STEPS.map((s, k) => `<li>${k === 3 ? STEP4.flat + " " + STEP4.curved : s[2]}</li>`).join("") + "</ol>";
  }
  let clock = 0, lastNow = null, running = false, paused = false, onScreen = true, ctxLost = false, drawQueued = false;
  let cycleIdx = null;
  let qStart = new THREE.Quaternion().setFromEuler(new THREE.Euler(0.18, -0.35, 0)), qTarget = qStart.clone();
  const qBase = new THREE.Quaternion(), qUser = new THREE.Quaternion(), qSway = new THREE.Quaternion(), eul = new THREE.Euler();
  const CAM_POS = camera.position.clone(), CAM_LOOK = V(0, -0.08, 0), camTarget = CAM_LOOK.clone();
  // the close-up: camera direction from the output = outward + wA * (side A) + wB * (side B), per cycle
  const FIT = 1.3, AIM_X = 0.28, AIM_Y = 0.12, keyC = V(), keyC2 = V(), kp = V(), camDir = V(), camRight = V(), camUp = V(), aim = V();
  function draw() {
    if (ctxLost) return;
    dbg.frames = (dbg.frames || 0) + 1;
    const T = FIXT !== null ? FIXT : prm ? PRM_T : clock;
    const ci = Math.floor(T / CYCLE);
    if (ci !== cycleIdx) {
      // a frozen ?t= frame replays the earlier cycles' turns so it matches the live animation
      const from = cycleIdx === null ? 0 : cycleIdx + 1;
      for (let k = Math.max(from, ci - 12); k <= ci; k++) { qStart = qTarget.clone(); qTarget = newCycle(qStart, k); }
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
    // look from the diagonal of the push's parts at the output (outward plus the cycle's two side
    // directions): on the cone's side they appear about 120 degrees apart and equally long; on the flat
    // cut the wall is seen about 42 degrees off its normal
    const sideA = loss.camA.clone().transformDirection(world.matrixWorld), sideB = loss.camB.clone().transformDirection(world.matrixWorld);
    camDir.copy(nW).addScaledVector(sideA, loss.wA).addScaledVector(sideB, loss.wB).normalize();
    // fit: centre of the key points' bounding sphere, distance so the sphere fills the view
    // the patch view, blended into a tighter view of the output while the push is shown and split
    const fitOf = (keys, c) => {
      c.set(0, 0, 0);
      keys.forEach(p => c.add(kp.copy(p).applyMatrix4(world.matrixWorld)));
      c.multiplyScalar(1 / keys.length);
      let r = 0; keys.forEach(p => { r = Math.max(r, kp.copy(p).applyMatrix4(world.matrixWorld).distanceTo(c)); });
      return r;
    };
    const wP = ease(7.0, 8.2, tc) * (1 - ease(13.2, 14.4, tc)), wX = ease(9.7, 10.5, tc) * (1 - ease(13.2, 14.4, tc));
    const rA = fitOf(loss.key, keyC), rP = fitOf(loss.keyPush, keyC2);
    keyC.lerp(keyC2, wP);
    const rX = fitOf(loss.keySplit, keyC2);
    keyC.lerp(keyC2, wX);
    const R = (rA + (rP - rA) * wP) * (1 - wX) + rX * wX;
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

