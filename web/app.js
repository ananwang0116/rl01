"use strict";

// ---- 矩阵与四元数工具（列主序 4x4，MuJoCo Z-up 坐标系） ----
function mat4Identity() {
  const m = new Float32Array(16);
  m[0] = m[5] = m[10] = m[15] = 1;
  return m;
}

function mat4Mul(a, b) {
  const out = new Float32Array(16);
  for (let c = 0; c < 4; c++) {
    const b0 = b[c * 4], b1 = b[c * 4 + 1], b2 = b[c * 4 + 2], b3 = b[c * 4 + 3];
    out[c * 4]     = a[0] * b0 + a[4] * b1 + a[8]  * b2 + a[12] * b3;
    out[c * 4 + 1] = a[1] * b0 + a[5] * b1 + a[9]  * b2 + a[13] * b3;
    out[c * 4 + 2] = a[2] * b0 + a[6] * b1 + a[10] * b2 + a[14] * b3;
    out[c * 4 + 3] = a[3] * b0 + a[7] * b1 + a[11] * b2 + a[15] * b3;
  }
  return out;
}

function mat4FromPosQuat(pos, q, scale = null) {
  const [w, x, y, z] = q;
  const xx = x * x, yy = y * y, zz = z * z;
  const xy = x * y, xz = x * z, yz = y * z;
  const wx = w * x, wy = w * y, wz = w * z;
  const sx = scale ? scale[0] : 1;
  const sy = scale ? scale[1] : 1;
  const sz = scale ? scale[2] : 1;
  return new Float32Array([
    (1 - 2 * (yy + zz)) * sx, (2 * (xy + wz)) * sx,     (2 * (xz - wy)) * sx,     0,
    (2 * (xy - wz)) * sy,     (1 - 2 * (xx + zz)) * sy, (2 * (yz + wx)) * sy,     0,
    (2 * (xz + wy)) * sz,     (2 * (yz - wx)) * sz,     (1 - 2 * (xx + yy)) * sz, 0,
    pos[0],                   pos[1],                   pos[2],                   1,
  ]);
}

function mat4Perspective(fovy, aspect, near, far) {
  const f = 1.0 / Math.tan(fovy / 2);
  const nf = 1.0 / (near - far);
  return new Float32Array([
    f / aspect, 0, 0, 0,
    0, f, 0, 0,
    0, 0, (far + near) * nf, -1,
    0, 0, 2 * far * near * nf, 0,
  ]);
}

function mat4LookAt(eye, target, up) {
  let zx = eye[0] - target[0], zy = eye[1] - target[1], zz = eye[2] - target[2];
  let len = Math.hypot(zx, zy, zz) || 1;
  zx /= len; zy /= len; zz /= len;

  let xx = up[1] * zz - up[2] * zy;
  let xy = up[2] * zx - up[0] * zz;
  let xz = up[0] * zy - up[1] * zx;
  len = Math.hypot(xx, xy, xz) || 1;
  xx /= len; xy /= len; xz /= len;

  const yx = zy * xz - zz * xy;
  const yy = zz * xx - zx * xz;
  const yz = zx * xy - zy * xx;

  return new Float32Array([
    xx, yx, zx, 0,
    xy, yy, zy, 0,
    xz, yz, zz, 0,
    -(xx * eye[0] + xy * eye[1] + xz * eye[2]),
    -(yx * eye[0] + yy * eye[1] + yz * eye[2]),
    -(zx * eye[0] + zy * eye[1] + zz * eye[2]),
    1,
  ]);
}

// ---- WebGL2 3D 渲染器 ----
class Go2Renderer {
  constructor(canvas) {
    this.canvas = canvas;
    this.gl = canvas.getContext("webgl2", { antialias: true, alpha: true });
    if (!this.gl) throw new Error("浏览器不支持 WebGL2");
    this.meshes = [];
    this.geoms = [];
    this.stairs = [];
    this.trails = [[], [], [], []];

    // 轨道相机参数 (Z-up)
    this.cam = {
      target: [0, 0, 0.28],
      yaw: 0.72,
      pitch: 0.36,
      dist: 1.55,
      follow: true,
    };

    this._initShaders();
    this._initPrimitives();
    this._bindCameraEvents();
  }

  _compile(type, src) {
    const gl = this.gl;
    const s = gl.createShader(type);
    gl.shaderSource(s, src);
    gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {
      throw new Error(gl.getShaderInfoLog(s));
    }
    return s;
  }

  _link(vsSrc, fsSrc) {
    const gl = this.gl;
    const p = gl.createProgram();
    gl.attachShader(p, this._compile(gl.VERTEX_SHADER, vsSrc));
    gl.attachShader(p, this._compile(gl.FRAGMENT_SHADER, fsSrc));
    gl.linkProgram(p);
    if (!gl.getProgramParameter(p, gl.LINK_STATUS)) {
      throw new Error(gl.getProgramInfoLog(p));
    }
    return p;
  }

  _initShaders() {
    const vsMesh = `#version 300 es
      layout(location=0) in vec3 aPos;
      layout(location=1) in vec3 aNorm;
      uniform mat4 uVP;
      uniform mat4 uModel;
      out vec3 vNorm;
      out vec3 vWorld;
      void main() {
        vec4 w = uModel * vec4(aPos, 1.0);
        vWorld = w.xyz;
        vNorm = mat3(uModel) * aNorm;
        gl_Position = uVP * w;
      }`;

    const fsMesh = `#version 300 es
      precision highp float;
      in vec3 vNorm;
      in vec3 vWorld;
      uniform vec4 uColor;
      uniform int uIsFloor;
      out vec4 fragColor;
      void main() {
        if (uIsFloor == 1) {
          float tile = mod(floor(vWorld.x * 2.0) + floor(vWorld.y * 2.0), 2.0);
          vec3 c = mix(vec3(0.955, 0.941, 0.910), vec3(0.895, 0.875, 0.825), tile);
          if (abs(vWorld.y) < 0.012 && vWorld.x > 0.0) c = mix(c, vec3(0.145, 0.388, 0.922), 0.7);
          if (abs(vWorld.x) < 0.012 && vWorld.y > 0.0) c = mix(c, vec3(1.0, 0.353, 0.122), 0.7);
          float dist = length(vWorld.xy);
          float fade = smoothstep(28.0, 6.0, dist);
          vec3 bg = vec3(0.957, 0.945, 0.918);
          fragColor = vec4(mix(bg, c, fade), 1.0);
          return;
        }
        vec3 n = normalize(vNorm);
        vec3 l1 = normalize(vec3(0.5, 0.4, 0.9));
        vec3 l2 = normalize(vec3(-0.6, -0.5, 0.4));
        float d = max(dot(n, l1), 0.0) * 0.72 + max(dot(n, l2), 0.0) * 0.28 + 0.34;
        vec3 col = uColor.rgb * min(d, 1.12);
        fragColor = vec4(col, uColor.a);
      }`;

    this.progMesh = this._link(vsMesh, fsMesh);
    this.uVP = this.gl.getUniformLocation(this.progMesh, "uVP");
    this.uModel = this.gl.getUniformLocation(this.progMesh, "uModel");
    this.uColor = this.gl.getUniformLocation(this.progMesh, "uColor");
    this.uIsFloor = this.gl.getUniformLocation(this.progMesh, "uIsFloor");

    const vsLine = `#version 300 es
      layout(location=0) in vec3 aPos;
      uniform mat4 uVP;
      void main() { gl_Position = uVP * vec4(aPos, 1.0); }`;
    const fsLine = `#version 300 es
      precision highp float;
      uniform vec4 uLineColor;
      out vec4 fragColor;
      void main() { fragColor = uLineColor; }`;
    this.progLine = this._link(vsLine, fsLine);
    this.uLineVP = this.gl.getUniformLocation(this.progLine, "uVP");
    this.uLineColor = this.gl.getUniformLocation(this.progLine, "uLineColor");
  }

  _uploadMesh(verts, norms, indices) {
    const gl = this.gl;
    const vao = gl.createVertexArray();
    gl.bindVertexArray(vao);

    const vbo = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
    gl.bufferData(gl.ARRAY_BUFFER, verts, gl.STATIC_DRAW);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 0, 0);

    const nbo = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, nbo);
    gl.bufferData(gl.ARRAY_BUFFER, norms, gl.STATIC_DRAW);
    gl.enableVertexAttribArray(1);
    gl.vertexAttribPointer(1, 3, gl.FLOAT, false, 0, 0);

    const ibo = gl.createBuffer();
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, ibo);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, indices, gl.STATIC_DRAW);

    gl.bindVertexArray(null);
    return { vao, count: indices.length };
  }

  _initPrimitives() {
    // 地板平面
    this.floorMesh = this._uploadMesh(
      new Float32Array([-30, -30, 0, 30, -30, 0, 30, 30, 0, -30, 30, 0]),
      new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1]),
      new Uint32Array([0, 1, 2, 0, 2, 3])
    );

    // 单位立方体 ([-1, 1]^3，配合 MuJoCo 半尺寸缩放)
    const p = [
      // +Z
      -1,-1, 1,  1,-1, 1,  1, 1, 1, -1, 1, 1,
      // -Z
      -1,-1,-1, -1, 1,-1,  1, 1,-1,  1,-1,-1,
      // +X
       1,-1,-1,  1, 1,-1,  1, 1, 1,  1,-1, 1,
      // -X
      -1,-1,-1, -1,-1, 1, -1, 1, 1, -1, 1,-1,
      // +Y
      -1, 1,-1, -1, 1, 1,  1, 1, 1,  1, 1,-1,
      // -Y
      -1,-1,-1,  1,-1,-1,  1,-1, 1, -1,-1, 1,
    ];
    const n = [
      0,0,1, 0,0,1, 0,0,1, 0,0,1,
      0,0,-1, 0,0,-1, 0,0,-1, 0,0,-1,
      1,0,0, 1,0,0, 1,0,0, 1,0,0,
      -1,0,0, -1,0,0, -1,0,0, -1,0,0,
      0,1,0, 0,1,0, 0,1,0, 0,1,0,
      0,-1,0, 0,-1,0, 0,-1,0, 0,-1,0,
    ];
    const idx = [];
    for (let f = 0; f < 6; f++) {
      const b = f * 4;
      idx.push(b, b + 1, b + 2, b, b + 2, b + 3);
    }
    this.boxMesh = this._uploadMesh(new Float32Array(p), new Float32Array(n), new Uint32Array(idx));

    // 单位球体 (用于碰撞体球/胶囊)
    const sv = [], sn = [], si = [];
    const rings = 10, segs = 16;
    for (let r = 0; r <= rings; r++) {
      const th = (r / rings) * Math.PI;
      for (let s = 0; s <= segs; s++) {
        const ph = (s / segs) * Math.PI * 2;
        const x = Math.sin(th) * Math.cos(ph);
        const y = Math.sin(th) * Math.sin(ph);
        const z = Math.cos(th);
        sv.push(x, y, z);
        sn.push(x, y, z);
      }
    }
    for (let r = 0; r < rings; r++) {
      for (let s = 0; s < segs; s++) {
        const a = r * (segs + 1) + s;
        const b = a + segs + 1;
        si.push(a, b, a + 1, a + 1, b, b + 1);
      }
    }
    this.sphereMesh = this._uploadMesh(new Float32Array(sv), new Float32Array(sn), new Uint32Array(si));

    // 轨迹线缓冲
    const gl = this.gl;
    this.trailVao = gl.createVertexArray();
    this.trailVbo = gl.createBuffer();
    gl.bindVertexArray(this.trailVao);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.trailVbo);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 0, 0);
    gl.bindVertexArray(null);
  }

  loadMeshesBinary(buf) {
    const dv = new DataView(buf);
    let off = 0;
    const nMesh = dv.getUint32(off, true);
    off += 4;
    this.meshes = [];
    for (let i = 0; i < nMesh; i++) {
      const nVert = dv.getUint32(off, true);
      const nIdx = dv.getUint32(off + 4, true);
      off += 8;
      const vBytes = nVert * 3 * 4;
      const verts = new Float32Array(buf, off, nVert * 3);
      off += vBytes;
      const norms = new Float32Array(buf, off, nVert * 3);
      off += vBytes;
      const indices = new Uint32Array(buf, off, nIdx);
      off += nIdx * 4;
      this.meshes.push(this._uploadMesh(verts, norms, indices));
    }
  }

  setGeoms(geoms) {
    this.geoms = geoms.map((g) => ({
      ...g,
      localMat: mat4FromPosQuat(g.pos, g.quat),
    }));
  }

  _bindCameraEvents() {
    let dragMode = null;
    let lx = 0, ly = 0;
    this.canvas.addEventListener("contextmenu", (e) => e.preventDefault());
    this.canvas.addEventListener("pointerdown", (e) => {
      dragMode = (e.button === 2 || e.shiftKey) ? "pan" : "orbit";
      lx = e.clientX; ly = e.clientY;
      this.canvas.setPointerCapture(e.pointerId);
    });
    this.canvas.addEventListener("pointermove", (e) => {
      if (!dragMode) return;
      const dx = e.clientX - lx, dy = e.clientY - ly;
      lx = e.clientX; ly = e.clientY;
      if (dragMode === "orbit") {
        this.cam.yaw -= dx * 0.008;
        this.cam.pitch = Math.max(-0.2, Math.min(1.45, this.cam.pitch + dy * 0.007));
      } else {
        const sp = this.cam.dist * 0.0018;
        const cy = Math.cos(this.cam.yaw), sy = Math.sin(this.cam.yaw);
        // 向左拖拽(dx<0)视角向右(+x_cam)，向下拖拽(dy>0)视角向上(场景随鼠标下移)
        this.cam.target[0] += (sy * dx - cy * dy) * sp;
        this.cam.target[1] += (-cy * dx - sy * dy) * sp;
      }
    });
    this.canvas.addEventListener("pointerup", () => { dragMode = null; });
    this.canvas.addEventListener("wheel", (e) => {
      e.preventDefault();
      this.cam.dist = Math.max(0.45, Math.min(12.0, this.cam.dist * (1 + e.deltaY * 0.001)));
    }, { passive: false });
  }

  resetCamera(basePos = [0, 0, 0.28]) {
    this.cam.target = [...basePos];
    this.cam.yaw = 0.72;
    this.cam.pitch = 0.36;
    this.cam.dist = 1.55;
  }

  resize() {
    const dpr = window.devicePixelRatio || 1;
    const w = Math.max(1, Math.floor(this.canvas.clientWidth * dpr));
    const h = Math.max(1, Math.floor(this.canvas.clientHeight * dpr));
    if (this.canvas.width !== w || this.canvas.height !== h) {
      this.canvas.width = w;
      this.canvas.height = h;
    }
    this.gl.viewport(0, 0, w, h);
    return { w, h, dpr };
  }

  render(state, opts) {
    const { w, h } = this.resize();
    const gl = this.gl;
    gl.clearColor(0.957, 0.945, 0.918, 1.0);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    gl.enable(gl.DEPTH_TEST);

    if (state && this.cam.follow) {
      this.cam.target[0] += (state.base_pos[0] - this.cam.target[0]) * 0.22;
      this.cam.target[1] += (state.base_pos[1] - this.cam.target[1]) * 0.22;
      this.cam.target[2] += (state.base_pos[2] - this.cam.target[2]) * 0.12;
    }

    const { yaw, pitch, dist, target } = this.cam;
    const eye = [
      target[0] + dist * Math.cos(pitch) * Math.cos(yaw),
      target[1] + dist * Math.cos(pitch) * Math.sin(yaw),
      target[2] + dist * Math.sin(pitch),
    ];
    const proj = mat4Perspective(Math.PI / 4, w / h, 0.05, 80.0);
    const view = mat4LookAt(eye, target, [0, 0, 1]);
    const vp = mat4Mul(proj, view);

    gl.useProgram(this.progMesh);
    gl.uniformMatrix4fv(this.uVP, false, vp);

    // 1. 地板
    gl.uniform1i(this.uIsFloor, 1);
    gl.uniformMatrix4fv(this.uModel, false, mat4Identity());
    gl.uniform4f(this.uColor, 1, 1, 1, 1);
    gl.bindVertexArray(this.floorMesh.vao);
    gl.drawElements(gl.TRIANGLES, this.floorMesh.count, gl.UNSIGNED_INT, 0);

    // 2. 楼梯
    gl.uniform1i(this.uIsFloor, 0);
    for (const st of this.stairs) {
      const m = mat4FromPosQuat(st.pos, [1, 0, 0, 0], st.size);
      gl.uniformMatrix4fv(this.uModel, false, m);
      gl.uniform4fv(this.uColor, st.rgba);
      gl.bindVertexArray(this.boxMesh.vao);
      gl.drawElements(gl.TRIANGLES, this.boxMesh.count, gl.UNSIGNED_INT, 0);
    }

    // 3. 机器人模型
    if (state && state.body_poses && this.meshes.length > 0) {
      const bodyMats = state.body_poses.map((p) => mat4FromPosQuat([p[0], p[1], p[2]], [p[3], p[4], p[5], p[6]]));

      // 蓝影（参考站姿半透明模型）
      if (opts.showGhost && state.ghost_poses) {
        gl.enable(gl.BLEND);
        gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
        gl.depthMask(false);
        const ghostMats = state.ghost_poses.map((p) => mat4FromPosQuat([p[0], p[1], p[2]], [p[3], p[4], p[5], p[6]]));
        for (const g of this.geoms) {
          if (g.group !== 2 || g.mesh_id < 0 || g.mesh_id >= this.meshes.length) continue;
          const m = mat4Mul(ghostMats[g.body_id - 1], g.localMat);
          gl.uniformMatrix4fv(this.uModel, false, m);
          gl.uniform4f(this.uColor, 0.145, 0.388, 0.922, 0.28);
          const mesh = this.meshes[g.mesh_id];
          gl.bindVertexArray(mesh.vao);
          gl.drawElements(gl.TRIANGLES, mesh.count, gl.UNSIGNED_INT, 0);
        }
        gl.depthMask(true);
        gl.disable(gl.BLEND);
      }

      // 实机视觉网格
      for (const g of this.geoms) {
        if (g.group !== 2 || g.mesh_id < 0 || g.mesh_id >= this.meshes.length) continue;
        const m = mat4Mul(bodyMats[g.body_id - 1], g.localMat);
        gl.uniformMatrix4fv(this.uModel, false, m);
        gl.uniform4fv(this.uColor, g.rgba);
        const mesh = this.meshes[g.mesh_id];
        gl.bindVertexArray(mesh.vao);
        gl.drawElements(gl.TRIANGLES, mesh.count, gl.UNSIGNED_INT, 0);
      }

      // 碰撞体叠加
      if (opts.showColliders) {
        gl.enable(gl.BLEND);
        gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
        for (const g of this.geoms) {
          if (g.group !== 3) continue;
          const bMat = bodyMats[g.body_id - 1];
          let prim = this.sphereMesh;
          let scale = [g.size[0], g.size[0], g.size[0]];
          if (g.type === 6) {
            prim = this.boxMesh;
            scale = g.size;
          } else if (g.type === 5) {
            prim = this.sphereMesh;
            scale = [g.size[0], g.size[0], g.size[1] || g.size[0]];
          }
          const sm = mat4FromPosQuat(g.pos, g.quat, scale);
          const m = mat4Mul(bMat, sm);
          gl.uniformMatrix4fv(this.uModel, false, m);
          gl.uniform4f(this.uColor, 1.0, 0.353, 0.122, 0.38);
          gl.bindVertexArray(prim.vao);
          gl.drawElements(gl.TRIANGLES, prim.count, gl.UNSIGNED_INT, 0);
        }
        gl.disable(gl.BLEND);
      }

      // 轨迹线
      if (opts.showTrail && state.foot_pos) {
        gl.useProgram(this.progLine);
        gl.uniformMatrix4fv(this.uLineVP, false, vp);
        const trailColors = [
          [0.145, 0.388, 0.922, 0.85],
          [1.0, 0.353, 0.122, 0.85],
          [0.059, 0.624, 0.431, 0.85],
          [0.545, 0.361, 0.965, 0.85],
        ];
        gl.bindVertexArray(this.trailVao);
        for (let leg = 0; leg < 4; leg++) {
          const pts = this.trails[leg];
          if (pts.length < 2) continue;
          const flat = new Float32Array(pts.length * 3);
          for (let k = 0; k < pts.length; k++) {
            flat[k * 3] = pts[k][0];
            flat[k * 3 + 1] = pts[k][1];
            flat[k * 3 + 2] = pts[k][2] + 0.004;
          }
          gl.bindBuffer(gl.ARRAY_BUFFER, this.trailVbo);
          gl.bufferData(gl.ARRAY_BUFFER, flat, gl.DYNAMIC_DRAW);
          gl.uniform4fv(this.uLineColor, trailColors[leg]);
          gl.drawArrays(gl.LINE_STRIP, 0, pts.length);
        }
      }
    }

    return vp;
  }

  pushTrail(footPos) {
    if (!footPos) return;
    for (let i = 0; i < 4; i++) {
      const arr = this.trails[i];
      arr.push(footPos[i]);
      if (arr.length > 90) arr.shift();
    }
  }

  clearTrails() {
    this.trails = [[], [], [], []];
  }
}

// ---- 应用主控制器 ----
const app = {
  meta: null,
  state: null,
  ws: null,
  renderer: null,
  selectedJoint: -1,
  showColliders: false,
  showLabels: true,
  showGhost: false,
  showTrail: true,
  cmd: { vx: 0, vy: 0, wz: 0 },
  scopePaused: false,
  scopeBuf: { t: [], vx: [], cmdVx: [], roll: [], pitch: [], q: [], qTgt: [], tau: [] },
  labelEls: [],
  torqueCells: [],
  detailRows: [],
  keysDown: new Set(),
};

function sendWS(obj) {
  if (app.ws && app.ws.readyState === WebSocket.OPEN) {
    app.ws.send(JSON.stringify(obj));
  }
}

function projectPoint(p, vp, cssW, cssH) {
  const x = vp[0] * p[0] + vp[4] * p[1] + vp[8] * p[2] + vp[12];
  const y = vp[1] * p[0] + vp[5] * p[1] + vp[9] * p[2] + vp[13];
  const w = vp[3] * p[0] + vp[7] * p[1] + vp[11] * p[2] + vp[15];
  if (w <= 0.01) return null;
  const ndcX = x / w, ndcY = y / w;
  if (ndcX < -1.1 || ndcX > 1.1 || ndcY < -1.1 || ndcY > 1.1) return null;
  return [(ndcX * 0.5 + 0.5) * cssW, (-ndcY * 0.5 + 0.5) * cssH];
}

function selectJoint(idx) {
  app.selectedJoint = idx;
  const slider = document.getElementById("jogSlider");
  const title = document.getElementById("selJointName");
  if (idx >= 0 && app.meta) {
    const name = app.meta.joint_names[idx];
    const [lo, hi] = app.meta.joint_ranges[idx];
    title.textContent = `#${idx} ${name}`;
    slider.min = lo.toFixed(2);
    slider.max = hi.toFixed(2);
    slider.disabled = false;
    if (app.state) {
      slider.value = app.state.q_tgt[idx];
      document.getElementById("jogVal").textContent = app.state.q_tgt[idx].toFixed(2);
    }
  } else {
    title.textContent = "— (点 3D/标签/力矩条)";
    slider.disabled = true;
    document.getElementById("jogVal").textContent = "—";
  }
  app.labelEls.forEach((el, i) => el.classList.toggle("selected", i === idx));
  app.torqueCells.forEach((el, i) => el.classList.toggle("selected", i === idx));
}

// ---- 仪表盘绘制 (米白撞色风格) ----
function drawHorizon(rollDeg, pitchDeg) {
  const cv = document.getElementById("horizonCanvas");
  const ctx = cv.getContext("2d");
  const w = cv.width, h = cv.height, cx = w / 2, cy = h / 2, r = Math.min(w, h) / 2 - 5;
  ctx.clearRect(0, 0, w, h);

  ctx.save();
  ctx.beginPath();
  ctx.arc(cx, cy, r, 0, Math.PI * 2);
  ctx.clip();

  ctx.translate(cx, cy);
  ctx.rotate((-rollDeg * Math.PI) / 180);
  const dy = (pitchDeg / 45) * r;
  ctx.fillStyle = "#dbeafe";
  ctx.fillRect(-r * 2, -r * 2, r * 4, r * 2 + dy);
  ctx.fillStyle = "#fed7aa";
  ctx.fillRect(-r * 2, dy, r * 4, r * 2);
  ctx.strokeStyle = "#17171b";
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.moveTo(-r * 2, dy);
  ctx.lineTo(r * 2, dy);
  ctx.stroke();
  ctx.restore();

  // 中心基准十字
  ctx.strokeStyle = "#ff5a1f";
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(cx - 12, cy); ctx.lineTo(cx - 4, cy);
  ctx.moveTo(cx + 4, cy); ctx.lineTo(cx + 12, cy);
  ctx.moveTo(cx, cy - 4); ctx.lineTo(cx, cy + 4);
  ctx.stroke();

  ctx.strokeStyle = "#17171b";
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.arc(cx, cy, r, 0, Math.PI * 2);
  ctx.stroke();
}

function drawCompass(yawDeg) {
  const cv = document.getElementById("compassCanvas");
  const ctx = cv.getContext("2d");
  const w = cv.width, h = cv.height, cx = w / 2, cy = h / 2, r = Math.min(w, h) / 2 - 5;
  ctx.clearRect(0, 0, w, h);

  ctx.fillStyle = "#f4f1ea";
  ctx.beginPath();
  ctx.arc(cx, cy, r, 0, Math.PI * 2);
  ctx.fill();
  ctx.strokeStyle = "#17171b";
  ctx.lineWidth = 1.5;
  ctx.stroke();

  ctx.fillStyle = "#65635d";
  ctx.font = "bold 9px sans-serif";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText("N", cx, cy - r + 8);
  ctx.fillText("S", cx, cy + r - 8);
  ctx.fillText("W", cx - r + 8, cy);
  ctx.fillText("E", cx + r - 8, cy);

  ctx.save();
  ctx.translate(cx, cy);
  ctx.rotate((-yawDeg * Math.PI) / 180);
  ctx.fillStyle = "#2563eb";
  ctx.beginPath();
  ctx.moveTo(0, -r + 12);
  ctx.lineTo(6, 8);
  ctx.lineTo(0, 4);
  ctx.lineTo(-6, 8);
  ctx.closePath();
  ctx.fill();
  ctx.restore();
}

function drawLineChart(canvasId, seriesList, xVals, yLabel = "") {
  const cv = document.getElementById(canvasId);
  if (!cv) return;
  const ctx = cv.getContext("2d");
  const w = cv.width, h = cv.height;
  ctx.clearRect(0, 0, w, h);

  const pad = { l: 44, r: 14, t: 18, b: 26 };
  const pw = w - pad.l - pad.r;
  const ph = h - pad.t - pad.b;

  let yMin = Infinity, yMax = -Infinity;
  for (const s of seriesList) {
    for (const v of s.data) {
      if (v < yMin) yMin = v;
      if (v > yMax) yMax = v;
    }
  }
  if (!Number.isFinite(yMin) || seriesList[0].data.length === 0) {
    ctx.fillStyle = "#65635d";
    ctx.font = "12px sans-serif";
    ctx.fillText("暂无数据", w / 2 - 24, h / 2);
    return;
  }
  if (Math.abs(yMax - yMin) < 1e-4) { yMin -= 0.5; yMax += 0.5; }
  const margin = (yMax - yMin) * 0.1;
  yMin -= margin; yMax += margin;

  // 网格
  ctx.strokeStyle = "#e5dfd2";
  ctx.lineWidth = 1;
  ctx.fillStyle = "#65635d";
  ctx.font = "10px monospace";
  for (let i = 0; i <= 4; i++) {
    const yy = pad.t + (ph * i) / 4;
    ctx.beginPath();
    ctx.moveTo(pad.l, yy);
    ctx.lineTo(pad.l + pw, yy);
    ctx.stroke();
    const val = yMax - ((yMax - yMin) * i) / 4;
    ctx.fillText(val.toFixed(2), 4, yy + 3);
  }

  // 图例
  let lx = pad.l;
  ctx.font = "bold 11px sans-serif";
  for (const s of seriesList) {
    ctx.fillStyle = s.color;
    ctx.fillRect(lx, 5, 10, 4);
    ctx.fillStyle = "#17171b";
    ctx.fillText(s.label, lx + 14, 10);
    lx += ctx.measureText(s.label).width + 28;
  }
  if (yLabel) {
    ctx.fillStyle = "#65635d";
    ctx.fillText(yLabel, w - 50, 10);
  }

  // 折线
  const n = seriesList[0].data.length;
  for (const s of seriesList) {
    ctx.strokeStyle = s.color;
    ctx.lineWidth = 2;
    ctx.beginPath();
    for (let i = 0; i < n; i++) {
      const px = pad.l + (n === 1 ? pw / 2 : (i / (n - 1)) * pw);
      const py = pad.t + ((yMax - s.data[i]) / (yMax - yMin)) * ph;
      if (i === 0) ctx.moveTo(px, py);
      else ctx.lineTo(px, py);
    }
    ctx.stroke();
  }

  // X 轴首尾标签
  if (xVals && xVals.length > 0) {
    ctx.fillStyle = "#65635d";
    ctx.font = "10px monospace";
    ctx.fillText(String(xVals[0]), pad.l, h - 6);
    const endTxt = String(xVals[xVals.length - 1]);
    ctx.fillText(endTxt, pad.l + pw - ctx.measureText(endTxt).width, h - 6);
  }
}

// ---- 摇杆与遥控 ----
function setupControls() {
  // 2D 摇杆
  const pad = document.getElementById("joyPad");
  const knob = document.getElementById("joyKnob");
  let joyActive = false;
  const updateJoy = (e) => {
    const r = pad.getBoundingClientRect();
    const cx = r.left + r.width / 2, cy = r.top + r.height / 2;
    const maxR = r.width / 2 - 14;
    let dx = e.clientX - cx, dy = e.clientY - cy;
    const d = Math.hypot(dx, dy);
    if (d > maxR) { dx = (dx / d) * maxR; dy = (dy / d) * maxR; }
    knob.style.left = `${50 + (dx / (r.width / 2)) * 50}%`;
    knob.style.top = `${50 + (dy / (r.height / 2)) * 50}%`;
    const lim = parseFloat(document.getElementById("speedLimit").value);
    app.cmd.vx = (-dy / maxR) * lim;
    app.cmd.vy = (-dx / maxR) * 0.6;
    sendCmd();
  };
  pad.addEventListener("pointerdown", (e) => {
    joyActive = true;
    pad.setPointerCapture(e.pointerId);
    updateJoy(e);
  });
  pad.addEventListener("pointermove", (e) => { if (joyActive) updateJoy(e); });
  pad.addEventListener("pointerup", () => {
    joyActive = false;
    knob.style.left = "50%"; knob.style.top = "50%";
    app.cmd.vx = 0; app.cmd.vy = 0;
    sendCmd();
  });

  // Yaw 转向滑条
  const yTrack = document.getElementById("yawTrack");
  const yKnob = document.getElementById("yawKnob");
  let yawActive = false;
  const updateYaw = (e) => {
    const r = yTrack.getBoundingClientRect();
    const cx = r.left + r.width / 2;
    const maxX = r.width / 2 - 12;
    const dx = Math.max(-maxX, Math.min(maxX, e.clientX - cx));
    yKnob.style.left = `${50 + (dx / (r.width / 2)) * 50}%`;
    app.cmd.wz = (-dx / maxX) * 1.5;
    sendCmd();
  };
  yTrack.addEventListener("pointerdown", (e) => {
    yawActive = true;
    yTrack.setPointerCapture(e.pointerId);
    updateYaw(e);
  });
  yTrack.addEventListener("pointermove", (e) => { if (yawActive) updateYaw(e); });
  yTrack.addEventListener("pointerup", () => {
    yawActive = false;
    yKnob.style.left = "50%";
    app.cmd.wz = 0;
    sendCmd();
  });

  // 键盘遥控
  window.addEventListener("keydown", (e) => {
    if (["INPUT", "SELECT", "TEXTAREA"].includes(document.activeElement.tagName)) return;
    if (e.key === "r" || e.key === "R") {
      app.renderer.resetCamera(app.state ? app.state.base_pos : [0, 0, 0.28]);
      return;
    }
    const k = e.key.toLowerCase();
    if ("wasdqe".includes(k) && !app.keysDown.has(k)) {
      app.keysDown.add(k);
      applyKeyCmd();
    }
  });
  window.addEventListener("keyup", (e) => {
    const k = e.key.toLowerCase();
    if (app.keysDown.delete(k)) applyKeyCmd();
  });
}

function applyKeyCmd() {
  const lim = parseFloat(document.getElementById("speedLimit").value);
  let vx = 0, vy = 0, wz = 0;
  if (app.keysDown.has("w")) vx += lim;
  if (app.keysDown.has("s")) vx -= lim * 0.6;
  if (app.keysDown.has("a")) vy += 0.5;
  if (app.keysDown.has("d")) vy -= 0.5;
  if (app.keysDown.has("q")) wz += 1.2;
  if (app.keysDown.has("e")) wz -= 1.2;
  app.cmd = { vx, vy, wz };
  sendCmd();
}

function sendCmd() {
  document.getElementById("cmdLin").textContent =
    `vx ${app.cmd.vx.toFixed(2)} · vy ${app.cmd.vy.toFixed(2)}`;
  document.getElementById("cmdWz").textContent =
    `wz ${app.cmd.wz.toFixed(2)}`;
  sendWS({ action: "cmd", ...app.cmd });
}

// ---- UI 更新与绑定 ----
function buildStaticDOM() {
  const ckSel = document.getElementById("ckptSelect");
  ckSel.innerHTML = "";
  for (const p of app.meta.checkpoints) {
    const opt = document.createElement("option");
    opt.value = p;
    opt.textContent = p.replace(/^runs\//, "");
    if (p === app.meta.current_checkpoint) opt.selected = true;
    ckSel.appendChild(opt);
  }
  ckSel.addEventListener("change", () => sendWS({ action: "checkpoint", path: ckSel.value }));

  // 3D 标签与力矩条与详表
  const labelsLayer = document.getElementById("labelsLayer");
  const tGrid = document.getElementById("torqueGrid");
  const dBody = document.getElementById("detailBody");

  app.meta.joint_names.forEach((name, i) => {
    const lbl = document.createElement("div");
    lbl.className = "joint-label mono";
    lbl.textContent = `#${i} ${name.replace("_joint", "")}`;
    lbl.addEventListener("click", () => selectJoint(i));
    labelsLayer.appendChild(lbl);
    app.labelEls.push(lbl);

    const cell = document.createElement("div");
    cell.className = "torque-cell";
    cell.innerHTML = `<div class="torque-box"><div class="torque-mid"></div><div class="torque-bar"></div></div><div class="torque-idx mono">${i}</div>`;
    cell.addEventListener("click", () => selectJoint(i));
    tGrid.appendChild(cell);
    app.torqueCells.push(cell);

    const tr = document.createElement("tr");
    const [lo, hi] = app.meta.joint_ranges[i];
    tr.innerHTML = `<td class="mono">${i}</td><td class="mono">${name}</td><td class="mono">0</td><td class="mono">0</td><td class="mono">0</td><td class="mono">0</td><td class="mono">0</td><td class="mono">[${lo.toFixed(2)}, ${hi.toFixed(2)}]</td>`;
    tr.style.cursor = "pointer";
    tr.addEventListener("click", () => selectJoint(i));
    dBody.appendChild(tr);
    app.detailRows.push(tr);
  });

  // 地形控制
  const kindSel = document.getElementById("terrainKind");
  const stairCtrl = document.getElementById("stairControls");
  const stepH = document.getElementById("stepHSlider");
  const tread = document.getElementById("treadSlider");
  stepH.addEventListener("input", () => { document.getElementById("stepHVal").textContent = `${stepH.value}cm`; });
  tread.addEventListener("input", () => { document.getElementById("treadVal").textContent = `${tread.value}cm`; });
  const pushTerrain = () => {
    stairCtrl.style.display = kindSel.value === "stairs" ? "inline-flex" : "none";
    sendWS({
      action: "terrain",
      kind: kindSel.value,
      step_h_cm: parseFloat(stepH.value),
      n_steps: parseInt(document.getElementById("nStepsInput").value, 10) || 10,
      tread_cm: parseFloat(tread.value),
    });
  };
  kindSel.addEventListener("change", pushTerrain);
  document.getElementById("applyTerrainBtn").addEventListener("click", pushTerrain);

  // 连接与模式
  document.getElementById("btnConnect").addEventListener("click", () => sendWS({ action: "connect", running: true }));
  document.getElementById("btnDisconnect").addEventListener("click", () => sendWS({ action: "connect", running: false }));
  document.getElementById("btnStandby").addEventListener("click", () => sendWS({ action: "mode", mode: "standby" }));
  document.getElementById("btnRL").addEventListener("click", () => sendWS({ action: "mode", mode: "rl" }));
  document.getElementById("btnReset").addEventListener("click", () => {
    app.renderer.clearTrails();
    sendWS({ action: "reset" });
  });
  document.getElementById("btnRelax").addEventListener("click", () => sendWS({ action: "mode", mode: "relax" }));
  document.getElementById("btnEstop").addEventListener("click", () => sendWS({ action: "estop" }));

  // 视图选项
  document.getElementById("btnCollider").addEventListener("click", (e) => {
    app.showColliders = !app.showColliders;
    e.currentTarget.classList.toggle("btn-active", app.showColliders);
  });
  document.getElementById("btnFollow").addEventListener("click", (e) => {
    app.renderer.cam.follow = !app.renderer.cam.follow;
    e.currentTarget.classList.toggle("btn-active", app.renderer.cam.follow);
  });
  document.getElementById("btnResetCam").addEventListener("click", () => {
    app.renderer.resetCamera(app.state ? app.state.base_pos : [0, 0, 0.28]);
  });
  document.getElementById("speedMult").addEventListener("change", (e) => {
    sendWS({ action: "speed_mult", value: parseFloat(e.target.value) });
  });
  document.getElementById("chkLabels").addEventListener("change", (e) => {
    app.showLabels = e.target.checked;
    labelsLayer.style.display = app.showLabels ? "block" : "none";
  });
  document.getElementById("chkGhost").addEventListener("change", (e) => { app.showGhost = e.target.checked; });
  document.getElementById("chkTrail").addEventListener("change", (e) => { app.showTrail = e.target.checked; });

  // 负载
  ["payloadMass", "payloadX", "payloadY", "payloadZ"].forEach((id) => {
    document.getElementById(id).addEventListener("change", () => {
      sendWS({
        action: "payload",
        mass: parseFloat(document.getElementById("payloadMass").value) || 0,
        x: parseFloat(document.getElementById("payloadX").value) || 0,
        y: parseFloat(document.getElementById("payloadY").value) || 0,
        z: parseFloat(document.getElementById("payloadZ").value) || 0,
      });
    });
  });

  // 限速 & 推力
  const sl = document.getElementById("speedLimit");
  sl.addEventListener("input", () => {
    document.getElementById("speedLimitVal").textContent = `${parseFloat(sl.value).toFixed(1)} m/s`;
    sendWS({ action: "speed_limit", value: parseFloat(sl.value) });
  });
  document.querySelectorAll("[data-push]").forEach((btn) => {
    btn.addEventListener("click", () => {
      sendWS({
        action: "push",
        dir: btn.dataset.push,
        mag: parseFloat(document.getElementById("pushVel").value) || 1.0,
      });
    });
  });
  document.querySelectorAll("[data-spush]").forEach((btn) => {
    btn.addEventListener("click", () => {
      sendWS({
        action: "sustain_push",
        dir: btn.dataset.spush,
        force: parseFloat(document.getElementById("sustainForce").value) || 200,
        dur: parseFloat(document.getElementById("sustainDur").value) || 0.2,
      });
    });
  });

  // 快捷姿态 & JOG
  const pdur = document.getElementById("postureDur");
  pdur.addEventListener("input", () => {
    document.getElementById("postureDurVal").textContent = `${parseFloat(pdur.value).toFixed(1)}s`;
  });
  document.querySelectorAll("[data-posture]").forEach((btn) => {
    btn.addEventListener("click", () => {
      sendWS({ action: "posture", name: btn.dataset.posture, dur: parseFloat(pdur.value) });
    });
  });
  const jog = document.getElementById("jogSlider");
  jog.addEventListener("input", () => {
    if (app.selectedJoint < 0) return;
    const v = parseFloat(jog.value);
    document.getElementById("jogVal").textContent = v.toFixed(2);
    sendWS({ action: "jog", index: app.selectedJoint, value: v });
  });

  // 右侧页签切换
  document.querySelectorAll(".side-tab").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".side-tab").forEach((b) => b.classList.remove("active"));
      document.querySelectorAll(".tab-panel").forEach((p) => p.classList.remove("active"));
      btn.classList.add("active");
      document.getElementById(`tab-${btn.dataset.tab}`).classList.add("active");
    });
  });

  // 训练日志选择
  const logSel = document.getElementById("logSelect");
  for (const f of app.meta.logs) {
    const opt = document.createElement("option");
    opt.value = f; opt.textContent = f;
    logSel.appendChild(opt);
  }
  const loadLog = async () => {
    if (!logSel.value) return;
    const res = await fetch(`/api/log?file=${encodeURIComponent(logSel.value)}`);
    const d = await res.json();
    drawLineChart("chartReturn", [{ label: "回合回报", data: d.ep_return, color: "#2563eb" }], d.iters);
    drawLineChart("chartTrack", [
      { label: "线速度跟踪", data: d.track_lin, color: "#0f9f6e" },
      { label: "存活比例", data: d.survive, color: "#ff5a1f" },
    ], d.iters);
    drawLineChart("chartVel", [
      { label: "实际 vx", data: d.vx, color: "#2563eb" },
      { label: "指令 vx", data: d.cmd_vx, color: "#ff5a1f" },
    ], d.iters);
    drawLineChart("chartFalls", [{ label: "摔倒数", data: d.falls, color: "#e11d48" }], d.iters);
  };
  logSel.addEventListener("change", loadLog);
  if (app.meta.logs.length > 0) loadLog();

  document.getElementById("btnScopePause").addEventListener("click", (e) => {
    app.scopePaused = !app.scopePaused;
    e.currentTarget.textContent = app.scopePaused ? "恢复采样" : "暂停采样";
  });
}

function updateUI(s) {
  // 状态标签
  const modeLabel = { standby: "站立/单机", rl: "RL 运控", relax: "松电机", estop: "急停" }[s.mode] || s.mode;
  const pillConn = document.getElementById("pillConn");
  pillConn.textContent = s.running ? "仿真已连接" : "仿真已暂停";
  pillConn.className = `pill ${s.running ? "pill-green" : "pill-orange"}`;

  const pillMode = document.getElementById("pillMode");
  pillMode.textContent = `模式 ${modeLabel}`;
  pillMode.className = `pill ${s.mode === "estop" ? "pill-red" : s.mode === "rl" ? "pill-green" : ""}`;
  document.getElementById("pillFalls").textContent = `摔倒 ${s.falls}`;

  document.getElementById("btnStandby").classList.toggle("btn-active", s.mode === "standby");
  document.getElementById("btnRL").classList.toggle("btn-active", s.mode === "rl");
  document.getElementById("watermark").textContent = `weblab rl01 · 服务 ${s.uptime_h}h · t=${s.t.toFixed(1)}s`;
  document.getElementById("hudOverlay").textContent =
    `速度 ${s.vx.toFixed(2)}  目标 ${s.cmd[0].toFixed(2)}  基座 ${s.base_pos[0].toFixed(2)}, ${s.base_pos[1].toFixed(2)}, ${s.base_pos[2].toFixed(2)}`;

  // IMU 仪表
  drawHorizon(s.roll, s.pitch);
  drawCompass(s.yaw);
  document.getElementById("rpText").textContent = `roll ${s.roll.toFixed(1)}° · pitch ${s.pitch.toFixed(1)}°`;
  document.getElementById("yawText").textContent = `yaw ${s.yaw.toFixed(1)}°`;

  document.getElementById("valVx").textContent = `${s.vx.toFixed(2)} m/s`;
  document.getElementById("valVy").textContent = `${s.vy.toFixed(2)} m/s`;
  document.getElementById("valWz").textContent = `${s.wz.toFixed(2)} rad/s`;
  document.getElementById("barVx").style.width = `${Math.min(100, Math.abs(s.vx) * 50)}%`;
  document.getElementById("barVy").style.width = `${Math.min(100, Math.abs(s.vy) * 100)}%`;
  document.getElementById("barWz").style.width = `${Math.min(100, Math.abs(s.wz) * 50)}%`;

  // 选中关节
  if (app.selectedJoint >= 0) {
    const i = app.selectedJoint;
    const q = s.q[i], qt = s.q_tgt[i], tau = s.tau[i];
    document.getElementById("jActual").textContent = q.toFixed(3);
    document.getElementById("jTarget").textContent = qt.toFixed(3);
    document.getElementById("jError").textContent = (qt - q).toFixed(3);
    document.getElementById("jTorque").textContent = `${tau.toFixed(2)} N·m`;
    const jog = document.getElementById("jogSlider");
    if (document.activeElement !== jog) {
      jog.value = qt;
      document.getElementById("jogVal").textContent = qt.toFixed(2);
    }
  }

  // 力矩总览
  const lims = app.meta.torque_limits;
  for (let i = 0; i < 12; i++) {
    const ratio = Math.max(-1, Math.min(1, s.tau[i] / lims[i]));
    const bar = app.torqueCells[i].querySelector(".torque-bar");
    const pct = Math.abs(ratio) * 48;
    bar.style.height = `${Math.max(2, pct)}%`;
    if (ratio >= 0) {
      bar.style.bottom = "50%";
      bar.style.top = "auto";
      bar.classList.remove("neg");
    } else {
      bar.style.top = "50%";
      bar.style.bottom = "auto";
      bar.classList.add("neg");
    }
  }

  // 详表 (仅在打开时刷新)
  if (document.getElementById("tab-detail").classList.contains("active")) {
    for (let i = 0; i < 12; i++) {
      const cells = app.detailRows[i].children;
      cells[2].textContent = s.q[i].toFixed(3);
      cells[3].textContent = s.q_tgt[i].toFixed(3);
      cells[4].textContent = (s.q_tgt[i] - s.q[i]).toFixed(3);
      cells[5].textContent = s.dq[i].toFixed(2);
      cells[6].textContent = s.tau[i].toFixed(2);
    }
  }

  // 示波器采样与绘制
  if (!app.scopePaused) {
    const b = app.scopeBuf;
    const ji = app.selectedJoint >= 0 ? app.selectedJoint : 2;
    b.t.push(s.t.toFixed(1));
    b.vx.push(s.vx);
    b.cmdVx.push(s.cmd[0]);
    b.roll.push(s.roll);
    b.pitch.push(s.pitch);
    b.q.push(s.q[ji]);
    b.qTgt.push(s.q_tgt[ji]);
    b.tau.push(s.tau[ji]);
    if (b.t.length > 150) {
      Object.values(b).forEach((arr) => arr.shift());
    }
    if (document.getElementById("tab-scope").classList.contains("active")) {
      drawLineChart("scopeVel", [
        { label: "实际 vx", data: b.vx, color: "#2563eb" },
        { label: "指令 vx", data: b.cmdVx, color: "#ff5a1f" },
      ], b.t);
      drawLineChart("scopeAtt", [
        { label: "Roll", data: b.roll, color: "#2563eb" },
        { label: "Pitch", data: b.pitch, color: "#0f9f6e" },
      ], b.t);
      drawLineChart("scopeJointQ", [
        { label: `q[${ji}]`, data: b.q, color: "#2563eb" },
        { label: `tgt[${ji}]`, data: b.qTgt, color: "#ff5a1f" },
      ], b.t);
      drawLineChart("scopeJointTau", [
        { label: `tau[${ji}]`, data: b.tau, color: "#e11d48" },
      ], b.t);
    }
  }
}

function connectWS() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${proto}://${location.host}/ws`);
  app.ws = ws;
  ws.onmessage = (ev) => {
    const msg = JSON.parse(ev.data);
    if (msg.type === "state") {
      app.state = msg;
      app.renderer.pushTrail(msg.foot_pos);
      updateUI(msg);
    } else if (msg.type === "stairs") {
      app.renderer.stairs = msg.stairs || [];
    } else if (msg.type === "checkpoint_status" && !msg.ok) {
      alert(`加载检查点失败: ${msg.error}`);
    }
  };
  ws.onclose = () => {
    document.getElementById("pillConn").textContent = "连接断开，重连中...";
    document.getElementById("pillConn").className = "pill pill-red";
    setTimeout(connectWS, 1500);
  };
}

function renderLoop() {
  if (app.renderer) {
    const vp = app.renderer.render(app.state, {
      showColliders: app.showColliders,
      showGhost: app.showGhost,
      showTrail: app.showTrail,
    });
    if (app.state && app.showLabels && vp) {
      const cv = app.renderer.canvas;
      const cw = cv.clientWidth, ch = cv.clientHeight;
      app.state.joint_anchors.forEach((pt, i) => {
        const el = app.labelEls[i];
        const scr = projectPoint(pt, vp, cw, ch);
        if (!scr) {
          el.style.display = "none";
        } else {
          el.style.display = "block";
          el.style.left = `${scr[0]}px`;
          el.style.top = `${scr[1] - 6}px`;
        }
      });
    }
  }
  requestAnimationFrame(renderLoop);
}

async function boot() {
  app.renderer = new Go2Renderer(document.getElementById("glCanvas"));
  const [metaRes, meshRes] = await Promise.all([
    fetch("/api/init"),
    fetch("/api/meshes.bin"),
  ]);
  app.meta = await metaRes.json();
  app.renderer.loadMeshesBinary(await meshRes.arrayBuffer());
  app.renderer.setGeoms(app.meta.geoms);
  app.renderer.stairs = app.meta.stairs || [];

  buildStaticDOM();
  setupControls();
  connectWS();
  requestAnimationFrame(renderLoop);
}

boot().catch((err) => {
  console.error(err);
  alert("WebLab 初始化失败: " + err.message);
});
