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
    this.roundColliders = new Map();
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
      roundMesh: g.group === 3 && (g.type === 3 || g.type === 5) ? this._roundCollider(g) : null,
    }));
  }

  _roundCollider(g) {
    const [radius, length] = g.size;
    const key = [g.type, radius, length].join(":");
    if (this.roundColliders.has(key)) return this.roundColliders.get(key);
    // Each ring contains radius, z, radial normal and z normal.
    const rings = [];
    if (g.type === 3) {
      for (const side of [-1, 1]) {
        for (let i = 0; i <= 8; i++) {
          const a = (side < 0 ? i - 8 : i) * Math.PI / 16;
          rings.push([radius * Math.cos(a), side * length + radius * Math.sin(a), Math.cos(a), Math.sin(a)]);
        }
      }
    } else {
      rings.push([0, -length, 0, -1], [radius, -length, 0, -1], [radius, -length, 1, 0],
                 [radius, length, 1, 0], [radius, length, 0, 1], [0, length, 0, 1]);
    }
    const vertices = [], normals = [], indices = [];
    const segments = 20;
    for (const [r, z, nr, nz] of rings) {
      for (let i = 0; i < segments; i++) {
        const a = i * 2 * Math.PI / segments;
        vertices.push(r * Math.cos(a), r * Math.sin(a), z);
        normals.push(nr * Math.cos(a), nr * Math.sin(a), nz);
      }
    }
    for (let j = 0; j < rings.length - 1; j++) {
      for (let i = 0; i < segments; i++) {
        const a = j * segments + i, b = j * segments + (i + 1) % segments;
        indices.push(a, b, a + segments, b, b + segments, a + segments);
      }
    }
    const mesh = this._uploadMesh(new Float32Array(vertices), new Float32Array(normals), new Uint32Array(indices));
    this.roundColliders.set(key, mesh);
    return mesh;
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
          if (g.roundMesh) {
            prim = g.roundMesh;
            scale = [1, 1, 1];
          } else if (g.type === 6) {
            prim = this.boxMesh;
            scale = g.size;
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
  robot: new URL(location.href).searchParams.get("robot") === "go2" ? "go2" : "d1-max",
  active: false,
  reconnect: 0,
  modelAssets: new Map(),
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
    title.textContent = `#${idx} ${name}`;
    const unit = targetUnit(idx);
    document.getElementById("scopeJointTitle").textContent = unit === "rad/s"
      ? "选中轮毂 实际转速 vs 目标转速 (rad/s)" : "选中关节 实际角 vs 目标角 (rad)";
    const wheelLimit = app.meta.wheel_speed_limit ?? 15;
    const [lo, hi] = app.meta.joint_ranges[idx] ?? [-wheelLimit, wheelLimit];
    slider.disabled = false;
    slider.min = lo.toFixed(2);
    slider.max = hi.toFixed(2);
    slider.step = unit === "rad/s" ? "0.1" : "0.01";
    if (app.state) {
      slider.value = app.state.q_tgt[idx];
      document.getElementById("jogVal").textContent = app.state.q_tgt[idx].toFixed(2);
    }
  } else {
    title.textContent = "— (点 3D/标签/力矩条)";
    slider.disabled = true;
    document.getElementById("jogVal").textContent = "—";
    document.getElementById("scopeJointTitle").textContent = "选中关节 实际角 vs 目标角 (rad)";
  }
  for (const key of ["q", "qTgt", "tau"]) app.scopeBuf[key] = Array(app.scopeBuf.t.length).fill(null);
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
      if (!Number.isFinite(v)) continue;
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
    let joined = false;
    for (let i = 0; i < n; i++) {
      if (!Number.isFinite(s.data[i])) { joined = false; continue; }
      const px = pad.l + (n === 1 ? pw / 2 : (i / (n - 1)) * pw);
      const py = pad.t + ((yMax - s.data[i]) / (yMax - yMin)) * ph;
      if (!joined) ctx.moveTo(px, py);
      else ctx.lineTo(px, py);
      joined = true;
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
const D1_TURN_HOLD_MS = 160;

function setupControls() {
  const pad = document.getElementById("joyPad");
  const knob = document.getElementById("joyKnob");
  const yTrack = document.getElementById("yawTrack");
  const yKnob = document.getElementById("yawKnob");
  let joyPointer = null, yawPointer = null;
  const pendingTurns = new Map();
  app.pointerCmd = { x: 0, y: 0, yaw: 0 };

  const endJoy = () => {
    joyPointer = null;
    knob.style.left = "50%"; knob.style.top = "50%";
    app.pointerCmd.x = 0; app.pointerCmd.y = 0;
    applyKeyCmd();
  };
  const endYaw = () => {
    yawPointer = null;
    yKnob.style.left = "50%";
    app.pointerCmd.yaw = 0;
    applyKeyCmd();
  };
  const updateJoy = (e) => {
    const r = pad.getBoundingClientRect();
    const maxR = r.width / 2 - 14;
    let dx = e.clientX - r.left - r.width / 2;
    let dy = e.clientY - r.top - r.height / 2;
    const d = Math.hypot(dx, dy);
    if (d > maxR) { dx *= maxR / d; dy *= maxR / d; }
    knob.style.left = `${50 + dx / (r.width / 2) * 50}%`;
    knob.style.top = `${50 + dy / (r.height / 2) * 50}%`;
    app.pointerCmd.x = -dy / maxR;
    app.pointerCmd.y = -dx / maxR;
    applyKeyCmd();
  };
  pad.addEventListener("pointerdown", (e) => {
    if (joyPointer !== null) return;
    joyPointer = e.pointerId;
    pad.setPointerCapture(e.pointerId);
    e.preventDefault();
    updateJoy(e);
  });
  pad.addEventListener("pointermove", (e) => { if (e.pointerId === joyPointer) updateJoy(e); });
  for (const event of ["pointerup", "pointercancel", "lostpointercapture"]) {
    pad.addEventListener(event, (e) => { if (e.pointerId === joyPointer) endJoy(); });
  }

  const updateYaw = (e) => {
    const r = yTrack.getBoundingClientRect();
    const maxX = r.width / 2 - 12;
    const dx = Math.max(-maxX, Math.min(maxX, e.clientX - r.left - r.width / 2));
    yKnob.style.left = `${50 + dx / (r.width / 2) * 50}%`;
    app.pointerCmd.yaw = -dx / maxX;
    applyKeyCmd();
  };
  yTrack.addEventListener("pointerdown", (e) => {
    if (yawPointer !== null) return;
    yawPointer = e.pointerId;
    yTrack.setPointerCapture(e.pointerId);
    e.preventDefault();
    updateYaw(e);
  });
  yTrack.addEventListener("pointermove", (e) => { if (e.pointerId === yawPointer) updateYaw(e); });
  for (const event of ["pointerup", "pointercancel", "lostpointercapture"]) {
    yTrack.addEventListener(event, (e) => { if (e.pointerId === yawPointer) endYaw(); });
  }

  app.stopControls = () => {
    for (const {timer} of pendingTurns.values()) clearTimeout(timer);
    pendingTurns.clear();
    app.keysDown.clear();
    joyPointer = null; yawPointer = null;
    app.pointerCmd = { x: 0, y: 0, yaw: 0 };
    knob.style.left = knob.style.top = yKnob.style.left = "50%";
    applyKeyCmd();
  };
  const stop = () => {
    app.stopControls();
    if (isD1()) sendWS({action: "stop"});
  };
  const editable = (target) => target && (
    ["SELECT", "TEXTAREA"].includes(target.tagName) || target.isContentEditable ||
    (target.tagName === "INPUT" && !["range", "checkbox", "radio", "button"].includes(target.type))
  );
  const controlKey = (e) => {
    if (/^Key[WASDQER]$/.test(e.code)) return e.code.slice(3).toLowerCase();
    return e.key.toLowerCase();
  };
  window.addEventListener("keydown", (e) => {
    if (!app.active || editable(e.target) || e.altKey || e.ctrlKey || e.metaKey) return;
    const k = controlKey(e);
    if (k === " ") {
      e.preventDefault();
      stop();
    } else if (k === "r") {
      resetArenaCamera();
    } else if (["w", "a", "s", "d", "q", "e"].includes(k)) {
      e.preventDefault();
      if (isD1() && (k === "q" || k === "e")) {
        if (e.repeat || pendingTurns.has(k) || app.keysDown.has(k)) return;
        const deadline = performance.now() + D1_TURN_HOLD_MS;
        const timer = setTimeout(() => {
          if (!pendingTurns.delete(k)) return;
          app.keysDown.add(k);
          applyKeyCmd();
        }, D1_TURN_HOLD_MS);
        pendingTurns.set(k, {deadline, timer});
        return;
      }
      if (!app.keysDown.has(k)) {
        app.keysDown.add(k);
        applyKeyCmd();
      }
    }
  });
  window.addEventListener("keyup", (e) => {
    const k = controlKey(e);
    const pending = pendingTurns.get(k);
    if (pending) {
      e.preventDefault();
      pendingTurns.delete(k);
      clearTimeout(pending.timer);
      if (performance.now() < pending.deadline) tapD1Turn(k === "q" ? 1 : -1);
    } else if (app.keysDown.delete(k)) {
      e.preventDefault();
      applyKeyCmd();
    }
  });
  window.addEventListener("blur", app.stopControls);
  document.addEventListener("visibilitychange", () => { if (document.hidden) app.stopControls(); });
  document.addEventListener("focusin", (e) => { if (editable(e.target)) app.stopControls(); });
  document.querySelectorAll('#app select, #app input[type="range"]').forEach((control) => {
    control.addEventListener("change", () => control.blur());
  });
  document.getElementById("btnStop").addEventListener("click", stop);
}

function tapD1Turn(direction) {
  if (!app.active || !isD1() || !app.state?.running || app.state.mode !== "rl" || app.state.tap_active) return;
  app.stopControls();
  sendWS({action: "tap", direction, angle_deg: 2});
}

function applyKeyCmd() {
  if (!app.active) return;
  const lim = parseFloat(document.getElementById("speedLimit").value);
  const keys = app.keysDown;
  const x = Number(keys.has("w")) - Number(keys.has("s"));
  const y = Number(keys.has("a")) - Number(keys.has("d"));
  const norm = Math.max(1, Math.hypot(x, y));
  const pointer = app.pointerCmd || { x: 0, y: 0, yaw: 0 };
  const keyboardXY = ["w", "s", "a", "d"].some((k) => keys.has(k));
  const keyboardYaw = keys.has("q") || keys.has("e");
  const lateralLimit = isD1() ? app.meta.limits[1] : 0.6;
  const reverse = isD1() ? Math.min(lim, app.meta.reverse_limit ?? lim) : lim * 0.6;
  const vx = keyboardXY ? x * (x < 0 ? reverse : lim) / norm
    : pointer.x * (isD1() && pointer.x < 0 ? reverse : lim);
  const vy = (keyboardXY ? y / norm : pointer.y) * Math.min(lim, lateralLimit);
  const yawInput = keyboardYaw ? Number(keys.has("q")) - Number(keys.has("e")) : pointer.yaw;
  const config = turnConfig();
  const speed = Math.hypot(vx, vy);
  let yawCap = isD1() ? app.meta.limits[2] : 1.2;
  if (isD1() && app.meta.limits[0] >= 3.5) {
    const fast = Math.max(0, Math.min(1, (Math.abs(vx) - 2.0) / 1.5));
    yawCap = (1 - fast) * yawCap + fast * 0.26;
  }
  const yawLimit = Math.min(yawCap, speed > 1e-6 ? speed / config.turn_radius : config.inplace_yaw_rate);
  app.cmd = { vx, vy, wz: yawInput * yawLimit };
  sendCmd();
}

function turnConfig() {
  const radius = parseFloat(document.getElementById("turnRadius").value);
  const rate = parseFloat(document.getElementById("inplaceYawRate").value);
  return {
    turn_radius: Math.max(0.25, Math.min(5, Number.isFinite(radius) ? radius : 1)),
    inplace_yaw_rate: Math.max(0.3, Math.min(1.2, Number.isFinite(rate) ? rate : 0.8)),
  };
}

function sendCmd() {
  document.getElementById("cmdLin").textContent =
    `vx ${app.cmd.vx.toFixed(2)} · vy ${app.cmd.vy.toFixed(2)}`;
  document.getElementById("cmdWz").textContent =
    `wz ${app.cmd.wz.toFixed(2)}`;
  sendWS({ action: "cmd", ...app.cmd, ...turnConfig() });
}

// ---- UI 更新与绑定 ----
function isD1() { return app.robot === "d1-max"; }

function targetUnit(i) {
  return app.meta?.target_types?.[i] === "velocity" ? "rad/s" : "rad";
}

function targetActual(state, i) {
  return targetUnit(i) === "rad/s" ? state.dq[i] : state.q[i];
}

function buildRobotDOM() {
  const { meta } = app;
  const ckSel = document.getElementById("ckptSelect");
  ckSel.replaceChildren(...meta.checkpoints.map((path) => new Option(
    path.replace(/^runs\//, ""), path, false, path === meta.current_checkpoint,
  )));
  ckSel.disabled = !meta.checkpoints.length;
  const n = meta.joint_names.length;
  document.getElementById("turnKeyHint").textContent = isD1()
    ? "Q/E 轻点与按钮相同 · 长按持续转向" : "转向 ←左转 / 右转→";
  document.getElementById("detailTitle").textContent = `${n} 关节实时状态详表`;
  document.getElementById("detailTargetTitle").textContent = isD1() ? "控制目标" : "目标角 (rad)";
  document.getElementById("detailErrorTitle").textContent = isD1() ? "跟踪误差" : "跟踪误差 (rad)";

  const labelsLayer = document.getElementById("labelsLayer");
  const tGrid = document.getElementById("torqueGrid");
  const dBody = document.getElementById("detailBody");
  labelsLayer.replaceChildren();
  tGrid.replaceChildren();
  dBody.replaceChildren();
  tGrid.style.setProperty("--joint-count", n);
  app.labelEls = [];
  app.torqueCells = [];
  app.detailRows = [];
  meta.joint_names.forEach((name, i) => {
    const lbl = document.createElement("div");
    lbl.className = "joint-label mono";
    lbl.textContent = `#${i} ${name.replace(/_joint$/i, "")}`;
    lbl.addEventListener("click", () => selectJoint(i));
    labelsLayer.appendChild(lbl);
    app.labelEls.push(lbl);

    const cell = document.createElement("div");
    cell.className = "torque-cell";
    const limit = meta.torque_limits[i];
    cell.title = `${name} · 仿真力矩上限 ${limit} N·m`;
    cell.innerHTML = `<div class="torque-box"><div class="torque-mid"></div><div class="torque-bar"></div></div><div class="torque-idx mono">${i}</div>`;
    cell.addEventListener("click", () => selectJoint(i));
    tGrid.appendChild(cell);
    app.torqueCells.push(cell);

    const tr = document.createElement("tr");
    const range = meta.joint_ranges[i];
    const bound = range ? `[${range[0].toFixed(2)}, ${range[1].toFixed(2)}] rad` : "连续旋转";
    for (const value of [i, name, "—", "—", "—", "—", "—", bound]) {
      const cell = tr.insertCell();
      cell.className = "mono";
      cell.textContent = value;
    }
    tr.style.cursor = "pointer";
    tr.addEventListener("click", () => selectJoint(i));
    dBody.appendChild(tr);
    app.detailRows.push(tr);
  });
  selectJoint(-1);
  for (const id of ["jActual", "jTarget", "jError", "jTorque"]) document.getElementById(id).textContent = "—";

  const logSel = document.getElementById("logSelect");
  logSel.replaceChildren(...meta.logs.map((file) => new Option(file, file)));
  document.getElementById("chartReturnTitle").textContent = isD1() ? "每步平均奖励" : "回合回报 (ep_return)";
  document.getElementById("chartTrackTitle").textContent = isD1() ? "前向 / 横向速度平均绝对误差 (m/s)" : "线速度跟踪 & 存活率";
  document.getElementById("chartVelTitle").textContent = isD1() ? "平均净空 (mm) · 机身倾角 (°)" : "实际 vx vs 指令 vx (m/s)";
  document.getElementById("chartFallsTitle").textContent = "摔倒次数 (falls)";
  for (const id of ["chartReturn", "chartTrack", "chartVel", "chartFalls", "scopeVel", "scopeAtt", "scopeJointQ", "scopeJointTau"]) {
    const canvas = document.getElementById(id);
    canvas.getContext("2d").clearRect(0, 0, canvas.width, canvas.height);
  }
  void loadTrainingLog();
}

function buildStaticDOM() {
  const labelsLayer = document.getElementById("labelsLayer");
  const ckSel = document.getElementById("ckptSelect");
  ckSel.addEventListener("change", () => {
    app.stopControls?.();
    sendWS({ action: "checkpoint", path: ckSel.value });
    ckSel.blur();
  });

  // 地形控制
  const kindSel = document.getElementById("terrainKind");
  const stairCtrl = document.getElementById("stairControls");
  const stepH = document.getElementById("stepHSlider");
  const tread = document.getElementById("treadSlider");
  stepH.addEventListener("input", () => { document.getElementById("stepHVal").textContent = `${stepH.value}cm`; });
  tread.addEventListener("input", () => { document.getElementById("treadVal").textContent = `${tread.value}cm`; });
  const pushTerrain = () => {
    if (!app.active) return;
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
    app.stopControls?.();
    app.renderer.clearTrails();
    sendWS({ action: "reset" });
  });
  document.getElementById("btnRelax").addEventListener("click", () => sendWS({ action: "mode", mode: "relax" }));
  document.getElementById("btnEstop").addEventListener("click", () => {
    app.stopControls?.();
    sendWS({ action: "estop" });
  });

  // 视图选项
  document.getElementById("btnCollider").addEventListener("click", (e) => {
    app.showColliders = !app.showColliders;
    e.currentTarget.classList.toggle("btn-active", app.showColliders);
  });
  document.getElementById("btnFollow").addEventListener("click", (e) => {
    app.renderer.cam.follow = !app.renderer.cam.follow;
    e.currentTarget.classList.toggle("btn-active", app.renderer.cam.follow);
  });
  document.getElementById("btnResetCam").addEventListener("click", resetArenaCamera);
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
    document.getElementById("speedLimitVal").textContent = `${parseFloat(sl.value).toFixed(isD1() ? 2 : 1)} m/s`;
    sendWS({ action: "speed_limit", value: parseFloat(sl.value) });
    applyKeyCmd();
  });
  sl.addEventListener("change", () => sl.blur());
  for (const id of ["turnRadius", "inplaceYawRate"]) {
    const input = document.getElementById(id);
    input.addEventListener("input", applyKeyCmd);
    input.addEventListener("change", () => {
      const config = turnConfig();
      input.value = id === "turnRadius" ? config.turn_radius : config.inplace_yaw_rate;
      applyKeyCmd();
      input.blur();
    });
  }
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

  document.getElementById("logSelect").addEventListener("change", loadTrainingLog);
  document.getElementById("d1Push").addEventListener("click", () => {
    sendWS({action: "push", angle: Number(document.getElementById("d1PushDirection").value)});
  });
  for (const button of document.querySelectorAll("[data-tap]")) {
    button.addEventListener("click", () => tapD1Turn(Number(button.dataset.tap)));
  }

  document.getElementById("btnScopePause").addEventListener("click", (e) => {
    app.scopePaused = !app.scopePaused;
    e.currentTarget.textContent = app.scopePaused ? "恢复采样" : "暂停采样";
  });
}

async function loadTrainingLog() {
  const file = document.getElementById("logSelect").value;
  if (!file) return;
  const robot = app.robot;
  const d1 = isD1();
  try {
    const response = await fetch(d1 ? "/d1-max/metrics" : `/api/log?file=${encodeURIComponent(file)}`);
    if (!response.ok) return;
    const d = await response.json();
    if (robot !== app.robot || file !== document.getElementById("logSelect").value) return;
    const iters = d1 ? d.iteration : d.iters;
    const series = (label, data, color) => ({label, data, color});
    drawLineChart("chartReturn", [series(d1 ? "每步奖励" : "回合回报", d1 ? d.reward : d.ep_return, "#2563eb")], iters);
    drawLineChart("chartTrack", d1 ? [
      series("前向 MAE", d.vx_mae, "#0f9f6e"), series("横向 MAE", d.vy_mae, "#ff5a1f"),
    ] : [
      series("线速度跟踪", d.track_lin, "#0f9f6e"), series("存活比例", d.survive, "#ff5a1f"),
    ], iters);
    drawLineChart("chartVel", d1 ? [
      series("轮脚净空 (mm)", d.lift_mm, "#2563eb"), series("机身倾角 (°)", d.tilt_deg, "#ff5a1f"),
    ] : [
      series("实际 vx", d.vx, "#2563eb"), series("指令 vx", d.cmd_vx, "#ff5a1f"),
    ], iters);
    drawLineChart("chartFalls", [series("摔倒数", d.falls, "#e11d48")], iters);
  } catch (err) {
    console.error("训练曲线加载失败", err);
  }
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

  const policy = s.policy_info || {};
  const heading = document.getElementById("headingStatus");
  heading.textContent = isD1()
    ? (app.renderer.stairs.length ? "平地策略 · 台阶供仿真测试"
      : app.meta.champ_gait ? "CHAMP 对角步 · 平地试训" : "轮足运控 · 支持点按转向")
    : policy.heading_hold
    ? `航向保持${policy.gait_phase_obs ? " · 步态相位" : ""} · 偏差 ${(s.heading_error || 0).toFixed(1)}°`
    : "旧策略 · 无航向误差输入";
  document.getElementById("policyCommandHint").textContent = isD1()
    ? `训练上限 前 ${app.meta.limits[0]} / 后 ${app.meta.reverse_limit} / 侧 ${app.meta.limits[1]} m/s · 转 ${app.meta.limits[2]} rad/s`
    : policy.omni
      ? `训练上限 前${(policy.vx_max || 0).toFixed(1)} / 后${(policy.reverse_max || 0).toFixed(1)} / 侧${(policy.vy_range || 0).toFixed(1)} m/s · 转${(policy.yaw_range || 0).toFixed(1)} rad/s`
      : "此模型训练以向前为主，其他方向需要补训练";
  if (isD1()) {
    document.getElementById("d1Lift").textContent = `轮脚净空 ${s.tap_peak_mm.toFixed(1)} mm`;
    document.querySelectorAll("[data-tap]").forEach((button) => {
      button.disabled = s.tap_active || !s.running || s.mode !== "rl";
    });
  }

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
    const q = targetActual(s, i), qt = s.q_tgt[i], tau = s.tau[i];
    const unit = targetUnit(i);
    document.getElementById("jActual").textContent = `${q.toFixed(3)} ${unit}`;
    document.getElementById("jTarget").textContent = `${qt.toFixed(3)} ${unit}`;
    document.getElementById("jError").textContent = `${(qt - q).toFixed(3)} ${unit}`;
    document.getElementById("jTorque").textContent = `${tau.toFixed(2)} N·m`;
    const jog = document.getElementById("jogSlider");
    if (document.activeElement !== jog) {
      jog.value = qt;
      document.getElementById("jogVal").textContent = qt.toFixed(2);
    }
  }

  // 力矩总览
  const lims = app.meta.torque_limits;
  for (let i = 0; i < lims.length; i++) {
    const ratio = Math.max(-1, Math.min(1, s.tau[i] / lims[i]));
    const bar = app.torqueCells[i].querySelector(".torque-bar");
    app.torqueCells[i].title = `${app.meta.joint_names[i]} · ${s.tau[i].toFixed(2)} / ±${lims[i]} N·m`;
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
    for (let i = 0; i < app.detailRows.length; i++) {
      const cells = app.detailRows[i].children;
      cells[2].textContent = s.q[i].toFixed(3);
      const unit = targetUnit(i);
      const suffix = isD1() ? ` ${unit}` : "";
      cells[3].textContent = s.q_tgt[i].toFixed(3) + suffix;
      cells[4].textContent = (s.q_tgt[i] - targetActual(s, i)).toFixed(3) + suffix;
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
    b.q.push(targetActual(s, ji));
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
        { label: `实际 [${ji}]`, data: b.q, color: "#2563eb" },
        { label: `目标 [${ji}]`, data: b.qTgt, color: "#ff5a1f" },
      ], b.t);
      drawLineChart("scopeJointTau", [
        { label: `tau[${ji}]`, data: b.tau, color: "#e11d48" },
      ], b.t);
    }
  }
}

function connectWS() {
  if (!app.active || app.ws) return;
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const endpoint = isD1() ? "/d1-max/ws" : "/ws";
  const ws = new WebSocket(`${proto}://${location.host}${endpoint}`);
  app.ws = ws;
  ws.onopen = () => {
    if (app.ws !== ws) return;
    app.stopControls?.();
    sendWS({action: "speed_mult", value: Number(document.getElementById("speedMult").value)});
  };
  ws.onmessage = (ev) => {
    if (app.ws !== ws) return;
    const msg = JSON.parse(ev.data);
    if (msg.type === "state") {
      app.state = msg;
      app.renderer.pushTrail(msg.foot_pos);
      updateUI(msg);
    } else if (msg.type === "stairs") {
      const previousStairs = app.renderer.stairs.length;
      app.renderer.stairs = msg.stairs || [];
      if (msg.terrain) syncTerrainControls(msg.terrain);
      if (isD1() && (previousStairs || app.renderer.stairs.length)) resetArenaCamera();
    } else if (msg.type === "checkpoint_status") {
      if (!msg.ok) {
        alert(`加载检查点失败: ${msg.error}`);
      } else if (isD1()) {
        app.meta.current_checkpoint = msg.checkpoint;
        app.meta.limits = msg.limits;
        app.meta.reverse_limit = msg.reverse_limit;
        app.meta.wheel_speed_limit = msg.wheel_speed_limit;
        app.meta.champ_gait = msg.champ_gait;
        app.meta.logs = msg.logs || [];
        const logSel = document.getElementById("logSelect");
        logSel.replaceChildren(...app.meta.logs.map((file) => new Option(file, file)));
        void loadTrainingLog();
      }
    }
  };
  ws.onclose = () => {
    if (app.ws !== ws) return;
    app.ws = null;
    app.stopControls?.();
    document.getElementById("pillConn").textContent = "连接断开，重连中...";
    document.getElementById("pillConn").className = "pill pill-red";
    app.reconnect = setTimeout(connectWS, 1500);
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

async function activateRobot(robot) {
  const selector = document.getElementById("robotSelect");
  const connection = document.getElementById("pillConn");
  app.stopControls?.();
  app.active = false;
  clearTimeout(app.reconnect);
  const ws = app.ws;
  app.ws = null;
  ws?.close();
  app.robot = robot;
  app.state = null;
  app.renderer.meshes = [];
  app.renderer.stairs = [];
  app.renderer.clearTrails();
  Object.values(app.scopeBuf).forEach((series) => { series.length = 0; });
  document.getElementById("app").dataset.robot = robot;
  document.getElementById("labelsLayer").replaceChildren();
  document.getElementById("torqueGrid").replaceChildren();
  document.getElementById("detailBody").replaceChildren();
  document.getElementById("ckptSelect").replaceChildren();
  connection.textContent = "正在加载模型";
  connection.className = "pill";
  selector.value = robot;
  selector.disabled = true;

  try {
    let asset = app.modelAssets.get(robot);
    if (!asset) {
      const prefix = isD1() ? "/d1-max" : "/api";
      const [metaRes, meshRes] = await Promise.all([
        fetch(`${prefix}/init`), fetch(`${prefix}/meshes.bin`),
      ]);
      if (!metaRes.ok || !meshRes.ok) throw new Error("模型或检查点无法加载");
      const meta = await metaRes.json();
      app.renderer.loadMeshesBinary(await meshRes.arrayBuffer());
      asset = {meta, meshes: app.renderer.meshes};
      app.modelAssets.set(robot, asset);
    }
    app.meta = asset.meta;
    app.renderer.meshes = asset.meshes;
    app.renderer.setGeoms(app.meta.geoms);
    app.renderer.stairs = app.meta.stairs || [];
    syncTerrainControls({kind: "flat", step_h_cm: 10, n_steps: 10, tread_cm: 25});
    app.renderer.resetCamera(isD1() ? [0, 0, 0.55] : [0, 0, 0.28]);
    if (isD1()) app.renderer.cam.dist = 2.05;

    const controls = app.meta.control_config || {};
    document.getElementById("turnRadius").value = controls.turn_radius ?? 1;
    document.getElementById("inplaceYawRate").value = isD1() ? app.meta.limits[2] : controls.inplace_yaw_rate ?? 0.8;
    const speed = document.getElementById("speedLimit");
    speed.min = isD1() ? 0.1 : 0.3;
    speed.max = isD1() ? app.meta.limits[0] : 2.2;
    speed.step = isD1() ? 0.05 : 0.1;
    speed.value = isD1() ? 0.45 : 1.2;
    document.getElementById("speedLimitVal").textContent = `${Number(speed.value).toFixed(isD1() ? 2 : 1)} m/s`;

    buildRobotDOM();
    app.active = true;
    connectWS();

    const url = new URL(location.href);
    if (isD1()) url.searchParams.delete("robot");
    else url.searchParams.set("robot", robot);
    history.replaceState(null, "", url);
  } catch (err) {
    console.error(err);
    connection.textContent = err.message;
    connection.className = "pill pill-red";
  } finally {
    selector.disabled = false;
  }
}

function syncTerrainControls(terrain) {
  document.getElementById("terrainKind").value = terrain.kind;
  document.getElementById("stairControls").style.display = terrain.kind === "stairs" ? "inline-flex" : "none";
  document.getElementById("stepHSlider").value = terrain.step_h_cm;
  document.getElementById("stepHVal").textContent = `${terrain.step_h_cm}cm`;
  document.getElementById("nStepsInput").value = terrain.n_steps;
  document.getElementById("treadSlider").value = terrain.tread_cm;
  document.getElementById("treadVal").textContent = `${terrain.tread_cm}cm`;
}

function resetArenaCamera() {
  const renderer = app.renderer;
  const top = isD1() ? renderer.stairs.at(-1) : null;
  renderer.resetCamera(app.state?.base_pos ?? [0, 0, isD1() ? 0.55 : 0.28]);
  if (top) {
    const end = top.pos[0] + top.size[0];
    renderer.cam.target = [end / 2, 0, Math.max(0.32, top.pos[2])];
    renderer.cam.yaw = -1.2;
    renderer.cam.pitch = 0.5;
    renderer.cam.dist = Math.max(2.7, end * 1.08, top.pos[2] * 3.4);
    renderer.cam.follow = false;
    document.getElementById("btnFollow").classList.remove("btn-active");
  } else if (isD1()) {
    renderer.cam.dist = 2.05;
  }
}

function boot() {
  app.renderer = new Go2Renderer(document.getElementById("glCanvas"));
  buildStaticDOM();
  setupControls();
  document.getElementById("robotSelect").addEventListener("change", (event) => {
    event.currentTarget.blur();
    void activateRobot(event.currentTarget.value);
  });
  setInterval(() => {
    if (app.active && isD1() && Object.values(app.cmd).some((value) => value !== 0)) sendCmd();
  }, 100);
  requestAnimationFrame(renderLoop);
  return activateRobot(app.robot);
}

void boot();
