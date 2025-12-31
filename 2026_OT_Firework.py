import time
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import moderngl
import moderngl_window as mglw

try:
    from zoneinfo import ZoneInfo
except Exception:
    ZoneInfo = None


# ---------------------------
# Text -> point cloud
# ---------------------------
_FONT_CANDIDATES = [
    "DejaVuSans-Bold.ttf",
    "DejaVuSans.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/Library/Fonts/Arial.ttf",
    "C:/Windows/Fonts/arialbd.ttf",
    "C:/Windows/Fonts/arial.ttf",
]


def load_font(size: int) -> ImageFont.FreeTypeFont:
    for p in _FONT_CANDIDATES:
        try:
            return ImageFont.truetype(p, size)
        except Exception:
            pass
    return ImageFont.load_default()


def text_points(text: str, n: int, img_size: int = 1024, pad: int = 80, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = Image.new("L", (img_size, img_size), 0)
    draw = ImageDraw.Draw(img)

    font = load_font(int(img_size * 0.38))
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = (img_size - tw) // 2 - bbox[0]
    y = (img_size - th) // 2 - bbox[1]
    draw.text((x, y), text, 255, font=font)

    arr = np.array(img)
    ys, xs = np.where(arr > 20)
    if len(xs) < n:
        raise RuntimeError(f"Not enough pixels for '{text}'. Increase img_size or reduce n.")

    idx = rng.integers(0, len(xs), size=n)
    px = xs[idx].astype(np.float32)
    py = ys[idx].astype(np.float32)

    px += rng.random(n).astype(np.float32) - 0.5
    py += rng.random(n).astype(np.float32) - 0.5

    px = (px - img_size * 0.5) / (img_size * 0.5 - pad)
    py = -(py - img_size * 0.5) / (img_size * 0.5 - pad)

    pts = np.stack([px, py], axis=1)
    return np.clip(pts, -1.2, 1.2)


# ---------------------------
# Sliced OT: rank-based 1:1 matching
# ---------------------------
def sliced_ot_perm_map(X: np.ndarray, Y: np.ndarray, n_proj: int = 64, seed: int = 0) -> np.ndarray:
    assert X.shape == Y.shape and X.shape[1] == 2
    N = X.shape[0]
    rng = np.random.default_rng(seed)

    scoreX = np.zeros(N, dtype=np.float32)
    scoreY = np.zeros(N, dtype=np.float32)

    for _ in range(n_proj):
        th = rng.normal(size=(2,)).astype(np.float32)
        th /= (np.linalg.norm(th) + 1e-9)

        ox = np.argsort(X @ th, kind="mergesort")
        oy = np.argsort(Y @ th, kind="mergesort")

        rx = np.empty(N, dtype=np.float32)
        ry = np.empty(N, dtype=np.float32)
        rx[ox] = np.arange(N, dtype=np.float32)
        ry[oy] = np.arange(N, dtype=np.float32)

        scoreX += rx
        scoreY += ry

    orderX = np.argsort(scoreX, kind="mergesort")
    orderY = np.argsort(scoreY, kind="mergesort")

    T = np.empty_like(X, dtype=np.float32)
    T[orderX] = Y[orderY]
    return T


# ---------------------------
# Time helpers
# ---------------------------
def try_seoul_tz() -> Optional["ZoneInfo"]:
    if ZoneInfo is None:
        return None
    try:
        return ZoneInfo("Asia/Seoul")
    except Exception:
        return None


def next_new_year(now: datetime, tz: Optional["ZoneInfo"]) -> datetime:
    if tz is not None and now.tzinfo is None:
        now = now.replace(tzinfo=tz)
    if tz is not None:
        return datetime(now.year + 1, 1, 1, 0, 0, 0, tzinfo=tz)
    return datetime(now.year + 1, 1, 1, 0, 0, 0)


# ---------------------------
# Shaders
# ---------------------------
UPDATE_VS = r"""
#version 330

in vec2 in_pos;
in vec2 in_vel;
in vec2 in_home;
in vec2 in_target;
in float in_seed;

uniform float u_time;
uniform float u_dt;
uniform float u_phase;
uniform float u_boom;
uniform float u_aspect;
uniform float u_impulse;
uniform float u_brake;

out vec2 out_pos;
out vec2 out_vel;
out vec2 out_home;
out vec2 out_target;
out float out_seed;

float hash11(float p) {
    p = fract(p * 0.1031);
    p *= p + 33.33;
    p *= p + p;
    return fract(p);
}

vec2 curl(vec2 p) {
    float dfdx = 3.0 * cos(p.x * 3.0 + u_time * 0.7);
    float dfdy = -3.0 * sin(p.y * 3.0 - u_time * 0.9);
    return vec2(dfdy, -dfdx);
}

float ease_in_out(float x) {
    x = clamp(x, 0.0, 1.0);
    return x * x * (3.0 - 2.0 * x);
}

void main() {
    vec2 pos = in_pos;
    vec2 vel = in_vel;

    float e = ease_in_out(u_phase);
    vec2 morph_target = mix(in_home, in_target, e);

    float boom = clamp(u_boom, 0.0, 1.0);
    float hold = smoothstep(0.0, 0.15, boom);

    // spring (weak during explosion-hold)
    float k = mix(5.0, 18.0, e);
    k *= mix(1.0, 0.03, hold);
    vec2 f = (morph_target - pos) * k;

    // chaos
    float chaos = (1.0 - e);
    chaos = chaos * chaos;

    vec2 cn = curl(pos);
    vec2 tang = vec2(-pos.y, pos.x);

    // idle damping: reduce noise when already near the target (2025 stable)
    float d = length(morph_target - pos);
    float idle = smoothstep(0.01, 0.10, d);

    const float NOISE_IDLE = 0.12; // smaller => steadier "2025"
    float noise_scale = mix(NOISE_IDLE, 1.0, max(hold, idle));

    f += cn   * (0.35 * chaos * noise_scale);
    f += tang * (0.25 * chaos * noise_scale);

    // one-shot impulse
    if (u_impulse > 0.5) {
        float r = length(pos) + 1e-6;
        vec2 radial = pos / r;
        float jitter = 0.4 + 0.6 * hash11(in_seed * 91.17);

        float kick = 1.6 + 2.6 * jitter;
        vel += radial * kick;
        vel += tang   * (0.55 * (1.0 - jitter));
    }

    // damping (stable at phase=0, stronger stabilize as phase increases)
    const float BASE_IDLE  = 0.750; // larger => more floaty, smaller => more locked
    const float BASE_MORPH = 0.93;

    float base = mix(BASE_IDLE, BASE_MORPH, e);
    base = mix(base, 0.9985, hold);             // explosion hold: long tail motion
    float damp = pow(base, u_dt * 60.0);

    float damp_brake = pow(0.55, u_dt * 60.0);
    damp = mix(damp, damp_brake, clamp(u_brake, 0.0, 1.0));

    vel = vel * damp + f * u_dt;
    pos = pos + vel * u_dt;

    float bound = mix(1.25, 2.2, hold);
    if (pos.x > bound) { pos.x = bound; vel.x *= -0.6; }
    if (pos.x < -bound){ pos.x = -bound; vel.x *= -0.6; }
    if (pos.y > bound) { pos.y = bound; vel.y *= -0.6; }
    if (pos.y < -bound){ pos.y = -bound; vel.y *= -0.6; }

    out_pos = pos;
    out_vel = vel;
    out_home = in_home;
    out_target = in_target;
    out_seed = in_seed;
}
"""

RENDER_VS = r"""
#version 330
in vec2 in_pos;
in vec2 in_vel;
in float in_seed;

uniform float u_aspect;
uniform float u_phase;
uniform float u_boom;

out float v_speed;
out float v_seed;

void main() {
    vec2 p = in_pos;
    p.y /= u_aspect;

    gl_Position = vec4(p, 0.0, 1.0);

    float sp = length(in_vel);
    v_speed = sp;
    v_seed = in_seed;

    gl_PointSize = 2.0 + 5.0 * clamp(sp, 0.0, 1.0) + 10.0 * u_boom + 1.5 * u_phase;
}
"""

RENDER_FS = r"""
#version 330
in float v_speed;
in float v_seed;

uniform float u_exposure;

out vec4 f_color;

float hash11(float p) {
    p = fract(p * 0.1031);
    p *= p + 33.33;
    p *= p + p;
    return fract(p);
}

void main() {
    vec2 uv = gl_PointCoord * 2.0 - 1.0;
    float d = dot(uv, uv);
    float a = smoothstep(1.0, 0.0, d);

    float h = hash11(v_seed * 17.3);
    float s = clamp(v_speed * 0.9 + 0.15, 0.0, 1.0);

    vec3 col = 0.6 + 0.4 * cos(6.28318 * (vec3(0.0, 0.33, 0.67) + h));
    col *= (0.8 + 2.2 * s);
    col *= u_exposure;

    f_color = vec4(col, a * 0.75);
}
"""

QUAD_VS = r"""
#version 330
in vec2 in_pos;
out vec2 v_uv;
void main() {
    v_uv = in_pos * 0.5 + 0.5;
    gl_Position = vec4(in_pos, 0.0, 1.0);
}
"""

FADE_FS = r"""
#version 330
uniform sampler2D u_tex;
uniform float u_fade;
in vec2 v_uv;
out vec4 f_color;
void main() {
    vec4 c = texture(u_tex, v_uv);
    f_color = c * u_fade;
}
"""


# ---------------------------
# Control model
# ---------------------------
@dataclass
class Controls:
    sim_time: float
    phase: float
    boom: float
    impulse: float
    brake: float


# ---------------------------
# App
# ---------------------------
class NewYearOT(mglw.WindowConfig):
    gl_version = (3, 3)
    title = "New Year 2026 - OT Fireworks (moderngl)"
    window_size = (1280, 720)
    aspect_ratio = None
    resizable = True
    samples = 0

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.ctx: moderngl.Context = self.wnd.ctx
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE

        # visual
        self.exposure = 2.8
        self.trail_keep = 0.985

        # simulation config
        self.N = 100_000
        self.n_proj = 72
        self.seed = 2026

        self.explode_dur = 10.0
        self.morph_dur = 5.0

        # state
        self.manual_boom = 0.0
        self.manual_phase: Optional[float] = None

        self.preview = False
        self.seq_t = 0.0
        self.seq_speed = 6.0
        self.did_impulse = False
        self.auto_impulsed = False

        self.tz = try_seoul_tz()
        now = datetime.now(self.tz) if self.tz else datetime.now()
        self.new_year = next_new_year(now, self.tz)

        # generate particle data
        self._regen_clouds()

        # buffers: pos2 vel2 home2 target2 seed1 => 9 floats
        self.buf_a = self.ctx.buffer(self.particles.tobytes())
        self.buf_b = self.ctx.buffer(reserve=self.buf_a.size)

        self.update_prog = self.ctx.program(
            vertex_shader=UPDATE_VS,
            varyings=["out_pos", "out_vel", "out_home", "out_target", "out_seed"],
        )
        self.render_prog = self.ctx.program(vertex_shader=RENDER_VS, fragment_shader=RENDER_FS)

        self.vao_update_a = self.ctx.vertex_array(
            self.update_prog,
            [(self.buf_a, "2f 2f 2f 2f 1f", "in_pos", "in_vel", "in_home", "in_target", "in_seed")],
        )
        self.vao_update_b = self.ctx.vertex_array(
            self.update_prog,
            [(self.buf_b, "2f 2f 2f 2f 1f", "in_pos", "in_vel", "in_home", "in_target", "in_seed")],
        )

        self.vao_render_a = self.ctx.vertex_array(
            self.render_prog,
            [(self.buf_a, "2f 2f 2x4 2x4 1f", "in_pos", "in_vel", "in_seed")],
        )
        self.vao_render_b = self.ctx.vertex_array(
            self.render_prog,
            [(self.buf_b, "2f 2f 2x4 2x4 1f", "in_pos", "in_vel", "in_seed")],
        )

        quad = np.array([[-1, -1], [1, -1], [-1, 1], [-1, 1], [1, -1], [1, 1]], dtype=np.float32)
        self.quad_vbo = self.ctx.buffer(quad.tobytes())
        self.fade_prog = self.ctx.program(vertex_shader=QUAD_VS, fragment_shader=FADE_FS)
        self.quad_vao = self.ctx.vertex_array(self.fade_prog, [(self.quad_vbo, "2f", "in_pos")])

        self._create_fbos()

        # ping-pong indices
        self.ping = 0        # particle buffers
        self.trail_ping = 0  # trail textures

    def _create_fbos(self):
        w, h = self.wnd.size

        def make_tex():
            return self.ctx.texture((w, h), 4, dtype="f1")

        self.tex0 = make_tex()
        self.tex1 = make_tex()
        self.fbo0 = self.ctx.framebuffer(color_attachments=[self.tex0])
        self.fbo1 = self.ctx.framebuffer(color_attachments=[self.tex1])

        self.fbo0.clear(0, 0, 0, 1)
        self.fbo1.clear(0, 0, 0, 1)
        self.trail_ping = 0

    def on_resize(self, width: int, height: int):
        self._create_fbos()

    def _regen_clouds(self):
        X = text_points("2025", self.N, seed=self.seed + 1)
        Y = text_points("2026", self.N, seed=self.seed + 2)
        T = sliced_ot_perm_map(X, Y, n_proj=self.n_proj, seed=self.seed + 3)

        rng = np.random.default_rng(self.seed + 4)
        vel = rng.normal(scale=0.01, size=(self.N, 2)).astype(np.float32)
        seed = rng.random(self.N).astype(np.float32)

        self.particles = np.zeros((self.N, 9), dtype=np.float32)
        self.particles[:, 0:2] = X
        self.particles[:, 2:4] = vel
        self.particles[:, 4:6] = X
        self.particles[:, 6:8] = T
        self.particles[:, 8] = seed

    def _reset_sim(self, clear_trails: bool = True):
        self.buf_a.write(self.particles.tobytes())
        self.buf_b.write(self.particles.tobytes())
        self.ping = 0

        if clear_trails:
            self.fbo0.clear(0, 0, 0, 1)
            self.fbo1.clear(0, 0, 0, 1)
            self.trail_ping = 0

        self.manual_boom = 0.0
        self.manual_phase = None

    def on_key_event(self, key, action, modifiers):
        keys = self.wnd.keys
        if action != keys.ACTION_PRESS:
            return

        if key == keys.ESCAPE:
            self.wnd.close()
            return

        if key == keys.R:
            self._regen_clouds()
            self._reset_sim(clear_trails=True)
            return

        if key == keys.SPACE:
            self.manual_boom = 1.0
            return

        if key == keys.M:
            self.manual_phase = 0.0 if self.manual_phase is None else None
            return

        if key == keys.UP and self.manual_phase is not None:
            self.manual_phase = float(min(1.0, self.manual_phase + 0.05))
            return

        if key == keys.DOWN and self.manual_phase is not None:
            self.manual_phase = float(max(0.0, self.manual_phase - 0.05))
            return

        # preview toggle/restart
        if key == keys.F:
            self.preview = not self.preview
            if self.preview:
                self.seq_t = 0.0
                self.did_impulse = False
            return

        if key == keys.G:
            self._reset_sim(clear_trails=True)
            self.preview = True
            self.seq_t = 0.0
            self.did_impulse = False
            return

        # exposure
        if key == keys.EQUAL:
            self.exposure = min(8.0, self.exposure + 0.3)
            return
        if key == keys.MINUS:
            self.exposure = max(0.3, self.exposure - 0.3)
            return

        # trail keep
        if key == keys.COMMA:
            self.trail_keep = min(0.997, self.trail_keep + 0.002)
            return
        if key == keys.PERIOD:
            self.trail_keep = max(0.94, self.trail_keep - 0.002)
            return

    def _auto_phase_and_boom(self, now_dt: datetime) -> Tuple[float, float]:
        dt = (now_dt - self.new_year).total_seconds()
        if dt < 0.0:
            return 0.0, 0.0

        if dt < self.explode_dur:
            x = dt / self.explode_dur
            boom = 1.0 - abs(2.0 * x - 1.0)
            boom = max(0.0, min(1.0, boom))
            boom = boom * boom * (3.0 - 2.0 * boom)
            return 0.0, boom

        t2 = dt - self.explode_dur
        phase = max(0.0, min(1.0, t2 / self.morph_dur))
        return phase, 0.0

    def _compute_controls(self, t: float, dt: float) -> Controls:
        now_dt = datetime.now(self.tz) if self.tz else datetime.now()

        auto_phase, auto_boom = self._auto_phase_and_boom(now_dt)
        phase = auto_phase if self.manual_phase is None else float(self.manual_phase)

        self.manual_boom = max(0.0, self.manual_boom - dt * 1.2)
        boom = max(auto_boom, self.manual_boom)

        impulse = 0.0
        brake = 0.0
        sim_time = float(t)

        if self.preview:
            self.seq_t += dt * self.seq_speed
            tt = self.seq_t
            sim_time = float(self.seq_t)

            ex = self.explode_dur
            mo = self.morph_dur

            if tt < ex:
                x = tt / max(1e-6, ex)
                boom = 1.0 - abs(2.0 * x - 1.0)
                boom = max(0.0, min(1.0, boom))
                boom = boom * boom * (3.0 - 2.0 * boom)
                phase = 0.0

                if not self.did_impulse:
                    impulse = 1.0
                    self.did_impulse = True

            elif tt < ex + mo:
                x = (tt - ex) / max(1e-6, mo)
                x = max(0.0, min(1.0, x))
                phase = x * x * (3.0 - 2.0 * x)
                boom = 0.0
                brake = max(0.0, min(1.0, x / 0.25))
            else:
                phase = 1.0
                boom = 0.0
                brake = 1.0

        else:
            # one-shot impulse on real new year
            auto_dt = (now_dt - self.new_year).total_seconds()
            if auto_dt < 0.0:
                self.auto_impulsed = False
            else:
                if auto_dt < self.explode_dur and (not self.auto_impulsed):
                    impulse = 1.0
                    self.auto_impulsed = True
                if auto_dt >= self.explode_dur:
                    x = (auto_dt - self.explode_dur) / max(1e-6, self.morph_dur)
                    brake = max(0.0, min(1.0, x / 0.25))

        return Controls(sim_time=sim_time, phase=phase, boom=boom, impulse=impulse, brake=brake)

    def on_render(self, time: float, frame_time: float):
        self.render(time, frame_time)

    def render(self, t: float, frame_time: float):
        dt = float(min(frame_time, 1.0 / 30.0))
        ctl = self._compute_controls(t, dt)

        w, h = self.wnd.size
        aspect = h / max(1.0, w)

        # uniforms
        self.update_prog["u_time"].value = float(ctl.sim_time)
        self.update_prog["u_dt"].value = dt
        self.update_prog["u_phase"].value = float(ctl.phase)
        if "u_boom" in self.update_prog:
            self.update_prog["u_boom"].value = float(ctl.boom)
        if "u_aspect" in self.update_prog:
            self.update_prog["u_aspect"].value = float(aspect)
        self.update_prog["u_impulse"].value = float(ctl.impulse)
        self.update_prog["u_brake"].value = float(ctl.brake)

        # particle transform ping-pong
        if self.ping == 0:
            self.vao_update_a.transform(self.buf_b, vertices=self.N)
            in_render = self.vao_render_b
        else:
            self.vao_update_b.transform(self.buf_a, vertices=self.N)
            in_render = self.vao_render_a
        self.ping ^= 1

        # trail ping-pong
        if self.trail_ping == 0:
            src_tex, dst_fbo, out_tex = self.tex0, self.fbo1, self.tex1
        else:
            src_tex, dst_fbo, out_tex = self.tex1, self.fbo0, self.tex0

        dst_fbo.use()
        dst_fbo.clear(0, 0, 0, 1)

        if ctl.boom > 0.001:
            fade = self.trail_keep
        elif ctl.phase < 1.0:
            fade = min(self.trail_keep, 0.93)
        else:
            fade = 0.975

        # fade previous
        self.ctx.blend_func = moderngl.ONE, moderngl.ZERO
        src_tex.use(location=0)
        self.fade_prog["u_tex"].value = 0
        self.fade_prog["u_fade"].value = float(fade)
        self.quad_vao.render(moderngl.TRIANGLES)

        # draw particles
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE
        self.render_prog["u_aspect"].value = float(aspect)
        self.render_prog["u_phase"].value = float(ctl.phase)
        self.render_prog["u_boom"].value = float(ctl.boom)
        self.render_prog["u_exposure"].value = float(self.exposure)
        in_render.render(mode=moderngl.POINTS)

        # present
        self.ctx.screen.use()
        self.ctx.blend_func = moderngl.ONE, moderngl.ZERO
        out_tex.use(location=0)
        self.fade_prog["u_tex"].value = 0
        self.fade_prog["u_fade"].value = 1.0
        self.quad_vao.render(moderngl.TRIANGLES)

        self.trail_ping ^= 1


if __name__ == "__main__":
    mglw.run_window_config(NewYearOT)
