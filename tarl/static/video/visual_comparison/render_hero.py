import subprocess, sys, os
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = "/Users/uber/Desktop/TaRL"
S = "/tmp/"
OUT = f"{ROOT}/static/video/visual_comparison/hero.mp4"

# ---- axes geometry measured on the source frames (792x924) ----
X0, X1 = 133, 1057        # x spine of *_curve.mp4 (1080x540): frame 0 .. frame N-1
Y0, Y1 = 430, 34          # y spine: reward 0 .. 1
FPS = 30

TACTILE, VISUAL = "#0fa3a0", "#f39c12"
INK, INK2, MUTED, LINE = "#17191c", "#3c4045", "#6b7078", "#e6e8eb"

for cand in ["Inter", "Helvetica Neue", "Helvetica", "Arial"]:
    if any(f.name == cand for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = cand; break
plt.rcParams["font.size"] = 22

def n_frames(path):
    out = subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
                                   "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", path])
    return int(out.decode().strip())

def digitize(last_png, N):
    """Recover the two reward curves from the fully-drawn last frame."""
    im = np.asarray(Image.open(last_png).convert("RGB")).astype(int)
    R, G, B = im[..., 0], im[..., 1], im[..., 2]
    masks = {
        "visual": (R > 200) & (G > 110) & (G < 190) & (B < 80),
        "tactile": (R < 80) & (G > 150) & (B > 140),
    }
    curves = {}
    for k, m in masks.items():
        m = m.copy()
        m[:135, :420] = False                      # legend swatches
        m[:, :X0] = False; m[:, X1 + 1:] = False   # outside the axes
        m[:Y1 - 6, :] = False; m[Y0 + 6:, :] = False
        xs, ys = [], []
        for x in range(X0, X1 + 1):
            rows = np.where(m[:, x])[0]
            if len(rows):
                xs.append((x - X0) / (X1 - X0) * (N - 1)); ys.append((Y0 - np.median(rows)) / (Y0 - Y1))
        xs, ys = np.array(xs), np.array(ys)
        f = np.arange(N)
        curves[k] = np.clip(np.interp(f, xs, ys), 0, 1)
        curves[k][0] = 0.0
    return curves

def video_frames(path):
    w, h = [int(v) for v in subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height", "-of", "csv=p=0", path]).decode().strip().split(",")]
    raw = subprocess.check_output(["ffmpeg", "-v", "error", "-i", path, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"])
    return np.frombuffer(raw, np.uint8).reshape(-1, h, w, 3)

panels = [
    ("ID",  "Seen position"),
    ("OOD", "Unseen position"),
]
data = []
VC = f"{ROOT}/static/video/visual_comparison"
for tag, title in panels:
    src = f"{VC}/{tag}_curve.mp4"      # rendered curves -> digitized reward values
    N = n_frames(src)
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-sseof", "-0.05", "-i", src, "-vframes", "1", f"{S}last_{tag}.png"], check=True)
    curves = digitize(f"{S}last_{tag}.png", N)
    rgb = video_frames(f"{VC}/{tag}_rgb.mp4")
    print(tag, "N", N, "rgb", rgb.shape, "visual end %.2f tactile end %.2f" % (curves["visual"][-1], curves["tactile"][-1]))
    data.append((title, rgb, curves))
N = min(d[1].shape[0] for d in data)

# ---- figure: 1900x900 px ----
PW, PH = 1900, 1000
W, H = PW / 100, PH / 100
fig = plt.figure(figsize=(W, H), dpi=100, facecolor="white")
def box(x, y, w, h):  # pixel box (top-left origin) -> figure fraction
    return [x / PW, 1 - (y + h) / PH, w / PW, h / PH]
artists = []
for i, (title, rgb, curves) in enumerate(data):
    px = i * 950
    fig.text((px + 475) / PW, 1 - 52 / PH, title, ha="center", va="center", fontsize=30, fontweight="bold", color=INK)
    ax_im = fig.add_axes(box(px + 475 - 270, 100, 540, 540)); ax_im.axis("off")
    im_art = ax_im.imshow(rgb[0], interpolation="bilinear")
    ax = fig.add_axes(box(px + 120, 690, 760, 230))
    ax.set_facecolor("white")
    for sp in ["top", "right"]: ax.spines[sp].set_visible(False)
    for sp in ["left", "bottom"]: ax.spines[sp].set_color(LINE); ax.spines[sp].set_linewidth(1.5)
    ax.tick_params(colors=INK2, labelsize=17, length=0, pad=6)
    ax.grid(True, color=LINE, linewidth=1, alpha=0.9)
    ax.set_xlim(0, N - 1); ax.set_ylim(0, 1.0)
    ax.set_yticks([0, 0.5, 1.0]); ax.set_xticks(range(0, N, 50))
    ax.set_xlabel("Frame", fontsize=18, color=INK2, labelpad=4)
    ax.set_ylabel("Predicted reward", fontsize=18, color=INK2, labelpad=8)
    lv, = ax.plot([], [], color=VISUAL, lw=4, solid_capstyle="round", label="Visual reward")
    lt, = ax.plot([], [], color=TACTILE, lw=4, solid_capstyle="round", label="Tactile reward (TaRL)")
    mv, = ax.plot([], [], "o", color=VISUAL, ms=11, mec="white", mew=2)
    mt, = ax.plot([], [], "o", color=TACTILE, ms=11, mec="white", mew=2)
    ax.legend(loc="upper left", frameon=False, fontsize=17, handlelength=1.6, labelcolor=INK2, borderaxespad=0.2)
    artists.append((im_art, lv, lt, mv, mt, rgb, curves))

fig.add_artist(plt.Line2D([0.5, 0.5], [0.06, 0.94], color=LINE, lw=2))

enc = ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{PW}x{PH}", "-r", str(FPS), "-i", "-",
       "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", OUT]
proc = subprocess.Popen(enc, stdin=subprocess.PIPE)
fig.canvas.draw()
for f in range(N):
    for im_art, lv, lt, mv, mt, rgb, curves in artists:
        im_art.set_data(rgb[f])
        xs = np.arange(f + 1)
        lv.set_data(xs, curves["visual"][:f + 1]); lt.set_data(xs, curves["tactile"][:f + 1])
        mv.set_data([f], [curves["visual"][f]]); mt.set_data([f], [curves["tactile"][f]])
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[..., :3]
    proc.stdin.write(np.ascontiguousarray(buf).tobytes())
    if f % 50 == 0: print("frame", f, flush=True)
proc.stdin.close(); proc.wait()
print("done", OUT)
