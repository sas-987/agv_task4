#!/usr/bin/env python3
"""
ViZDoom Maze Navigator — Level 1
=================================
Autonomously navigates MAP01.wad from spawn to the Blue Skull key using:
  • Global planning  : RRT* on an occupancy grid built from sector geometry
  • Path smoothing   : greedy string-pulling (visibility-graph shortcutting)
  • Local control    : closed-loop proportional heading controller

Usage
-----
    python vizdoom_navigator.py              # expects MAP01.wad in cwd
    python vizdoom_navigator.py MAP01.wad    # explicit WAD path

Dependencies
------------
    pip install vizdoom numpy
    pip install matplotlib          # optional — saves rrt_path.png
    pip install opencv-python       # optional — faster dilation
    pip install scipy               # optional — fallback dilation

Coordinate conventions (Doom engine)
-------------------------------------
  • X+ = East,  Y+ = North  (standard maths axes)
  • ANGLE 0° = East,  90° = North,  180° = West  (CCW-positive)
  • math.atan2(dy, dx) matches ANGLE exactly → no sign gymnastics needed
"""

import os
import sys
import math
import time
import random
import numpy as np
import vizdoom as vzd

# ════════════════════════════════════════════════════════════════════
#  TUNABLE PARAMETERS
# ════════════════════════════════════════════════════════════════════

# ── Occupancy grid ──────────────────────────────────────────────────
GRID_RES        = 8      # world units per grid cell  (smaller → finer but slower RRT)
WALL_HALF_THICK = 1       # Bresenham half-thickness when drawing walls (grid cells)
INFLATION_R     = 1       # obstacle inflation radius  (grid cells ≥ player_radius/GRID_RES)

# ── RRT* ─────────────────────────────────────────────────────────────
RRT_MAX_ITER    = 50000   # maximum tree-growth iterations
RRT_STEP        = 6      # max step length (grid cells)
RRT_RADIUS      = 15      # rewire neighbourhood radius (grid cells)
GOAL_BIAS       = 0.15    # fraction of samples placed directly at the goal

# ── Trajectory following ─────────────────────────────────────────────
WP_REACH_WU     = 52.0    # waypoint acceptance radius (world units)
ALIGN_THRESH    = 22.0    # °: move forward only when heading error < this
STUCK_LIMIT     = 300     # tics without meaningful movement → trigger escape

EPISODE_TIMEOUT = 35000   # safety timeout (tics; 35 Hz → ~1 000 s)


# ════════════════════════════════════════════════════════════════════
#  SMALL UTILITIES
# ════════════════════════════════════════════════════════════════════

def angle_diff(target: float, current: float) -> float:
    """
    Signed difference  (target − current)  wrapped to (−180, +180].
    Positive ↔ need to turn CCW (TURN_LEFT).
    """
    d = (target - current) % 360.0
    return d - 360.0 if d > 180.0 else d


def _d2(ax, ay, bx, by) -> float:
    return (ax - bx) ** 2 + (ay - by) ** 2


# ════════════════════════════════════════════════════════════════════
#  OCCUPANCY GRID
# ════════════════════════════════════════════════════════════════════

def build_occupancy_grid(sectors):
    """
    Convert ViZDoom sector/line geometry into a 2-D binary grid.

    Returns
    -------
    grid  : np.ndarray shape (H, W),  uint8  0=free 1=obstacle
    w2g   : (wx, wy) → (gx, gy)  world → grid
    g2w   : (gx, gy) → (wx, wy)  grid  → world  (cell centre)
    """
    # ── collect unique blocking lines ───────────────────────────────
    seen  = set()
    lines = []
    for sec in sectors:
        for ln in sec.lines:
            if not ln.is_blocking:
                continue
            key = (round(ln.x1, 1), round(ln.y1, 1),
                   round(ln.x2, 1), round(ln.y2, 1))
            if key not in seen:
                seen.add(key)
                lines.append((ln.x1, ln.y1, ln.x2, ln.y2))

    if not lines:
        raise RuntimeError(
            "No blocking lines found — did you call set_sectors_info_enabled(True)?")

    all_x = [c for l in lines for c in (l[0], l[2])]
    all_y = [c for l in lines for c in (l[1], l[3])]
    pad   = (WALL_HALF_THICK + INFLATION_R + 4) * GRID_RES
    wx0   = min(all_x) - pad;  wy0 = min(all_y) - pad
    wx1   = max(all_x) + pad;  wy1 = max(all_y) + pad

    W = int(math.ceil((wx1 - wx0) / GRID_RES)) + 2
    H = int(math.ceil((wy1 - wy0) / GRID_RES)) + 2

    def w2g(wx, wy):
        return (int((wx - wx0) / GRID_RES),
                int((wy - wy0) / GRID_RES))

    def g2w(gx, gy):
        return (wx0 + (gx + 0.5) * GRID_RES,
                wy0 + (gy + 0.5) * GRID_RES)

    grid = np.zeros((H, W), dtype=np.uint8)

    # ── rasterise walls ─────────────────────────────────────────────
    for (x1, y1, x2, y2) in lines:
        _bresenham_thick(grid, w2g(x1, y1), w2g(x2, y2), WALL_HALF_THICK)

    # ── inflate obstacles (player body clearance) ────────────────────
    grid = _inflate(grid, INFLATION_R)

    total_cells = W * H
    wall_cells  = int(grid.sum())
    free_cells  = total_cells - wall_cells
    print(f"[Grid] {W}×{H} = {total_cells} cells | "
          f"obstacle={wall_cells} | free={free_cells}")
    return grid, w2g, g2w


def _bresenham_thick(grid, p1, p2, half_t):
    """Draw a thick line into *grid* in-place using Bresenham + square brush."""
    H, W = grid.shape
    x0, y0 = int(round(p1[0])), int(round(p1[1]))
    x1, y1 = int(round(p2[0])), int(round(p2[1]))
    dx = abs(x1 - x0);  sx = 1 if x0 < x1 else -1
    dy = abs(y1 - y0);  sy = 1 if y0 < y1 else -1
    err = dx - dy
    while True:
        for ddx in range(-half_t, half_t + 1):
            for ddy in range(-half_t, half_t + 1):
                nx, ny = x0 + ddx, y0 + ddy
                if 0 <= nx < W and 0 <= ny < H:
                    grid[ny, nx] = 1
        if x0 == x1 and y0 == y1:
            break
        e2 = err << 1
        if e2 > -dy: err -= dy; x0 += sx
        if e2 <  dx: err += dx; y0 += sy


def _inflate(grid, radius):
    """Binary dilation using the best available library."""
    try:
        import cv2
        k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
        return cv2.dilate(grid, k).astype(np.uint8)
    except ImportError:
        pass
    try:
        from scipy.ndimage import binary_dilation
        Y, X = np.ogrid[-radius: radius + 1, -radius: radius + 1]
        k    = (X ** 2 + Y ** 2 <= radius ** 2)
        return binary_dilation(grid, k).astype(np.uint8)
    except ImportError:
        pass
    # pure-NumPy fallback (slow but dependency-free)
    out = grid.copy()
    for r in range(1, radius + 1):
        for axis in (0, 1):
            out = np.maximum(out, np.roll(grid,  r, axis=axis))
            out = np.maximum(out, np.roll(grid, -r, axis=axis))
    return out.astype(np.uint8)


def nearest_free(grid, gx, gy):
    """Return the nearest obstacle-free cell to (gx, gy)."""
    H, W = grid.shape
    gx, gy = int(round(gx)), int(round(gy))
    gx = max(0, min(W - 1, gx))
    gy = max(0, min(H - 1, gy))
    if grid[gy, gx] == 0:
        return gx, gy
    
    for r in range(1, 60):
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                nx, ny = gx + dx, gy + dy
                if 0 <= nx < W and 0 <= ny < H and grid[ny, nx] == 0:
                    return nx, ny
    return gx, gy   # fallback (should not happen)



# ════════════════════════════════════════════════════════════════════
#  RRT*
# ════════════════════════════════════════════════════════════════════

class _Node:
    """Compact RRT* node."""
    __slots__ = ('x', 'y', 'parent', 'cost')

    def __init__(self, x, y):
        self.x = float(x)
        self.y = float(y)
        self.parent = None
        self.cost   = 0.0


def rrt_star(grid, start, goal):
    """
    RRT* path planner on a 2-D binary grid.

    Parameters
    ----------
    grid         : np.ndarray  0=free 1=obstacle
    start, goal  : (gx, gy) int or float tuples

    Returns
    -------
    List of (gx, gy) float tuples from start → goal,  or None if failed.
    """
    H, W = grid.shape

    def is_free(x, y):
        xi, yi = int(round(x)), int(round(y))
        return 0 <= xi < W and 0 <= yi < H and grid[yi, xi] == 0

    def line_clear(ax, ay, bx, by):
        steps = max(2, int(math.hypot(bx - ax, by - ay) * 2))
        for i in range(steps + 1):
            t = i / steps
            if not is_free(ax + t * (bx - ax), ay + t * (by - ay)):
                return False
        return True

    root      = _Node(*start)
    nodes     = [root]
    goal_node = _Node(*goal)
    best      = None          # best goal node found so far

    print(f"[RRT*] start={start}  goal={goal}  "
          f"grid={W}×{H}  iter={RRT_MAX_ITER}")

    for it in range(RRT_MAX_ITER):
        # ── sample ─────────────────────────────────────────────────
        if random.random() < GOAL_BIAS:
            rx, ry = goal
        else:
            rx = random.uniform(0, W - 1)
            ry = random.uniform(0, H - 1)

        # ── nearest node ───────────────────────────────────────────
        nn   = min(nodes, key=lambda n: _d2(n.x, n.y, rx, ry))
        dist = math.sqrt(_d2(nn.x, nn.y, rx, ry))
        if dist < 1e-6:
            continue
        ratio = min(RRT_STEP, dist) / dist
        nx    = nn.x + ratio * (rx - nn.x)
        ny    = nn.y + ratio * (ry - nn.y)

        if not is_free(nx, ny):
            continue
        if not line_clear(nn.x, nn.y, nx, ny):
            continue

        new = _Node(nx, ny)

        # ── choose best parent (RRT* cost optimisation) ────────────
        near       = [n for n in nodes
                      if _d2(n.x, n.y, nx, ny) <= RRT_RADIUS ** 2]
        best_parent = nn
        best_cost   = nn.cost + math.sqrt(_d2(nn.x, nn.y, nx, ny))
        for n in near:
            c = n.cost + math.sqrt(_d2(n.x, n.y, nx, ny))
            if c < best_cost and line_clear(n.x, n.y, nx, ny):
                best_cost   = c
                best_parent = n
        new.parent = best_parent
        new.cost   = best_cost
        nodes.append(new)

        # ── rewire nearby nodes ────────────────────────────────────
        for n in near:
            if n is best_parent:
                continue
            nc = new.cost + math.sqrt(_d2(new.x, new.y, n.x, n.y))
            if nc < n.cost and line_clear(new.x, new.y, n.x, n.y):
                n.parent = new
                n.cost   = nc

        # ── try connecting to goal ─────────────────────────────────
        dg = math.sqrt(_d2(new.x, new.y, *goal))
        if dg <= RRT_STEP and line_clear(new.x, new.y, *goal):
            gn         = _Node(*goal)
            gn.parent  = new
            gn.cost    = new.cost + dg
            if best is None or gn.cost < best.cost:
                best = gn
                nodes.append(gn)
                print(f"    ✓ iter={it:5d}  path_cost={gn.cost:.1f}  "
                      f"nodes={len(nodes)}")

    if best is None:
        print("[RRT*] No path found — try increasing RRT_MAX_ITER or GRID_RES.")
        return None

    # ── extract path ───────────────────────────────────────────────
    path = []
    n = best
    while n is not None:
        path.append((n.x, n.y))
        n = n.parent
    return list(reversed(path))


# ════════════════════════════════════════════════════════════════════
#  PATH SMOOTHING — string-pulling (greedy LOS shortcutting)
# ════════════════════════════════════════════════════════════════════

def smooth_path(path, grid, passes=3):
    """
    Remove unnecessary intermediate waypoints.
    Each pass tries to connect node i directly to the furthest j > i
    that has a clear line-of-sight, skipping all nodes in between.
    """
    H, W = grid.shape

    def clear(ax, ay, bx, by):
        steps = max(2, int(math.hypot(bx - ax, by - ay) * 2))
        for i in range(steps + 1):
            t  = i / steps
            xi = int(round(ax + t * (bx - ax)))
            yi = int(round(ay + t * (by - ay)))
            if not (0 <= xi < W and 0 <= yi < H) or grid[yi, xi]:
                return False
        return True

    for _ in range(passes):
        pruned = [path[0]]
        i = 0
        while i < len(path) - 1:
            # find furthest visible waypoint from i
            for j in range(len(path) - 1, i, -1):
                gxi, gyi = path[i]
                gxj, gyj = path[j]
                if clear(gxi, gyi, gxj, gyj):
                    pruned.append(path[j])
                    i = j
                    break
            else:
                i += 1
                if i < len(path):
                    pruned.append(path[i])
        path = pruned

    return path


# ════════════════════════════════════════════════════════════════════
#  VISUALISATION  (requires matplotlib — gracefully skipped if absent)
# ════════════════════════════════════════════════════════════════════

def visualise(grid, path_g, start_g, goal_g, out="rrt_path.png"):
    try:
        import matplotlib
        matplotlib.use("Agg")          # headless — saves to file
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(12, 9))
        ax.imshow(grid, cmap="gray_r", origin="lower", interpolation="none")

        if path_g and len(path_g) > 1:
            xs, ys = zip(*path_g)
            ax.plot(xs, ys,   "b-",  lw=1.5, alpha=0.8, label="RRT* path")
            ax.plot(xs[1:-1], ys[1:-1], "b.", ms=4)

        ax.plot(*start_g, "go",  ms=12, zorder=5, label="Start")
        ax.plot(*goal_g,  "r*",  ms=16, zorder=5, label="Goal (Blue Skull)")
        ax.set_title("Occupancy Grid + RRT* Planned Path", fontsize=13)
        ax.legend()
        plt.tight_layout()
        plt.savefig(out, dpi=130)
        plt.close(fig)
        print(f"[Vis] Path map saved → {out}")
    except Exception as e:
        print(f"[Vis] Skipped ({e})")


# ════════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════════

def main():
    wad_path = sys.argv[1] if len(sys.argv) > 1 else "MAP01.wad"
    wad_path = os.path.abspath(wad_path)

    if not os.path.isfile(wad_path):
        print(f"[Error] WAD file not found: {wad_path}")
        print("        Usage: python vizdoom_navigator.py [MAP01.wad]")
        sys.exit(1)

    print("═" * 58)
    print("   ViZDoom RRT* Maze Navigator  —  Level 1")
    print("═" * 58)
    print(f"[WAD] {wad_path}")

    # ── 1. Initialise ViZDoom ──────────────────────────────────────
    game = vzd.DoomGame()
    game.set_doom_scenario_path(wad_path)
    game.set_doom_map("MAP01")
    game.set_mode(vzd.Mode.PLAYER)

    # ─ display ─
    game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)
    game.set_screen_format(vzd.ScreenFormat.RGB24)
    game.set_render_hud(True)
    game.set_render_weapon(True)
    game.set_render_crosshair(False)
    game.set_render_decals(False)
    game.set_render_particles(False)
    game.set_window_visible(True)
    game.set_sound_enabled(False)

    # ─ automap (whole map, no textures = cleaner image) ─
    game.set_automap_buffer_enabled(True)
    game.set_automap_mode(vzd.AutomapMode.WHOLE)
    game.set_automap_rotate(False)
    game.set_automap_render_textures(False)

    # ─ world-geometry info ─
    game.set_objects_info_enabled(True)
    game.set_sectors_info_enabled(True)

    # ─ game variables we need ─
    game.add_available_game_variable(vzd.GameVariable.POSITION_X)
    game.add_available_game_variable(vzd.GameVariable.POSITION_Y)
    game.add_available_game_variable(vzd.GameVariable.ANGLE)

    # ─ action space: forward + binary turn ─
    #   index 0 → MOVE_FORWARD   (0=stop,  1=move)
    #   index 1 → TURN_LEFT      (0/1)
    #   index 2 → TURN_RIGHT     (0/1)
    game.set_available_buttons([
        vzd.Button.MOVE_FORWARD,
        vzd.Button.TURN_LEFT,
        vzd.Button.TURN_RIGHT,
    ])

    game.set_doom_skill(1)                  # easiest — no enemy interference
    game.set_episode_timeout(EPISODE_TIMEOUT)
    game.init()
    print("[Init] ViZDoom initialised.")

    # ── 2. Read initial game state ─────────────────────────────────
    state = game.get_state()
    gv    = state.game_variables            # ordered as added above
    px    = float(gv[0])
    py    = float(gv[1])
    pangle= float(gv[2])
    print(f"[Init] Player spawn: ({px:.0f}, {py:.0f})  "
          f"facing={pangle:.1f}°")

    # ── 3. Locate the blue-skull goal ─────────────────────────────
    # ViZDoom object names follow ZDoom conventions:
    #   Blue Skull Key → "BlueSkull"
    #   Blue Card Key  → "BlueCard"
    BLUE_NAMES = {"BlueSkull", "BlueCard", "BlueSkulltag"}

    print("[Objects] All objects visible in map:")
    goal_pos = None
    for obj in state.objects:
        is_goal = (obj.name in BLUE_NAMES or "blue" in obj.name.lower())
        tag = "  ← GOAL" if is_goal else ""
        print(f"  {obj.name:30s}  "
              f"({obj.position_x:8.0f}, {obj.position_y:8.0f}){tag}")
        if is_goal and goal_pos is None:
            goal_pos = (float(obj.position_x), float(obj.position_y))

    if goal_pos is None:
        print("\n[Error] No blue-skull/card object found.")
        print("        Check the object list above and update BLUE_NAMES if needed.")
        game.close()
        sys.exit(1)

    print(f"\n[Goal] Blue Skull at ({goal_pos[0]:.0f}, {goal_pos[1]:.0f})")

    # ── 4. Build occupancy grid from sector geometry ───────────────
    print("\n[Grid] Building occupancy grid …")
    grid, w2g, g2w = build_occupancy_grid(state.sectors)

    start_g = nearest_free(grid, *w2g(px, py))
    goal_g  = nearest_free(grid, *w2g(*goal_pos))
    print(f"[Grid] start_grid={start_g}  goal_grid={goal_g}")

    # Sanity: start and goal must differ
    if start_g == goal_g:
        print("[Warn] Start and goal map to the same grid cell — "
              "increase GRID_RES or adjust waypoints.")

    # ── 5. RRT* global path planning ──────────────────────────────
    print(f"\n[RRT*] Planning (max_iter={RRT_MAX_ITER}) …")
    t0     = time.time()
    path_g = rrt_star(grid, start_g, goal_g)
    dt     = time.time() - t0

    if path_g is None:
        print(f"[RRT*] FAILED after {dt:.1f} s.")
        print("       Try: larger RRT_MAX_ITER, smaller GRID_RES, "
              "smaller INFLATION_R")
        game.close()
        sys.exit(1)

    print(f"[RRT*] Solved in {dt:.1f} s  —  {len(path_g)} raw waypoints")

    # ── 6. Path smoothing ─────────────────────────────────────────
    path_g = smooth_path(path_g, grid)
    print(f"[Path] After string-pulling: {len(path_g)} waypoints")

    # Convert grid coords → world coords
    waypoints = [g2w(gx, gy) for gx, gy in path_g]
    print(f"[Path] World waypoints ({len(waypoints)}):")
    for i, (wx, wy) in enumerate(waypoints):
        print(f"  WP{i:3d}: ({wx:8.0f}, {wy:8.0f})")

    # ── Visualise (saves rrt_path.png) ────────────────────────────
    visualise(grid, path_g, start_g, goal_g)

    # ── 7. Trajectory following — closed-loop heading controller ──
    print("\n[Nav] Starting trajectory following …")
    print(f"      Alignment threshold = {ALIGN_THRESH}°  |  "
          f"Waypoint reach = {WP_REACH_WU:.0f} wu")

    wp_idx      = 0
    n_wps       = len(waypoints)
    last_pos    = (px, py)
    stuck_ctr   = 0
    t_start     = time.time()

    while not game.is_episode_finished():
        state = game.get_state()
        if state is None:
            break

        gv      = state.game_variables
        px      = float(gv[0])
        py      = float(gv[1])
        pangle  = float(gv[2])

        # ── stuck detection ────────────────────────────────────────
        moved = math.hypot(px - last_pos[0], py - last_pos[1])
        stuck_ctr = stuck_ctr + 1 if moved < 0.5 else 0
        last_pos  = (px, py)

        if stuck_ctr > STUCK_LIMIT:
            print(f"[Nav] Stuck detected at ({px:.0f},{py:.0f}) — "
                  f"attempting escape …")
            # back up a little then turn 90° to the nearest free direction
            game.make_action([0, 0, 0], 15)    # pause
            # try turning left
            game.make_action([0, 1, 0], 20)
            stuck_ctr = 0
            continue

        # ── check if all waypoints consumed ───────────────────────
        if wp_idx >= n_wps:
            print("[Nav] All waypoints consumed — episode complete!")
            break

        tx, ty   = waypoints[wp_idx]
        dist_wp  = math.hypot(tx - px, ty - py)

        # ── advance to next waypoint when close enough ─────────────
        if dist_wp < WP_REACH_WU:
            print(f"[Nav] ✓ WP{wp_idx:3d}/{n_wps}  reached  "
                  f"({tx:.0f},{ty:.0f})")
            wp_idx += 1
            if wp_idx >= n_wps:
                elapsed = time.time() - t_start
                print(f"[Nav] *** Goal reached in {elapsed:.1f} s "
                      f"({game.get_episode_time()} tics) ***")
                break
            tx, ty = waypoints[wp_idx]

        # ── proportional heading controller ───────────────────────
        #
        #  desired_deg = direction from player to current waypoint
        #              = atan2(dy, dx) in world frame
        #  err         = signed angular error (CCW-positive)
        #              > 0 → player must turn LEFT  (TURN_LEFT  button)
        #              < 0 → player must turn RIGHT (TURN_RIGHT button)
        #
        dx        = tx - px
        dy        = ty - py                     # Doom: Y+ = north
        desired   = math.degrees(math.atan2(dy, dx))
        err       = angle_diff(desired, pangle) # ∈ (−180, 180]

        fwd = 1 if abs(err) < ALIGN_THRESH else 0

        if err > 2.0:
            action = [fwd, 1, 0]   # [MOVE_FORWARD, TURN_LEFT, TURN_RIGHT]
        elif err < -2.0:
            action = [fwd, 0, 1]
        else:
            action = [1, 0, 0]     # perfectly aligned — just move forward

        game.make_action(action, 1)

    # ── Episode ended ─────────────────────────────────────────────
    print("\n[Nav] Episode finished.")
    time.sleep(3)
    game.close()
    print("[Done] ViZDoom closed cleanly.")


# ════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    main()