#!/usr/bin/env python3

import os
import sys
import math
import time
import random
import numpy as np
import vizdoom as vzd

GRID_RES = 8
WALL_HALF_THICK = 1
INFLATION_R = 1

RRT_MAX_ITER = 50000
RRT_STEP = 6
RRT_RADIUS = 15
GOAL_BIAS = 0.15

WP_REACH_WU = 52.0
ALIGN_THRESH = 22.0
STUCK_LIMIT = 300

EPISODE_TIMEOUT = 35000


def angle_diff(target: float, current: float) -> float:
    d = (target - current) % 360.0
    return d - 360.0 if d > 180.0 else d


def _d2(ax, ay, bx, by) -> float:
    return (ax - bx) ** 2 + (ay - by) ** 2


def build_occupancy_grid(sectors):
    seen = set()
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
        raise RuntimeError

    all_x = [c for l in lines for c in (l[0], l[2])]
    all_y = [c for l in lines for c in (l[1], l[3])]
    pad = (WALL_HALF_THICK + INFLATION_R + 4) * GRID_RES
    wx0 = min(all_x) - pad
    wy0 = min(all_y) - pad
    wx1 = max(all_x) + pad
    wy1 = max(all_y) + pad

    W = int(math.ceil((wx1 - wx0) / GRID_RES)) + 2
    H = int(math.ceil((wy1 - wy0) / GRID_RES)) + 2

    def w2g(wx, wy):
        return (int((wx - wx0) / GRID_RES),
                int((wy - wy0) / GRID_RES))

    def g2w(gx, gy):
        return (wx0 + (gx + 0.5) * GRID_RES,
                wy0 + (gy + 0.5) * GRID_RES)

    grid = np.zeros((H, W), dtype=np.uint8)

    for (x1, y1, x2, y2) in lines:
        _bresenham_thick(grid, w2g(x1, y1), w2g(x2, y2), WALL_HALF_THICK)

    grid = _inflate(grid, INFLATION_R)

    total_cells = W * H
    wall_cells = int(grid.sum())
    free_cells = total_cells - wall_cells
    print(f"[Grid] {W}×{H} = {total_cells} cells | obstacle={wall_cells} | free={free_cells}")
    return grid, w2g, g2w


def _bresenham_thick(grid, p1, p2, half_t):
    H, W = grid.shape
    x0, y0 = int(round(p1[0])), int(round(p1[1]))
    x1, y1 = int(round(p2[0])), int(round(p2[1]))
    dx = abs(x1 - x0)
    sx = 1 if x0 < x1 else -1
    dy = abs(y1 - y0)
    sy = 1 if y0 < y1 else -1
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
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy


def _inflate(grid, radius):
    try:
        import cv2
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
        return cv2.dilate(grid, k).astype(np.uint8)
    except ImportError:
        pass
    try:
        from scipy.ndimage import binary_dilation
        Y, X = np.ogrid[-radius: radius + 1, -radius: radius + 1]
        k = (X ** 2 + Y ** 2 <= radius ** 2)
        return binary_dilation(grid, k).astype(np.uint8)
    except ImportError:
        pass
    out = grid.copy()
    for r in range(1, radius + 1):
        for axis in (0, 1):
            out = np.maximum(out, np.roll(grid, r, axis=axis))
            out = np.maximum(out, np.roll(grid, -r, axis=axis))
    return out.astype(np.uint8)


def nearest_free(grid, gx, gy):
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
    return gx, gy


class _Node:
    __slots__ = ('x', 'y', 'parent', 'cost')

    def __init__(self, x, y):
        self.x = float(x)
        self.y = float(y)
        self.parent = None
        self.cost = 0.0


def rrt_star(grid, start, goal):
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

    root = _Node(*start)
    nodes = [root]
    best = None

    print(f"[RRT*] start={start} goal={goal} grid={W}×{H} iter={RRT_MAX_ITER}")

    for it in range(RRT_MAX_ITER):
        if random.random() < GOAL_BIAS:
            rx, ry = goal
        else:
            rx = random.uniform(0, W - 1)
            ry = random.uniform(0, H - 1)

        nn = min(nodes, key=lambda n: _d2(n.x, n.y, rx, ry))
        dist = math.sqrt(_d2(nn.x, nn.y, rx, ry))
        if dist < 1e-6:
            continue

        ratio = min(RRT_STEP, dist) / dist
        nx = nn.x + ratio * (rx - nn.x)
        ny = nn.y + ratio * (ry - nn.y)

        if not is_free(nx, ny):
            continue
        if not line_clear(nn.x, nn.y, nx, ny):
            continue

        new = _Node(nx, ny)

        near = [n for n in nodes if _d2(n.x, n.y, nx, ny) <= RRT_RADIUS ** 2]
        best_parent = nn
        best_cost = nn.cost + math.sqrt(_d2(nn.x, nn.y, nx, ny))

        for n in near:
            c = n.cost + math.sqrt(_d2(n.x, n.y, nx, ny))
            if c < best_cost and line_clear(n.x, n.y, nx, ny):
                best_cost = c
                best_parent = n

        new.parent = best_parent
        new.cost = best_cost
        nodes.append(new)

        for n in near:
            if n is best_parent:
                continue
            nc = new.cost + math.sqrt(_d2(new.x, new.y, n.x, n.y))
            if nc < n.cost and line_clear(new.x, new.y, n.x, n.y):
                n.parent = new
                n.cost = nc

        dg = math.sqrt(_d2(new.x, new.y, *goal))
        if dg <= RRT_STEP and line_clear(new.x, new.y, *goal):
            gn = _Node(*goal)
            gn.parent = new
            gn.cost = new.cost + dg
            if best is None or gn.cost < best.cost:
                best = gn
                nodes.append(gn)
                print(f"iter={it} cost={gn.cost:.1f}")

    if best is None:
        return None

    path = []
    n = best
    while n is not None:
        path.append((n.x, n.y))
        n = n.parent
    return list(reversed(path))


def smooth_path(path, grid, passes=3):
    H, W = grid.shape

    def clear(ax, ay, bx, by):
        steps = max(2, int(math.hypot(bx - ax, by - ay) * 2))
        for i in range(steps + 1):
            t = i / steps
            xi = int(round(ax + t * (bx - ax)))
            yi = int(round(ay + t * (by - ay)))
            if not (0 <= xi < W and 0 <= yi < H) or grid[yi, xi]:
                return False
        return True

    for _ in range(passes):
        pruned = [path[0]]
        i = 0
        while i < len(path) - 1:
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


def main():
    wad_path = sys.argv[1] if len(sys.argv) > 1 else "MAP01.wad"
    wad_path = os.path.abspath(wad_path)

    if not os.path.isfile(wad_path):
        sys.exit(1)

    game = vzd.DoomGame()
    game.set_doom_scenario_path(wad_path)
    game.set_doom_map("MAP01")
    game.set_mode(vzd.Mode.PLAYER)

    game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)
    game.set_screen_format(vzd.ScreenFormat.RGB24)
    game.set_render_hud(True)
    game.set_render_weapon(True)
    game.set_render_crosshair(False)
    game.set_render_decals(False)
    game.set_render_particles(False)
    game.set_window_visible(True)
    game.set_sound_enabled(False)

    game.set_automap_buffer_enabled(True)
    game.set_automap_mode(vzd.AutomapMode.WHOLE)
    game.set_automap_rotate(False)
    game.set_automap_render_textures(False)

    game.set_objects_info_enabled(True)
    game.set_sectors_info_enabled(True)

    game.add_available_game_variable(vzd.GameVariable.POSITION_X)
    game.add_available_game_variable(vzd.GameVariable.POSITION_Y)
    game.add_available_game_variable(vzd.GameVariable.ANGLE)

    game.set_available_buttons([
        vzd.Button.MOVE_FORWARD,
        vzd.Button.TURN_LEFT,
        vzd.Button.TURN_RIGHT,
    ])

    game.set_doom_skill(1)
    game.set_episode_timeout(EPISODE_TIMEOUT)
    game.init()

    state = game.get_state()
    gv = state.game_variables
    px = float(gv[0])
    py = float(gv[1])
    pangle = float(gv[2])

    BLUE_NAMES = {"BlueSkull", "BlueCard", "BlueSkulltag"}

    goal_pos = None
    for obj in state.objects:
        if (obj.name in BLUE_NAMES or "blue" in obj.name.lower()) and goal_pos is None:
            goal_pos = (float(obj.position_x), float(obj.position_y))

    if goal_pos is None:
        game.close()
        sys.exit(1)

    grid, w2g, g2w = build_occupancy_grid(state.sectors)

    start_g = nearest_free(grid, *w2g(px, py))
    goal_g = nearest_free(grid, *w2g(*goal_pos))

    path_g = rrt_star(grid, start_g, goal_g)

    if path_g is None:
        game.close()
        sys.exit(1)

    path_g = smooth_path(path_g, grid)
    waypoints = [g2w(gx, gy) for gx, gy in path_g]

    wp_idx = 0
    n_wps = len(waypoints)
    last_pos = (px, py)
    stuck_ctr = 0

    while not game.is_episode_finished():
        state = game.get_state()
        if state is None:
            break

        gv = state.game_variables
        px = float(gv[0])
        py = float(gv[1])
        pangle = float(gv[2])

        moved = math.hypot(px - last_pos[0], py - last_pos[1])
        stuck_ctr = stuck_ctr + 1 if moved < 0.5 else 0
        last_pos = (px, py)

        if stuck_ctr > STUCK_LIMIT:
            game.make_action([0, 0, 0], 15)
            game.make_action([0, 1, 0], 20)
            stuck_ctr = 0
            continue

        if wp_idx >= n_wps:
            break

        tx, ty = waypoints[wp_idx]
        dist_wp = math.hypot(tx - px, ty - py)

        if dist_wp < WP_REACH_WU:
            wp_idx += 1
            if wp_idx >= n_wps:
                break
            tx, ty = waypoints[wp_idx]

        dx = tx - px
        dy = ty - py
        desired = math.degrees(math.atan2(dy, dx))
        err = angle_diff(desired, pangle)

        fwd = 1 if abs(err) < ALIGN_THRESH else 0

        if err > 2.0:
            action = [fwd, 1, 0]
        elif err < -2.0:
            action = [fwd, 0, 1]
        else:
            action = [1, 0, 0]

        game.make_action(action, 1)

    time.sleep(3)
    game.close()


if __name__ == "__main__":
    main()
