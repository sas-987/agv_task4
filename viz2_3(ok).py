#!/usr/bin/env python3
"""
ViZDoom LiDAR-DFS Explorer — Level 2  (v8-tuned)
==================================================

Key fixes over v7
-----------------
1. THRESHOLD SEPARATION — DEPTH_DROP_THRESH (20) is now well below
   MIN_DEPTH_TO_EXPLORE (35).  In v7 they were both 45, so the drive
   stopped when forward depth dropped below 45, and the subsequent scan
   saw ~42 wu forward — still below the candidate threshold → instant
   dead-end even when a real path existed.

2. ONLY MARK EXACT HEADING AS TRIED — v7 marked ±10° around the chosen
   heading, which silently blocked adjacent valid directions.  Now only
   the exact chosen heading (and its direct reverse ±180°) is marked tried
   when the drive succeeds.  On wall/short failure only the exact heading
   is marked.

3. SIDE OPENING — tighter threshold: side must be > SIDE_OPEN_DEPTH (150)
   AND > fwd_depth * SIDE_RATIO (2.0).  This prevents plain corridor walls
   from being flagged as side openings.

4. CHECKPOINT INTERVAL — raised to 350 wu (was 100).  Avoids planting
   junk checkpoints every few steps in long corridors.

5. MIN_NODE_SEPARATION — raised to 250 wu.  Nodes are only planted at
   genuine junctions / side openings, not mid-corridor.

6. REVISIT_RADIUS — raised to 130 wu.  Prevents the loop guard from
   being bypassed by nodes that are nearly co-located.

Sensors used (Level 2 compliant)
---------------------------------
  depth_buffer  — LiDAR rays, forward depth, peripheral side depth (no turning)
  POSITION_X / POSITION_Y / ANGLE  — odometry
  objects_info  — goal detection only
"""

import os, sys, math, time, heapq
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import vizdoom as vzd


# ====================================================================
#  PARAMETERS
# ====================================================================

# LiDAR scan
SCAN_STEP_DEG        = 10       # angular resolution per ray (degrees)
SCAN_TICS_PER_STEP   = 1        # tics to rotate per SCAN_STEP_DEG
LIDAR_CENTER_FRAC    = 0.01     # screen-width fraction for center-depth column
LIDAR_DEPTH_MAX      = 800.0    # depth clamp (world units)

# Movement
ALIGN_DEG            = 4.0      # heading error tolerance before moving forward
TICS_PER_TURN        = 1        # tics per turn action
TICS_PER_FWD         = 1        # tics per forward action
WALL_PROBE_TICKS     = 8        # tics of forward push for wall test
WALL_DELTA_WU        = 2.5      # displacement below this = wall

# !! CRITICAL: DEPTH_DROP_THRESH must be LOWER than MIN_DEPTH_TO_EXPLORE !!
# Drive stops when forward depth < DEPTH_DROP_THRESH.
# After stopping, scan sees forward depth just above DEPTH_DROP_THRESH.
# MIN_DEPTH_TO_EXPLORE must be <= that value to keep the forward direction as a candidate.
DEPTH_DROP_THRESH    = 60.0     # stop when wall is within this distance — don't drive into the wall
JUNCTION_CONFIRM     = 4        # consecutive low-depth ticks to confirm (filters noise)
MIN_DEPTH_TO_EXPLORE = 35.0     # LiDAR candidate threshold

MIN_NODE_SEPARATION  = 250.0    # wu to travel before a terminal node may be planted
CRAMPED_DEPTH_THRESH = 60.0     # if max lidar depth at new node < this → genuine dead end → backtrack
                                 # (92wu IS a real corridor; only truly tiny alcoves score < 60)

# Side / room-center detection (no turning needed)
SIDE_BUFFER_FRAC       = 0.12   # outer screen fraction used as side-depth sensor
CHECKPOINT_INTERVAL_WU = 80     # sample side depths every 80wu — catch rooms mid-corridor
                                 # (350wu was far too coarse; entire rooms were passed unseen)
SIDE_OPEN_DEPTH        = 55.0   # either side > this wu → plant a checkpoint (no ratio gate)
                                 # ratio gate removed: when fwd is also open the ratio never triggers
ROOM_SIDE_THRESH       = 100.0  # BOTH sides > this → open room → stop & scan at center
SIDE_RATIO              = 1.0     # side must be > fwd * this ratio to count as opening

# Loop guard
REVISIT_RADIUS       = 130.0    # scanned node within this wu of projected endpoint → skip

# Navigation
NODE_ARRIVE_WU       = 40.0     # arrival acceptance radius
GOAL_RADIUS          = 100.0    # success radius (world units)
RENDER_DELAY         = 0.05     # seconds per action (controls playback speed)
EPISODE_TIMEOUT      = 150000

GOAL_NAMES = {"BlueSkull", "BlueCard", "BlueSkulltag"}


# ====================================================================
#  HELPERS
# ====================================================================

def adiff(target, cur):
    """Signed angular difference target - cur, wrapped to [-180, 180]."""
    d = (target - cur) % 360.0
    return d - 360.0 if d > 180.0 else d

def dist2(x1, y1, x2, y2):
    return math.hypot(x2 - x1, y2 - y1)

def project_point(x, y, hdg_deg, wu):
    """Project (x, y) by wu world-units along heading hdg_deg (Doom convention)."""
    rad = math.radians(hdg_deg)
    return x + wu * math.cos(rad), y + wu * math.sin(rad)

def get_pose(game):
    if game.is_episode_finished():
        return None
    s = game.get_state()
    if s is None:
        return None
    gv = s.game_variables
    return float(gv[0]), float(gv[1]), float(gv[2])

def get_full_state(game):
    if game.is_episode_finished():
        return None
    s = game.get_state()
    if s is None:
        return None
    gv = s.game_variables
    return float(gv[0]), float(gv[1]), float(gv[2]), s

def act(game, action, ticks=1):
    if not game.is_episode_finished():
        game.make_action(action, ticks)
        time.sleep(RENDER_DELAY)

FWD   = [1, 0, 0, 0]
BACK  = [0, 1, 0, 0]
LEFT  = [0, 0, 1, 0]
RIGHT = [0, 0, 0, 1]


# ====================================================================
#  DATA STRUCTURES
# ====================================================================

_node_id_counter = 0

@dataclass
class Node:
    nid        : int
    x          : float
    y          : float
    lidar      : dict = field(default_factory=dict)   # {heading_deg: depth}
    tried      : set  = field(default_factory=set)    # headings already probed
    parent_id  : int  = -1
    parent_node: Optional[object] = field(default=None, repr=False)
    scanned    : bool = True   # False = checkpoint: scan lazily when DFS visits

    def best_untried(self):
        """Return the absolute deepest untried heading."""
        candidates = sorted(
            [(d, h) for h, d in self.lidar.items()
             if h not in self.tried and d >= MIN_DEPTH_TO_EXPLORE],
            reverse=True
        )
        return candidates[0][1] if candidates else None


def new_node(x, y, lidar, parent_node=None, scanned=True):
    global _node_id_counter
    _node_id_counter += 1
    pid = parent_node.nid if parent_node else -1
    return Node(
        nid=_node_id_counter, x=x, y=y, lidar=lidar,
        parent_id=pid, parent_node=parent_node, scanned=scanned
    )


# ====================================================================
#  DEPTH READING
# ====================================================================

def read_center_depth(depth_buffer):
    """Min depth in a narrow center column band = forward-pointing LiDAR ray."""
    if depth_buffer is None:
        return LIDAR_DEPTH_MAX
    H, W = depth_buffer.shape
    cx   = W // 2
    half = max(1, int(W * LIDAR_CENTER_FRAC))
    band = depth_buffer[int(H * 0.45):int(H * 0.55),
                        max(0, cx - half):min(W - 1, cx + half)]
    return min(float(band.min()), LIDAR_DEPTH_MAX)


def read_side_depths(depth_buffer):
    """
    Read left and right peripheral depths WITHOUT turning.
    Uses the outermost SIDE_BUFFER_FRAC of screen columns.
    Returns (left_depth, right_depth) as 60th-percentile (noise-robust).
    """
    if depth_buffer is None:
        return 0.0, 0.0
    H, W   = depth_buffer.shape
    rows   = slice(int(H * 0.35), int(H * 0.65))
    cw     = max(1, int(W * SIDE_BUFFER_FRAC))
    left_d = float(np.percentile(depth_buffer[rows, :cw],   60))
    right_d= float(np.percentile(depth_buffer[rows, W-cw:], 60))
    return min(left_d, LIDAR_DEPTH_MAX), min(right_d, LIDAR_DEPTH_MAX)


# ====================================================================
#  LIDAR SCAN — full 360° polar depth map
# ====================================================================

def lidar_scan(game):
    """
    Rotate precisely to absolute headings [0, 10, 20... 350].
    Reads center depth at each exact heading.
    """
    r = get_full_state(game)
    if r is None:
        return {}, 0.0
    _, _, start_angle, _ = r
    polar = {}

    print("    [Scan] Performing exact 360° LiDAR sweep...")
    
    # Force the agent to specifically face every 10-degree mark on the compass
    for hdg in range(0, 360, SCAN_STEP_DEG):
        turn_to(game, hdg)  # Use the compass controller to guarantee the angle
        
        r = get_full_state(game)
        if r is None:
            break
        _, _, cur_angle, s = r
        
        # Read the depth exactly at this heading
        depth = read_center_depth(s.depth_buffer)
        polar[hdg] = depth

    # Snap back to original heading when done
    turn_to(game, start_angle)
    
    return polar, start_angle


def print_lidar(polar):
    top5  = sorted(polar.items(), key=lambda kv: -kv[1])[:5]
    parts = [f"{h}°:{d:.0f}" for h, d in top5]
    print(f"    LiDAR top-5: {', '.join(parts)}")


# ====================================================================
#  TURN CONTROLLER
# ====================================================================

def turn_to(game, target_deg, max_steps=120):
    for _ in range(max_steps):
        r = get_pose(game)
        if r is None:
            return False
        _, _, angle = r
        err = adiff(target_deg, angle)
        if abs(err) <= ALIGN_DEG:
            return True
        act(game, LEFT if err > 0 else RIGHT, TICS_PER_TURN)
    return False


# ====================================================================
#  WALL PROBE
# ====================================================================

def probe_wall(game, heading_deg):
    """
    Face heading, push WALL_PROBE_TICKS forward, measure displacement.
    Returns (is_wall: bool, disp: float).  Does NOT back out.
    """
    turn_to(game, heading_deg)
    r = get_pose(game)
    if r is None:
        return True, 0.0
    x0, y0, _ = r
    act(game, FWD, WALL_PROBE_TICKS)
    r = get_pose(game)
    if r is None:
        return True, 0.0
    disp = dist2(x0, y0, r[0], r[1])
    return disp < WALL_DELTA_WU, disp


# ====================================================================
#  NAVIGATE TO WORLD POSITION
# ====================================================================

def navigate_to(game, tx, ty, max_steps=500):
    for _ in range(max_steps):
        r = get_pose(game)
        if r is None:
            return False
        px, py, angle = r
        if dist2(px, py, tx, ty) < NODE_ARRIVE_WU:
            return True
        desired = math.degrees(math.atan2(ty - py, tx - px))
        err     = adiff(desired, angle)
        if abs(err) > ALIGN_DEG:
            act(game, LEFT if err > 0 else RIGHT, TICS_PER_TURN)
        else:
            x0, y0 = px, py
            act(game, FWD, TICS_PER_FWD)
            r2 = get_pose(game)
            if r2 is None:
                return False
            if dist2(x0, y0, r2[0], r2[1]) < WALL_DELTA_WU * 0.5:
                act(game, LEFT, 6)   # unstuck wiggle
    r = get_pose(game)
    return r is not None and dist2(r[0], r[1], tx, ty) < NODE_ARRIVE_WU * 2


def fast_navigate_to(game, tx, ty, max_steps=800):
    """
    Same as navigate_to but sends 4 forward tics per action.
    Used for backtracking over already-known safe corridors so the agent
    doesn't waste 13+ minutes crawling back to the parent node.
    """
    FWD_FAST_TICKS = 4
    for _ in range(max_steps):
        r = get_pose(game)
        if r is None:
            return False
        px, py, angle = r
        if dist2(px, py, tx, ty) < NODE_ARRIVE_WU:
            return True
        desired = math.degrees(math.atan2(ty - py, tx - px))
        err     = adiff(desired, angle)
        if abs(err) > ALIGN_DEG:
            act(game, LEFT if err > 0 else RIGHT, TICS_PER_TURN)
        else:
            x0, y0 = px, py
            act(game, FWD, FWD_FAST_TICKS)
            r2 = get_pose(game)
            if r2 is None:
                return False
            if dist2(x0, y0, r2[0], r2[1]) < WALL_DELTA_WU * 0.5:
                act(game, LEFT, 6)   # unstuck wiggle
    r = get_pose(game)
    return r is not None and dist2(r[0], r[1], tx, ty) < NODE_ARRIVE_WU * 2


# ====================================================================
#  A* ON NODE GRAPH  (backtracking)
# ====================================================================

def astar_nodes(all_nodes, start_nid, goal_nid):
    """
    A* over the parent-child node graph.
    Returns list of (x, y) waypoints from start to goal.
    """
    nmap = {n.nid: n for n in all_nodes}
    adj  = {}
    for n in all_nodes:
        if n.parent_node is not None:
            p = n.parent_node
            d = dist2(n.x, n.y, p.x, p.y)
            adj.setdefault(n.nid, []).append((p.nid, d))
            adj.setdefault(p.nid, []).append((n.nid, d))

    if start_nid == goal_nid:
        return []
    goal_node = nmap.get(goal_nid)
    if goal_node is None:
        return []

    def h(nid):
        nd = nmap[nid]
        return dist2(nd.x, nd.y, goal_node.x, goal_node.y)

    heap   = [(h(start_nid), 0.0, start_nid)]
    came   = {start_nid: None}
    g_map  = {start_nid: 0.0}
    closed = set()

    while heap:
        _, g, cur = heapq.heappop(heap)
        if cur in closed:
            continue
        closed.add(cur)
        if cur == goal_nid:
            path, c = [], cur
            while came[c] is not None:
                path.append(nmap[c])
                c = came[c]
            path.reverse()
            return [(n.x, n.y) for n in path]
        for nb_nid, ed in adj.get(cur, []):
            if nb_nid in closed:
                continue
            ng = g + ed
            if ng < g_map.get(nb_nid, 1e18):
                g_map[nb_nid] = ng
                came[nb_nid]  = cur
                heapq.heappush(heap, (ng + h(nb_nid), ng, nb_nid))
    return []


# ====================================================================
#  FORWARD DRIVE — side-opening checkpoints + room-center stop
# ====================================================================

def drive_to_max(game, heading_deg, all_nodes, max_ticks=8000):
    """
    Drive along heading_deg until one of four stop conditions:

      'ROOM_CENTER'  — BOTH side depths > ROOM_SIDE_THRESH (open room detected).
                       Stop here and scan: best position to see all room exits.
      'JUNCTION'     — Forward depth < DEPTH_DROP_THRESH for JUNCTION_CONFIRM ticks.
      'WALL'         — Position delta < threshold (physically stopped by wall).
      'END'          — max_ticks exhausted.

    While driving, side depths are sampled every CHECKPOINT_INTERVAL_WU.
    If either side > SIDE_OPEN_DEPTH a side-corridor checkpoint is recorded at
    that position — DFS will navigate back and scan it later.

    The room-center check also guards against planting duplicate nodes: if an
    existing node already sits near the detected room center, the drive continues
    to the next stop condition instead of terminating prematurely.

    Returns: (side_stops, final_pos, reason, travelled_wu)
    """
    turn_to(game, heading_deg)
    r = get_pose(game)
    if r is None:
        return [], (0, 0), 'FAIL', 0.0

    x0, y0, _   = r
    total        = 0.0
    junc_streak  = 0
    last_ckpt_wu = 0.0
    side_stops   = []

    for _ in range(max_ticks):
        r = get_full_state(game)
        if r is None:
            break
        px, py, angle, s = r
        travelled = dist2(x0, y0, px, py)
        fwd_d     = read_center_depth(s.depth_buffer)

        # ── Sample sides every CHECKPOINT_INTERVAL_WU ────────────────────
        # NOTE: no `travelled > MIN_NODE_SEPARATION` guard here — we must detect
        # side openings from the very start of the drive, not only after 250wu.
        if (travelled - last_ckpt_wu >= CHECKPOINT_INTERVAL_WU
                and travelled > 50.0):          # just skip the first ~50wu near the parent
            last_ckpt_wu = travelled
            left_d, right_d = read_side_depths(s.depth_buffer)

            # ── Room-center stop: both sides open ─────────────────────────
            if left_d > ROOM_SIDE_THRESH and right_d > ROOM_SIDE_THRESH:
                already_mapped = any(
                    dist2(px, py, n.x, n.y) < REVISIT_RADIUS
                    for n in all_nodes
                )
                if not already_mapped:
                    print(f"    [ROOM] Center at ({px:.0f},{py:.0f})"
                          f"  L={left_d:.0f} R={right_d:.0f} fwd={fwd_d:.0f} — stopping")
                    return side_stops, (px, py), 'ROOM_CENTER', travelled

            # ── Side corridor: EITHER side open — no ratio gate ───────────
            # Ratio gate removed: when forward is also open the ratio never
            # triggers, causing entire rooms on the side to be missed.
            side_open = (left_d > SIDE_OPEN_DEPTH) or (right_d > SIDE_OPEN_DEPTH)
            if side_open and not any(
                dist2(px, py, n.x, n.y) < REVISIT_RADIUS for n in all_nodes
            ):
                side_stops.append((px, py))
                print(f"    [SIDE] Opening at ({px:.0f},{py:.0f})"
                      f"  L={left_d:.0f} R={right_d:.0f} fwd={fwd_d:.0f}")

        # ── Junction detection ─────────────────────────────────────────────
        if fwd_d < DEPTH_DROP_THRESH and travelled > MIN_NODE_SEPARATION:
            junc_streak += 1
            if junc_streak >= JUNCTION_CONFIRM:
                return side_stops, (px, py), 'JUNCTION', travelled
        else:
            junc_streak = 0

        # ── Heading correction ────────────────────────────────────────────
        err = adiff(heading_deg, angle)
        if abs(err) > ALIGN_DEG:
            act(game, LEFT if err > 0 else RIGHT, TICS_PER_TURN)
            continue

        # ── Forward step + wall check ──────────────────────────────────────
        xb, yb = px, py
        act(game, FWD, TICS_PER_FWD)
        r2 = get_pose(game)
        if r2 is None:
            break
        step = dist2(xb, yb, r2[0], r2[1])
        total += step
        if step < WALL_DELTA_WU * 0.4:
            return side_stops, (r2[0], r2[1]), 'WALL', total

    r   = get_pose(game)
    pos = (r[0], r[1]) if r else (x0, y0)
    return side_stops, pos, 'END', total


# ====================================================================
#  MAIN EXPLORER
# ====================================================================

def explore(game, goal_pos):
    print("[Explorer] Scanning spawn position ...")
    polar, _ = lidar_scan(game)
    print_lidar(polar)

    r = get_pose(game)
    if r is None:
        return False
    root      = new_node(r[0], r[1], polar)
    dfs_stack = [root]
    all_nodes = [root]
    print(f"\n[DFS] Root #{root.nid}  ({root.x:.0f},{root.y:.0f})\n")

    goal_found = False

    while dfs_stack and not game.is_episode_finished():
        current = dfs_stack[-1]

        # ── Lazy scan: checkpoint nodes need to be navigated to + scanned ──
        if not current.scanned:
            print(f"\n  [VISIT] Checkpoint #{current.nid}"
                  f" ({current.x:.0f},{current.y:.0f}) — navigating & scanning ...")
            fast_navigate_to(game, current.x, current.y)
            r = get_pose(game)
            if r:
                current.x, current.y = r[0], r[1]   # snap to actual position
            polar, _ = lidar_scan(game)
            current.lidar   = polar
            current.scanned = True
            print_lidar(polar)

        # ── Goal check ────────────────────────────────────────────────────
        r = get_pose(game)
        if r and goal_pos and dist2(r[0], r[1], goal_pos[0], goal_pos[1]) < GOAL_RADIUS:
            print(f"\n[DFS]  GOAL REACHED at Node #{current.nid}!")
            goal_found = True
            break

        # ── Best untried direction (pre-ranked deepest first, loop-guarded) ─
        heading = current.best_untried()

        if heading is None:
            # Dead end — backtrack
            if len(dfs_stack) == 1:
                print("[DFS] Root exhausted — full map explored.")
                break

            print(f"\n  [DEAD END] #{current.nid} ({current.x:.0f},{current.y:.0f})"
                  f"  tried={sorted(current.tried)}")
            dfs_stack.pop()
            parent = dfs_stack[-1]
            print(f"  [BACK] → #{parent.nid} ({parent.x:.0f},{parent.y:.0f})")

            waypoints = astar_nodes(all_nodes, current.nid, parent.nid)
            if waypoints:
                print(f"  [A*] {len(waypoints)} waypoints back")
                for wx, wy in waypoints:
                    fast_navigate_to(game, wx, wy)
            else:
                fast_navigate_to(game, parent.x, parent.y)
            continue

        # ── Mark heading tried (exact only — ±10° was silently blocking neighbors) ─
        current.tried.add(heading)

        # ── Wall probe ─────────────────────────────────────────────────────
        is_wall, disp = probe_wall(game, heading)
        if is_wall:
            print(f"  [WALL] heading={heading}  disp={disp:.1f}wu — skipping")
            fast_navigate_to(game, current.x, current.y)
            continue

        # ── Drive forward ──────────────────────────────────────────────────
        print(f"  [GO  ] heading={heading}  probe_disp={disp:.1f}wu ...")
        side_stops, (nx, ny), reason, travelled = drive_to_max(
            game, heading, all_nodes
        )

        if travelled < MIN_NODE_SEPARATION:
            print(f"  [SHORT] Only {travelled:.1f}wu ({reason}) — trying next direction")
            fast_navigate_to(game, current.x, current.y)
            continue

        # ── Scan at terminal position ─────────────────────────────────────
        print(f"  [SCAN] {reason} at ({nx:.0f},{ny:.0f})"
              f"  moved={travelled:.0f}wu ...")
        polar, _ = lidar_scan(game)
        print_lidar(polar)

        # ── Cramped-room guard ────────────────────────────────────────────
        max_depth_here = max(polar.values()) if polar else 0.0
        if max_depth_here < CRAMPED_DEPTH_THRESH:
            print(f"  [CRAMPED] Max depth only {max_depth_here:.0f}wu < {CRAMPED_DEPTH_THRESH}wu"
                  f" — dead room, backtracking to #{current.nid}")
            fast_navigate_to(game, current.x, current.y)
            continue   # stay on current node; best_untried() picks next direction

        r = get_pose(game)
        if r is None:
            break
        nx, ny = r[0], r[1]

        # ── Build node chain: [side checkpoints] → terminal ────────────────
        #
        # Checkpoints are unscanned; they are pushed BELOW terminal on the
        # DFS stack so the terminal is explored first.  When the terminal is
        # exhausted and popped, DFS backtracks to each checkpoint in order,
        # navigates there, scans lazily, and explores.  The parent chain is:
        #   current → ckpt_1 → ckpt_2 → ... → terminal
        # A* can route through this chain for backtracking.

        chain_parent = current
        new_chain    = []

        for sx, sy in side_stops:
            if any(dist2(sx, sy, n.x, n.y) < REVISIT_RADIUS for n in all_nodes):
                continue   # area already covered
            cp = new_node(sx, sy, {}, parent_node=chain_parent, scanned=False)
            all_nodes.append(cp)
            new_chain.append(cp)
            chain_parent = cp
            print(f"  [CKPT] #{cp.nid} ({sx:.0f},{sy:.0f})"
                  f"  parent=#{cp.parent_id}  [lazy scan on visit]")

        # Terminal node (fully scanned right now)
        term_too_close = any(
            dist2(nx, ny, n.x, n.y) < REVISIT_RADIUS
            for n in all_nodes
            if n.nid != current.nid
        )
        if not term_too_close:
            cands    = sum(1 for d in polar.values() if d >= MIN_DEPTH_TO_EXPLORE)
            terminal = new_node(nx, ny, polar, parent_node=chain_parent)
            all_nodes.append(terminal)
            new_chain.append(terminal)
            # Mask the hallway we just arrived from so the child doesn't drive backward
            for offset in [-20, -10, 0, 10, 20]:
                terminal.tried.add((heading + 180 + offset) % 360)
            print(f"  [NODE] #{terminal.nid} ({nx:.0f},{ny:.0f})"
                  f"  parent=#{chain_parent.nid}  open_dirs={cands}"
                  f"  total={len(all_nodes)}")
        else:
            print(f"  [SKIP] Terminal ({nx:.0f},{ny:.0f}) too close to existing node"
                  f" — returning to current")
            fast_navigate_to(game, current.x, current.y)

        # Push chain (terminal ends up on top — explored first)
        for node in new_chain:
            dfs_stack.append(node)

        time.sleep(RENDER_DELAY * 4)

    return goal_found


# ====================================================================
#  VIZDOOM SETUP
# ====================================================================

def make_game(wad_path):
    game = vzd.DoomGame()
    game.set_doom_scenario_path(wad_path)
    game.set_doom_map("MAP01")
    game.set_mode(vzd.Mode.PLAYER)
    game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)
    game.set_screen_format(vzd.ScreenFormat.RGB24)
    game.set_render_hud(False)
    game.set_render_weapon(False)
    game.set_render_crosshair(False)
    game.set_render_decals(False)
    game.set_render_particles(False)
    game.set_window_visible(True)
    game.set_sound_enabled(False)
    game.set_automap_buffer_enabled(False)
    game.set_sectors_info_enabled(False)
    game.set_depth_buffer_enabled(True)
    game.set_objects_info_enabled(True)
    game.set_available_buttons([
        vzd.Button.MOVE_FORWARD,
        vzd.Button.MOVE_BACKWARD,
        vzd.Button.TURN_LEFT,
        vzd.Button.TURN_RIGHT,
    ])
    game.add_available_game_variable(vzd.GameVariable.POSITION_X)
    game.add_available_game_variable(vzd.GameVariable.POSITION_Y)
    game.add_available_game_variable(vzd.GameVariable.ANGLE)
    game.set_doom_skill(1)
    game.set_episode_timeout(EPISODE_TIMEOUT)
    game.init()
    return game


# ====================================================================
#  MAIN
# ====================================================================

def main():
    wad_path = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else "MAP01.wad")
    if not os.path.isfile(wad_path):
        sys.exit(f"[Error] WAD not found: {wad_path}")

    print("=" * 65)
    print("  ViZDoom LiDAR-DFS Explorer — Level 2  (v8-tuned)")
    print("=" * 65)
    print(f"  WAD            : {wad_path}")
    print(f"  LiDAR          : {SCAN_STEP_DEG}°  ({int(360 / SCAN_STEP_DEG)} rays/scan)")
    print(f"  Wall detect    : pos-delta < {WALL_DELTA_WU} wu")
    print(f"  Depth thresh   : drop={DEPTH_DROP_THRESH} wu  explore={MIN_DEPTH_TO_EXPLORE} wu")
    print(f"  Checkpoints    : every {CHECKPOINT_INTERVAL_WU} wu"
          f"  (side > {SIDE_OPEN_DEPTH} wu  &  {SIDE_RATIO:.0f}x fwd)")
    print(f"  Room center    : both sides > {ROOM_SIDE_THRESH} wu → stop + scan")
    print(f"  Loop guard     : scanned node < {REVISIT_RADIUS} wu of projected endpoint")
    print(f"  Min separation : {MIN_NODE_SEPARATION} wu between nodes")
    print()

    game = make_game(wad_path)
    print("[Init] ViZDoom ready.\n")

    game.new_episode()
    state      = game.get_state()
    gv         = state.game_variables
    px, py, ang = float(gv[0]), float(gv[1]), float(gv[2])
    print(f"[Init] Spawn ({px:.0f}, {py:.0f})  angle={ang:.1f}°\n")

    goal_pos = None
    print("[Objects]")
    for obj in state.objects:
        is_g = obj.name in GOAL_NAMES or "blue" in obj.name.lower()
        tag  = "  ← GOAL" if is_g else ""
        print(f"  {obj.name:30s}  ({obj.position_x:7.0f},{obj.position_y:7.0f}){tag}")
        if is_g and goal_pos is None:
            goal_pos = (float(obj.position_x), float(obj.position_y))

    if goal_pos:
        print(f"\n[Goal] ({goal_pos[0]:.0f}, {goal_pos[1]:.0f})\n")
    else:
        print("\n[Goal] None — full exploration mode.\n")

    t0      = time.time()
    found   = explore(game, goal_pos)
    elapsed = time.time() - t0

    print("\n" + "=" * 65)
    print(f"  RESULT  : {'GOAL FOUND' if found else 'Exploration complete.'}")
    print(f"  Elapsed : {elapsed:.1f} s")
    print(f"  Nodes   : {_node_id_counter}")
    print("=" * 65)

    time.sleep(8)
    game.close()
    print("[Done]")


if __name__ == "__main__":
    main()