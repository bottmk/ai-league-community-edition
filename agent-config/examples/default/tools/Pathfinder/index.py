# Pathfinder Tool - AI League
# ============================
# Route planner for the AI League dungeon game.
#
# Strategies:
#   swift     - BFS shortest path to treasure
#   get_coins - greedy nearest-coin collection, then treasure
#   value     - prize-collecting tour under time and life budgets (default)
#
# This file is deployed as AgentCoreGatewayTool-Pathfinder.
# Handler entrypoint: lambda_handler.lambda_handler

import heapq
import json
from collections import deque

DIRECTIONS = [(-1, 0, "up"), (1, 0, "down"), (0, -1, "left"), (0, 1, "right")]

WALL = "wall"
TREASURE = "treasure"
COIN = "c7"
SPIKE = "c8"

DOOR_TILES = {"c30", "c31", "c32", "c33"}
KEY_TILES = {"c40", "c41", "c42", "c43"}
KEY_TO_DOOR = {"c40": "c30", "c41": "c31", "c42": "c32", "c43": "c33"}
DOOR_TO_KEY = {door: key for key, door in KEY_TO_DOOR.items()}

# One remaining life is worth exactly livesBonusMultiplier at scoring time, so
# every decision below is denominated in points.
LIFE_VALUE = 250

# Values published in the AWS AI League in-game guide (Bonuses / Challenges).
# These govern scoring, so they drive every routing decision below.
#
# c17 and c18 do not appear in the official guide and may not exist; their
# entries are retained from the community edition and are UNVERIFIED. A tile
# type absent from a map simply never gets looked up, so keeping them is inert.
TILE_POINTS = {
    "c1": 400,    # Violent Violet — guardrail test
    "c2": 600,    # Blue Brain — code execution
    "c3": 800,    # Memento — memory
    "c4": 500,    # Dark Prophet — web scraping
    "c5": 250,    # Bonehead — simple question
    "c6": 2000,   # Dungeon boss — web fetch plus computation
    "c7": 250,    # coins, collected by walking over the tile
    "c8": 0,      # spike trap
    "c23": 1000,  # web fetch
    "c17": 750, "c18": 500,  # unverified, see note above
    "c30": 1000, "c31": 1000, "c32": 1000, "c33": 1000,  # doors
    "c40": 50, "c41": 50, "c42": 50, "c43": 50,          # keys
}
TILE_DAMAGE = {
    "c1": 1, "c2": 1, "c3": 1, "c4": 1, "c5": 1, "c6": 2,
    "c7": 0, "c8": 1, "c23": 1,
    "c17": 2, "c18": 1,  # unverified, see note above
    "c30": 5, "c31": 5, "c32": 5, "c33": 5,
    "c40": 0, "c41": 0, "c42": 0, "c43": 0,
}

# Tiles that cost one AgentCore invocation when a challenge is assigned.
INVOKING_TILES = set(TILE_POINTS) - {COIN, SPIKE}

# Per-type success probability. Overridable via the p_correct parameter.
DEFAULT_P_CORRECT = {
    "c1": 0.90, "c2": 0.85, "c3": 0.80, "c4": 0.75, "c5": 0.90, "c6": 0.60,
    "c23": 0.75, "c17": 0.90, "c18": 0.85,
    "c30": 0.80, "c31": 0.80, "c32": 0.80, "c33": 0.80,
    "c40": 1.00, "c41": 1.00, "c42": 1.00, "c43": 1.00,
}

# Step cost used by the segment planner. Spikes stay walkable but cost enough
# that any detour shorter than SPIKE_WEIGHT steps is preferred.
SPIKE_WEIGHT = 60.0

DEFAULTS = {
    "strategy": "value",
    "time_budget": 0.0,     # seconds; 0 disables the time constraint
    "lives": 5,
    "life_margin": 1.0,     # lives to keep in reserve
    "door_policy": "avoid",  # avoid | pass_with_key | answer
    "t_step": 0.05,         # seconds per grid step (one DynamoDB flush)
    "t_invoke": 4.5,        # seconds per challenge invocation
    "max_nodes": 40,
}


def lambda_handler(event, context):
    """
    AWS Lambda function for route planning.
    Handles both API Gateway format and direct AgentCore Gateway format.

    ---
    Tool: route
    Description: Route planner for known maps. Finds the highest-scoring path from start to treasure.
    Parameters:
        game_map     (required) - the 2D grid array as JSON string or list
        start_pos    (optional) - starting position as [row, col], defaults to [0,0] or "start" cell
        strategy     (optional) - routing mode: value, swift, get_coins. Defaults to value
        time_budget  (optional) - seconds available for the run. 0 means unlimited
        lives        (optional) - starting lives, defaults to 5
        door_policy  (optional) - avoid, pass_with_key, or answer. Defaults to avoid
        skip_tiles   (optional) - tile types to leave out of the route, e.g. "c6 c17"
        t_invoke     (optional) - seconds one challenge answer takes. Defaults to 4.5
    ---

    ## Map Definitions
        - "wall": non-walkable cell
        - "treasure": target cell; reaching it ends the game immediately
        - "normal": walkable cell with no special properties
        - "start": the start cell of your avatar, acts as normal cell
        - "c7": Coins that increase score when collected with no challenge
        - "c8": Spikes that reduce health traveled over
        - "c1".."c6", "c17", "c18": challenge cells
        - "c30".."c33": locked doors. Entering without the matching key costs 5 lives
        - "c40".."c43": keys that unlock the matching door

    ## Strategies
        value     - prize-collecting tour honouring time, lives and key/door order (default)
        swift     - BFS shortest path to treasure
        get_coins - greedily collect c7 coins on the way to treasure
    """

    try:
        if 'body' in event:
            body = json.loads(event['body']) if isinstance(event['body'], str) else event['body']
        else:
            body = event

        game_map = body.get('game_map', [])
        start_pos = body.get('start_pos', [0, 0])
        strategy = str(body.get('strategy') or DEFAULTS["strategy"]).strip().lower()
        print(f"DEBUG: strategy={strategy} start_pos={start_pos}")

        # Robustly parse game_map
        map_object = None
        if isinstance(game_map, str):
            game_map = game_map.strip()
            first_bracket = -1
            for i, ch in enumerate(game_map):
                if ch in ('[', '{'):
                    first_bracket = i
                    break
            last_bracket = -1
            for i in range(len(game_map) - 1, -1, -1):
                if game_map[i] in (']', '}'):
                    last_bracket = i
                    break
            if first_bracket >= 0 and last_bracket > first_bracket:
                game_map = game_map[first_bracket:last_bracket + 1]
            try:
                game_map = json.loads(game_map)
            except json.JSONDecodeError:
                return _err(400, f'game_map is not valid JSON: {game_map[:100]}')

        if isinstance(game_map, dict):
            map_object = game_map
            for key in ('grid', 'game_map', 'map'):
                if key in map_object:
                    game_map = map_object[key]
                    break

        # Parse start_pos if it's a string
        if isinstance(start_pos, str):
            try:
                start_pos = json.loads(start_pos)
            except json.JSONDecodeError:
                start_pos = [0, 0]

        # If map_object had a playerStart field, use it
        if map_object and 'playerStart' in map_object:
            ps = map_object['playerStart']
            if isinstance(ps, dict) and 'row' in ps and 'col' in ps:
                start_pos = [int(ps['row']), int(ps['col'])]

        if not game_map or not isinstance(game_map, list):
            return _err(400, 'Missing or invalid game_map')

        # Fallback: find "start" cell in the grid
        if not map_object or 'playerStart' not in (map_object or {}):
            for r_idx, row in enumerate(game_map):
                for c_idx, cell in enumerate(row):
                    if cell == 'start':
                        start_pos = [r_idx, c_idx]
                        break
                else:
                    continue
                break

        rows, cols = len(game_map), len(game_map[0])
        treasure = None
        for r in range(rows):
            for c in range(cols):
                if game_map[r][c] == TREASURE:
                    treasure = (r, c)
                    break
            if treasure:
                break

        if not treasure:
            return _err(400, 'No treasure found on map')

        start = (int(start_pos[0]), int(start_pos[1]))

        # A start on a wall yields an empty path, which silently scores zero.
        # Recover from the grid's own 'start' cell before giving up.
        if not (0 <= start[0] < rows and 0 <= start[1] < cols) or game_map[start[0]][start[1]] == WALL:
            recovered = None
            for r in range(rows):
                for c in range(cols):
                    if game_map[r][c] == 'start':
                        recovered = (r, c)
                        break
                if recovered:
                    break
            if recovered is None:
                return _err(400, f'start_pos {list(start)} is not a walkable cell and the map has no start tile')
            print(f"DEBUG: start_pos was unwalkable, recovered start cell {recovered}")
            start = recovered

        if strategy == 'get_coins':
            path = get_coins_path(game_map, rows, cols, start, treasure)
            result = {'path': path, 'steps': len(path), 'start_position': list(start),
                      'strategy': 'get_coins'}
        elif strategy == 'swift':
            path = swift_path(game_map, rows, cols, start, treasure)
            result = {'path': path, 'steps': len(path), 'start_position': list(start),
                      'strategy': 'swift'}
        else:
            cfg = _read_config(body)
            result = value_path(game_map, rows, cols, start, treasure, cfg)
            result['start_position'] = list(start)

        print(f"RESULT: strategy={result.get('strategy')} steps={result.get('steps')}")
        return {'statusCode': 200, 'body': json.dumps(result)}

    except Exception as e:
        print(f"ERROR: {e}")
        return _err(500, str(e))


def _err(code, msg):
    return {'statusCode': code, 'body': json.dumps({'error': msg})}


# ---------------------------------------------------------------------------
# Parameter parsing
# ---------------------------------------------------------------------------


def _as_float(value, fallback):
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def _as_tile_set(value):
    """Accept a list, a JSON array string, or a space/comma separated string."""
    if not value:
        return set()
    if isinstance(value, (list, tuple, set)):
        return {str(v).strip().lower() for v in value if str(v).strip()}
    text = str(value).strip()
    if text.startswith('['):
        try:
            return {str(v).strip().lower() for v in json.loads(text)}
        except json.JSONDecodeError:
            pass
    return {part.strip().lower() for part in text.replace(',', ' ').split() if part.strip()}


def _read_config(body):
    cfg = dict(DEFAULTS)
    cfg["time_budget"] = max(0.0, _as_float(body.get("time_budget"), DEFAULTS["time_budget"]))
    cfg["lives"] = _as_float(body.get("lives"), DEFAULTS["lives"])
    cfg["life_margin"] = _as_float(body.get("life_margin"), DEFAULTS["life_margin"])
    cfg["t_step"] = _as_float(body.get("t_step"), DEFAULTS["t_step"])
    cfg["t_invoke"] = _as_float(body.get("t_invoke"), DEFAULTS["t_invoke"])
    cfg["max_nodes"] = int(_as_float(body.get("max_nodes"), DEFAULTS["max_nodes"]))

    policy = str(body.get("door_policy") or DEFAULTS["door_policy"]).strip().lower()
    cfg["door_policy"] = policy if policy in ("avoid", "pass_with_key", "answer") else "avoid"

    cfg["skip_tiles"] = _as_tile_set(body.get("skip_tiles"))

    p_correct = dict(DEFAULT_P_CORRECT)
    raw_p = body.get("p_correct")
    if isinstance(raw_p, str):
        try:
            raw_p = json.loads(raw_p)
        except json.JSONDecodeError:
            raw_p = None
    if isinstance(raw_p, dict):
        for tile, prob in raw_p.items():
            value = _as_float(prob, None)
            if value is not None:
                p_correct[str(tile).strip().lower()] = min(1.0, max(0.0, value))
    cfg["p_correct"] = p_correct
    return cfg


# ---------------------------------------------------------------------------
# Tile economics
# ---------------------------------------------------------------------------


def cell_value(tile, cfg):
    """Expected points for stepping on a tile, in the common points currency."""
    if tile == COIN:
        return float(TILE_POINTS[COIN])
    if tile in KEY_TILES:
        return float(TILE_POINTS[tile])
    if tile in DOOR_TILES:
        if cfg["door_policy"] == "answer":
            return _challenge_value(tile, cfg)
        return 0.0  # pass_with_key: the door is a corridor, not a prize
    if tile in TILE_POINTS:
        return _challenge_value(tile, cfg)
    return 0.0


def _challenge_value(tile, cfg):
    p = cfg["p_correct"].get(tile, 0.5)
    points = TILE_POINTS.get(tile, 0)
    damage = TILE_DAMAGE.get(tile, 0)
    return points * p - LIFE_VALUE * damage * (1.0 - p)


def expected_damage(tile, cfg):
    """Expected lives lost by stepping on a tile."""
    if tile == SPIKE:
        return float(TILE_DAMAGE[SPIKE])
    if tile in DOOR_TILES and cfg["door_policy"] != "answer":
        return 0.0  # entered with the key and no challenge assigned
    if tile in TILE_POINTS and tile != COIN:
        p = cfg["p_correct"].get(tile, 0.5)
        return TILE_DAMAGE.get(tile, 0) * (1.0 - p)
    return 0.0


# ---------------------------------------------------------------------------
# Segment planning
# ---------------------------------------------------------------------------


def _walkable(tile, blocked_doors):
    if tile == WALL:
        return False
    if tile in DOOR_TILES and tile in blocked_doors:
        return False
    return True


def _segment(grid, rows, cols, start, goal, blocked_doors, cache, treasure=None):
    """Cheapest walk between two cells, preferring routes with fewer spikes.

    Returns (moves, cells) where cells lists every cell entered after start,
    or None when no walk exists. Dijkstra over step cost plus a spike surcharge.

    The treasure cell ends the game the instant it is entered, so it is
    impassable for every segment that is not aiming at it.
    """
    key = (start, goal, blocked_doors)
    if key in cache:
        return cache[key]

    if start == goal:
        cache[key] = ([], [])
        return cache[key]

    dist = {start: 0.0}
    prev = {}
    heap = [(0.0, start)]
    found = False
    while heap:
        cost, cell = heapq.heappop(heap)
        if cost > dist.get(cell, float('inf')):
            continue
        if cell == goal:
            found = True
            break
        r, c = cell
        for dr, dc, move in DIRECTIONS:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < rows and 0 <= nc < cols):
                continue
            tile = grid[nr][nc]
            if not _walkable(tile, blocked_doors):
                continue
            if treasure is not None and (nr, nc) == treasure and (nr, nc) != goal:
                continue
            step = 1.0 + (SPIKE_WEIGHT if tile == SPIKE else 0.0)
            ncost = cost + step
            if ncost < dist.get((nr, nc), float('inf')):
                dist[(nr, nc)] = ncost
                prev[(nr, nc)] = (cell, move)
                heapq.heappush(heap, (ncost, (nr, nc)))

    if not found:
        cache[key] = None
        return None

    moves, cells = [], []
    cell = goal
    while cell != start:
        parent, move = prev[cell]
        moves.append(move)
        cells.append(cell)
        cell = parent
    moves.reverse()
    cells.reverse()
    cache[key] = (moves, cells)
    return cache[key]


# ---------------------------------------------------------------------------
# Value strategy
# ---------------------------------------------------------------------------


class _Plan:
    __slots__ = ("moves", "score", "time", "damage", "invocations", "value", "spikes")

    def __init__(self, moves, score, time, damage, invocations, value, spikes):
        self.moves = moves
        self.score = score
        self.time = time
        self.damage = damage
        self.invocations = invocations
        self.value = value
        self.spikes = spikes


def _blocked_for(keys_held, cfg):
    """Doors that are impassable given the keys collected so far."""
    if cfg["door_policy"] == "avoid":
        return frozenset(DOOR_TILES)
    return frozenset(door for door in DOOR_TILES if DOOR_TO_KEY[door] not in keys_held)


def _evaluate(grid, rows, cols, tour, cfg, cache, treasure):
    """Walk a node tour and score it, or return None if a segment is unreachable.

    treasure is kept impassable throughout; it is only ever entered when it is
    the goal of the final segment, because entering it ends the game.
    """
    moves = []
    visited = set()
    keys_held = set()
    spikes = 0.0
    damage = 0.0
    value = 0.0
    invocations = 0
    steps = 0

    for index in range(1, len(tour)):
        blocked = _blocked_for(keys_held, cfg)
        segment = _segment(grid, rows, cols, tour[index - 1], tour[index], blocked, cache,
                           treasure=treasure)
        if segment is None:
            return None
        seg_moves, seg_cells = segment
        moves.extend(seg_moves)
        steps += len(seg_moves)

        for cell in seg_cells:
            if cell in visited:
                continue  # consumed_tiles: revisiting a tile has no effect
            visited.add(cell)
            tile = grid[cell[0]][cell[1]]
            if tile == TREASURE:
                continue  # scored separately; reaching it ends the game
            if tile == SPIKE:
                spikes += TILE_DAMAGE[SPIKE]
                damage += TILE_DAMAGE[SPIKE]
                continue
            if tile in KEY_TILES:
                keys_held.add(tile)
            value += cell_value(tile, cfg)
            damage += expected_damage(tile, cfg)
            if tile in INVOKING_TILES:
                invocations += 1

    score = value - LIFE_VALUE * spikes
    elapsed = steps * cfg["t_step"] + invocations * cfg["t_invoke"]
    return _Plan(moves, score, elapsed, damage, invocations, value, spikes)


def _feasible(plan, cfg):
    if plan is None:
        return False
    if cfg["time_budget"] > 0 and plan.time > cfg["time_budget"]:
        return False
    if plan.damage > max(0.0, cfg["lives"] - cfg["life_margin"]):
        return False
    return True


def _candidates(grid, rows, cols, start, treasure, cfg):
    """Valuable cells worth routing through, best first."""
    nodes = []
    for r in range(rows):
        for c in range(cols):
            cell = (r, c)
            if cell in (start, treasure):
                continue
            tile = grid[r][c]
            if tile in (WALL, TREASURE, SPIKE) or tile not in TILE_POINTS:
                continue
            if tile in cfg["skip_tiles"]:
                continue
            if tile in DOOR_TILES and cfg["door_policy"] == "avoid":
                continue
            value = cell_value(tile, cfg)
            if tile in DOOR_TILES and cfg["door_policy"] == "pass_with_key":
                continue  # zero value; only worth entering as a shortcut, never as a goal
            if value <= 0:
                continue
            nodes.append((value, cell))
    nodes.sort(key=lambda item: -item[0])
    return [cell for _, cell in nodes[:cfg["max_nodes"]]]


def _key_cells(grid, rows, cols, start, treasure):
    """Every key cell on the map, row-major, excluding start and treasure."""
    cells = []
    for r in range(rows):
        for c in range(cols):
            if (r, c) in (start, treasure):
                continue
            if grid[r][c] in KEY_TILES:
                cells.append((r, c))
    return cells


def _reach_depth(grid, rows, cols, tour, cfg, cache, treasure):
    """How many leading segments of a tour are actually walkable.

    Keys are picked up as the walk passes over them, so a door met later in the
    tour may well be open by then. Returns len(tour) - 1 when the whole tour is
    walkable, which is exactly when _evaluate returns a plan.
    """
    keys_held = set()
    for index in range(1, len(tour)):
        blocked = _blocked_for(keys_held, cfg)
        segment = _segment(grid, rows, cols, tour[index - 1], tour[index], blocked, cache,
                           treasure=treasure)
        if segment is None:
            return index - 1
        for cell in segment[1]:
            tile = grid[cell[0]][cell[1]]
            if tile in KEY_TILES:
                keys_held.add(tile)
    return len(tour) - 1


def _seed_keys(grid, rows, cols, tour, cfg, cache, treasure):
    """Insert key cells until the tour becomes walkable end to end.

    A treasure walled off behind a locked door makes the bare [start, treasure]
    tour unreachable, and no amount of value-driven insertion fixes that: a key
    is worth 50 points and would never be picked on its own merit. Detouring
    for the key is what makes the treasure reachable at all, so it happens
    first, before anything is chosen for its score.

    An insertion that makes the whole tour walkable wins outright. Failing
    that, the first insertion that at least reaches the newly placed key is
    taken and the search repeats, which is what unlocks a key that is itself
    sitting behind another door.

    Returns (tour, plan, seeded); plan is None when no seeding helped.
    """
    pool = [cell for cell in _key_cells(grid, rows, cols, tour[0], treasure)
            if cell not in tour]
    seeded = []

    while pool:
        stepping_stone = None
        for node in pool:
            for index in range(1, len(tour)):  # never after the treasure
                trial = tour[:index] + [node] + tour[index:]
                depth = _reach_depth(grid, rows, cols, trial, cfg, cache, treasure)
                if depth == len(trial) - 1:
                    plan = _evaluate(grid, rows, cols, trial, cfg, cache, treasure)
                    seeded.append(node)
                    return trial, plan, seeded
                if stepping_stone is None and depth >= index:
                    # The key itself is reachable: collecting it may open the
                    # door standing between us and one of the other keys.
                    stepping_stone = (node, trial)
        if stepping_stone is None:
            break
        node, tour = stepping_stone
        pool.remove(node)
        seeded.append(node)

    return tour, None, seeded


def _insert_by_value(grid, rows, cols, tour, plan, remaining, cfg, cache, treasure,
                     open_ended=False):
    """Greedy insertion: repeatedly splice in the node that gains the most score.

    Every trial tour is fully re-evaluated, so key-before-door order, the life
    budget and the time budget hold for the tour that is returned.

    With open_ended the walk has no fixed terminal and a node may be appended
    after the last one; otherwise the final node is the treasure and nothing may
    be placed after it, because entering the treasure ends the game.
    """
    remaining = list(remaining)
    while remaining:
        best = None
        limit = len(tour) + 1 if open_ended else len(tour)
        for node in remaining:
            for index in range(1, limit):
                trial = tour[:index] + [node] + tour[index:]
                trial_plan = _evaluate(grid, rows, cols, trial, cfg, cache, treasure)
                if not _feasible(trial_plan, cfg):
                    continue
                gain = trial_plan.score - plan.score
                if gain > 0 and (best is None or gain > best[0]):
                    best = (gain, node, trial, trial_plan)
        if best is None:
            break
        _, node, tour, plan = best
        remaining.remove(node)
    return tour, plan


def _unreachable_note(grid, rows, cols, start, treasure, cache, plan):
    """Say why the treasure was given up on, so the failure is visible."""
    if plan is not None:
        return 'treasure not reachable within the time and life budget; collected what fits'
    if _segment(grid, rows, cols, start, treasure, frozenset(), cache, treasure=treasure):
        return ('treasure unreachable without entering a locked door with no key; '
                'stopped short instead of paying 5 lives')
    return 'treasure is walled off from the start; collected what is reachable instead'


def _open_ended_path(grid, rows, cols, start, treasure, cfg, cache, note):
    """Best walk that never enters the treasure, for when the treasure is out of reach.

    Walking into a door without its key costs 5 lives unconditionally, which
    from a 5-life start is instant death and forfeits the whole run. Standing
    still keeps all five lives (1250 points of life bonus) plus the token bonus,
    so anything that ends in a locked door is worse than doing nothing. This
    collects whatever is safely reachable instead and simply stops there.
    """
    plan = _evaluate(grid, rows, cols, [start], cfg, cache, treasure)
    remaining = _candidates(grid, rows, cols, start, treasure, cfg)
    tour, plan = _insert_by_value(grid, rows, cols, [start], plan, remaining, cfg, cache,
                                  treasure, open_ended=True)
    result = _result(plan)
    result['note'] = note
    return result


def _result(plan):
    return {
        'path': plan.moves,
        'steps': len(plan.moves),
        'strategy': 'value',
        'estimated_score': round(plan.score),
        'estimated_time': round(plan.time, 1),
        'challenges': plan.invocations,
        'expected_damage': round(plan.damage, 2),
    }


def value_path(grid, rows, cols, start, treasure, cfg):
    """Prize-collecting tour: start -> valuable cells -> treasure.

    Greedy insertion keeps the tour feasible at every step, so key-before-door
    order, the life budget and the time budget all hold for the returned walk.
    When the treasure cannot be reached at all the walk stays open-ended and
    stops short of the treasure rather than charging a locked door.
    """
    cache = {}
    tour = [start, treasure]
    plan = _evaluate(grid, rows, cols, tour, cfg, cache, treasure)
    seeded = []

    if plan is None and cfg["door_policy"] != "avoid":
        # The treasure is walled off behind a locked door: fetch a key first.
        tour, plan, seeded = _seed_keys(grid, rows, cols, tour, cfg, cache, treasure)

    if not _feasible(plan, cfg):
        return _open_ended_path(grid, rows, cols, start, treasure, cfg, cache,
                                _unreachable_note(grid, rows, cols, start, treasure, cache, plan))

    seeded_set = set(seeded)
    remaining = [cell for cell in _candidates(grid, rows, cols, start, treasure, cfg)
                 if cell not in seeded_set]

    tour, plan = _insert_by_value(grid, rows, cols, tour, plan, remaining, cfg, cache, treasure)

    return _result(plan)


# ---------------------------------------------------------------------------
# Legacy strategies
# ---------------------------------------------------------------------------


def _bfs(game_map, rows, cols, start, goal):
    """BFS shortest path between two points."""
    queue = deque([(start[0], start[1], [])])
    visited = {(start[0], start[1])}
    while queue:
        r, c, path = queue.popleft()
        if (r, c) == goal:
            return path
        for dr, dc, move in DIRECTIONS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and game_map[nr][nc] != WALL and (nr, nc) not in visited:
                visited.add((nr, nc))
                queue.append((nr, nc, path + [move]))
    return None


def swift_path(game_map, rows, cols, start, treasure):
    """BFS shortest path to treasure."""
    return _bfs(game_map, rows, cols, start, treasure) or []


def get_coins_path(game_map, rows, cols, start, treasure):
    """Greedily BFS to the nearest c7 cell, repeatedly, then BFS to treasure."""
    board = [row[:] for row in game_map]
    r, c = start
    full_path = []

    for _ in range(50):
        queue = deque([(r, c, [])])
        visited = {(r, c)}
        targets = []
        while queue:
            cr, cc, p = queue.popleft()
            if board[cr][cc] == COIN and (cr, cc) != (r, c):
                targets.append((max(len(p), 1), p, cr, cc))
            for dr, dc, move in DIRECTIONS:
                nr, nc = cr + dr, cc + dc
                if 0 <= nr < rows and 0 <= nc < cols and board[nr][nc] != WALL and (nr, nc) not in visited:
                    visited.add((nr, nc))
                    queue.append((nr, nc, p + [move]))

        if not targets:
            break
        targets.sort()
        _, path_to, r, c = targets[0]
        full_path.extend(path_to)
        board[r][c] = 'normal'

    path_end = _bfs(board, rows, cols, (r, c), treasure)
    if path_end is not None:
        full_path.extend(path_end)
        return full_path
    return swift_path(game_map, rows, cols, start, treasure)
