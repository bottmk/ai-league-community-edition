"""Tests for the Pathfinder tool's value strategy.

The tool is deployed from two identical copies (lambda/pathfinder-tool and
agent-config/tools/Pathfinder), so both are loaded and exercised.

The central invariant is that a returned walk must survive replay under the
same rules game_runner applies: it never enters a wall, never enters a door
without the matching key, and touches the treasure only as its final step
(reaching the treasure ends the run, forfeiting everything not yet collected).
"""

import importlib.util
import json
import os

import pytest

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

_TOOL_PATHS = {
    "lambda": os.path.join(_REPO_ROOT, "lambda", "pathfinder-tool", "index.py"),
    "agent-config": os.path.join(_REPO_ROOT, "agent-config", "tools", "Pathfinder", "index.py"),
}

_MOVES = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}


def _load(path, name):
    spec = importlib.util.spec_from_file_location(f"pathfinder_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(params=sorted(_TOOL_PATHS), ids=sorted(_TOOL_PATHS))
def pathfinder(request):
    path = _TOOL_PATHS[request.param]
    if not os.path.exists(path):
        pytest.skip(f"tool copy not present: {path}")
    return _load(path, request.param)


def _route(pathfinder, grid, start, **params):
    body = {"game_map": grid, "start_pos": list(start), "strategy": "value"}
    body.update(params)
    response = pathfinder.lambda_handler(body, None)
    return response["statusCode"], json.loads(response["body"])


def _replay(pathfinder, grid, start, path):
    """Replay a walk the way game_runner does; return a list of rule violations.

    Mirrors run_game_session_v2: consumed tiles have no effect on revisit,
    walls and off-grid steps are fatal, and the treasure breaks the loop.
    """
    rows, cols = len(grid), len(grid[0])
    row, col = start
    consumed = set()
    keys = set()
    problems = []

    for index, move in enumerate(path):
        delta_row, delta_col = _MOVES[move]
        row, col = row + delta_row, col + delta_col

        if not (0 <= row < rows and 0 <= col < cols):
            problems.append(f"step {index}: left the grid at {(row, col)}")
            return problems
        tile = grid[row][col]
        if tile == "wall":
            problems.append(f"step {index}: walked into a wall at {(row, col)}")
            return problems

        if (row, col) in consumed:
            continue
        consumed.add((row, col))

        if tile == "treasure":
            if index != len(path) - 1:
                problems.append(
                    f"step {index}: reached the treasure with {len(path) - index - 1} steps left"
                )
            return problems

        if tile in pathfinder.KEY_TILES:
            keys.add(tile)
        elif tile in pathfinder.DOOR_TILES and pathfinder.DOOR_TO_KEY[tile] not in keys:
            problems.append(f"step {index}: entered door {tile} at {(row, col)} without its key")

    problems.append("walk ended without reaching the treasure")
    return problems


def _trace(grid, start, path):
    """Every cell the walk enters, in order."""
    row, col = start
    cells = []
    for move in path:
        delta_row, delta_col = _MOVES[move]
        row, col = row + delta_row, col + delta_col
        cells.append((row, col))
    return cells


# ---------------------------------------------------------------------------
# Fixtures — small grids, each isolating one rule
# ---------------------------------------------------------------------------

# The treasure sits between the start and a coin: a shortest-path planner walks
# onto it and ends the run, forfeiting the coin. The second row is the detour
# that makes the coin reachable without crossing the treasure.
TREASURE_IN_THE_WAY = [
    ["start", "normal", "treasure", "normal", "c7"],
    ["normal", "normal", "normal", "normal", "normal"],
]

# The direct corridor is paved with spikes; the longer way round is clean.
SPIKE_DETOUR = [
    ["start", "c8", "c8", "c8", "treasure"],
    ["normal", "normal", "normal", "normal", "normal"],
]

# The key sits past the treasure's corridor, so the door is only openable
# after a detour that a naive planner would never take.
KEY_BEFORE_DOOR = [
    ["start", "normal", "normal", "normal", "c40"],
    ["wall", "wall", "c30", "wall", "wall"],
    ["c7", "c7", "c7", "c7", "treasure"],
]

# The only way to the treasure runs through a locked door, and the matching key
# is sitting in plain sight. The planner has to detour for it: the key is worth
# 50 points and would never be chosen on its own merit.
TREASURE_BEHIND_A_DOOR = [
    ["start", "normal", "c40", "normal"],
    ["normal", "wall", "wall", "wall"],
    ["c7", "normal", "c30", "treasure"],
    ["wall", "wall", "wall", "wall"],
]

# Same shape, but no key exists anywhere. Entering the door costs 5 lives
# outright, which from a 5-life start forfeits the entire run.
TREASURE_BEHIND_A_DOOR_WITH_NO_KEY = [
    ["start", "normal", "c7", "normal"],
    ["normal", "wall", "wall", "wall"],
    ["c7", "normal", "c30", "treasure"],
    ["wall", "wall", "wall", "wall"],
]

COINS_AND_CHALLENGES = [
    ["start", "c7", "c7", "c5", "normal"],
    ["normal", "wall", "wall", "wall", "c8"],
    ["c7", "c7", "c4", "normal", "treasure"],
]


class TestWalkValidity:
    """Every returned walk must survive replay under game_runner's rules."""

    @pytest.mark.parametrize(
        "grid,start",
        [
            (TREASURE_IN_THE_WAY, (0, 0)),
            (SPIKE_DETOUR, (0, 0)),
            (KEY_BEFORE_DOOR, (0, 0)),
            (COINS_AND_CHALLENGES, (0, 0)),
        ],
    )
    def test_walk_replays_cleanly(self, pathfinder, grid, start):
        status, result = _route(pathfinder, grid, start, door_policy="pass_with_key")
        assert status == 200, result
        assert _replay(pathfinder, grid, start, result["path"]) == []

    def test_treasure_is_only_touched_last(self, pathfinder):
        """A coin behind the treasure must be collected before the run ends."""
        status, result = _route(pathfinder, TREASURE_IN_THE_WAY, (0, 0))
        assert status == 200, result
        assert _replay(pathfinder, TREASURE_IN_THE_WAY, (0, 0), result["path"]) == []
        # The coin at (0,4) is only reachable around the second row, so the walk
        # has to detour for it and double back to finish on the treasure.
        visited = _trace(TREASURE_IN_THE_WAY, (0, 0), result["path"])
        assert (0, 4) in visited, visited
        assert visited[-1] == (0, 2), visited

    def test_spikes_are_routed_around(self, pathfinder):
        status, result = _route(pathfinder, SPIKE_DETOUR, (0, 0))
        assert status == 200, result
        assert _replay(pathfinder, SPIKE_DETOUR, (0, 0), result["path"]) == []
        assert result["expected_damage"] == 0

    def test_key_is_collected_before_its_door(self, pathfinder):
        status, result = _route(pathfinder, KEY_BEFORE_DOOR, (0, 0), door_policy="pass_with_key")
        assert status == 200, result
        assert _replay(pathfinder, KEY_BEFORE_DOOR, (0, 0), result["path"]) == []

    def test_a_key_is_fetched_to_make_the_treasure_reachable(self, pathfinder):
        """The treasure is only reachable through a locked door: go get the key."""
        status, result = _route(
            pathfinder, TREASURE_BEHIND_A_DOOR, (0, 0), door_policy="pass_with_key"
        )
        assert status == 200, result
        assert _replay(pathfinder, TREASURE_BEHIND_A_DOOR, (0, 0), result["path"]) == []
        assert (0, 2) in _trace(TREASURE_BEHIND_A_DOOR, (0, 0), result["path"])

    def test_an_unreachable_treasure_never_costs_five_lives(self, pathfinder):
        """Standing short of a keyless door beats dying on it.

        The old fallback was the shortest path, which walked straight into the
        door for -5 lives and scored nothing. Stopping short keeps every life.
        """
        grid = TREASURE_BEHIND_A_DOOR_WITH_NO_KEY
        for policy in ("avoid", "pass_with_key", "answer"):
            status, result = _route(pathfinder, grid, (0, 0), door_policy=policy)
            assert status == 200, result
            trace = _trace(grid, (0, 0), result["path"])
            assert (2, 2) not in trace, (policy, trace)  # the door
            assert (2, 3) not in trace, (policy, trace)  # the treasure behind it
            assert result["note"], policy
            # It still collects what is safely reachable rather than standing still.
            assert (0, 2) in trace, (policy, trace)

    def test_doors_are_untouched_under_the_default_policy(self, pathfinder):
        """door_policy defaults to avoid: a keyless door costs 5 lives."""
        grid = [
            ["start", "c30", "c7"],
            ["normal", "normal", "treasure"],
        ]
        status, result = _route(pathfinder, grid, (0, 0))
        assert status == 200, result
        assert _replay(pathfinder, grid, (0, 0), result["path"]) == []


class TestBudgets:
    def test_time_budget_is_respected(self, pathfinder):
        """A tight budget must drop challenges rather than overrun."""
        generous = _route(pathfinder, COINS_AND_CHALLENGES, (0, 0), time_budget=0)[1]
        tight = _route(pathfinder, COINS_AND_CHALLENGES, (0, 0), time_budget=6)[1]

        assert tight["estimated_time"] <= 6
        assert tight["challenges"] <= generous["challenges"]
        assert _replay(pathfinder, COINS_AND_CHALLENGES, (0, 0), tight["path"]) == []

    def test_life_budget_limits_expected_damage(self, pathfinder):
        """With two lives and one in reserve, expected damage must stay under one."""
        status, result = _route(pathfinder, COINS_AND_CHALLENGES, (0, 0), lives=2)
        assert status == 200, result
        assert result["expected_damage"] <= 1.0
        assert _replay(pathfinder, COINS_AND_CHALLENGES, (0, 0), result["path"]) == []

    def test_skip_tiles_excludes_a_challenge_type(self, pathfinder):
        with_c4 = _route(pathfinder, COINS_AND_CHALLENGES, (0, 0))[1]
        without_c4 = _route(pathfinder, COINS_AND_CHALLENGES, (0, 0), skip_tiles="c4 c5")[1]
        assert without_c4["challenges"] < with_c4["challenges"]


class TestTileEconomics:
    def test_a_life_is_worth_the_lives_bonus_multiplier(self, pathfinder):
        assert pathfinder.LIFE_VALUE == 250

    # Published in the AWS AI League in-game guide (Bonuses / Challenges tabs).
    # The planner routes on these numbers, so a silent drift here quietly sends
    # the champion to the wrong tiles. c17 and c18 are absent from the guide and
    # are deliberately not asserted.
    OFFICIAL_TILES = {
        "c1": (400, 1), "c2": (600, 1), "c3": (800, 1), "c4": (500, 1),
        "c5": (250, 1), "c6": (2000, 2), "c7": (250, 0), "c8": (0, 1),
        "c23": (1000, 1),
        "c30": (1000, 5), "c31": (1000, 5), "c32": (1000, 5), "c33": (1000, 5),
        "c40": (50, 0), "c41": (50, 0), "c42": (50, 0), "c43": (50, 0),
    }

    def test_points_and_damage_match_the_official_rules(self, pathfinder):
        for tile, (points, damage) in self.OFFICIAL_TILES.items():
            assert pathfinder.TILE_POINTS[tile] == points, tile
            assert pathfinder.TILE_DAMAGE[tile] == damage, tile

    def test_the_boss_outranks_the_treasure(self, pathfinder):
        """c6 is worth 2000 — more than reaching the treasure — so a planner
        that skips it to finish early is leaving the biggest prize behind."""
        assert pathfinder.TILE_POINTS["c6"] > 1000

    def test_challenge_value_uses_the_break_even_formula(self, pathfinder):
        """At p*, a challenge is worth exactly nothing: p* = 250D / (P + 250D)."""
        cfg = dict(pathfinder.DEFAULTS)
        points, damage = pathfinder.TILE_POINTS["c5"], pathfinder.TILE_DAMAGE["c5"]
        break_even = 250 * damage / (points + 250 * damage)
        cfg["p_correct"] = {"c5": break_even}
        cfg["door_policy"] = "avoid"
        assert pathfinder.cell_value("c5", cfg) == pytest.approx(0.0)

    def test_coins_carry_no_risk(self, pathfinder):
        cfg = dict(pathfinder.DEFAULTS)
        cfg["p_correct"] = dict(pathfinder.DEFAULT_P_CORRECT)
        assert pathfinder.cell_value("c7", cfg) == 250
        assert pathfinder.expected_damage("c7", cfg) == 0


class TestLegacyStrategies:
    """swift and get_coins must keep behaving exactly as before."""

    def test_swift_returns_the_shortest_path(self, pathfinder):
        grid = [["start", "normal", "treasure"]]
        response = pathfinder.lambda_handler(
            {"game_map": grid, "start_pos": [0, 0], "strategy": "swift"}, None
        )
        assert json.loads(response["body"])["path"] == ["right", "right"]

    def test_get_coins_collects_every_reachable_coin(self, pathfinder):
        grid = [["start", "c7", "c7", "treasure"]]
        response = pathfinder.lambda_handler(
            {"game_map": grid, "start_pos": [0, 0], "strategy": "get_coins"}, None
        )
        assert json.loads(response["body"])["path"] == ["right", "right", "right"]


class TestInputHandling:
    def test_start_on_a_wall_recovers_from_the_start_tile(self, pathfinder):
        grid = [["wall", "normal", "treasure"], ["start", "normal", "normal"]]
        status, result = _route(pathfinder, grid, (0, 0))
        assert status == 200, result
        assert result["start_position"] == [1, 0]
        assert _replay(pathfinder, grid, (1, 0), result["path"]) == []

    def test_start_on_a_wall_with_no_start_tile_is_rejected(self, pathfinder):
        grid = [["wall", "normal", "treasure"]]
        status, result = _route(pathfinder, grid, (0, 0))
        assert status == 400
        assert "walkable" in result["error"]

    def test_missing_treasure_is_rejected(self, pathfinder):
        status, result = _route(pathfinder, [["start", "normal"]], (0, 0))
        assert status == 400
        assert "treasure" in result["error"].lower()

    def test_grid_accepts_a_json_string(self, pathfinder):
        grid = [["start", "c7", "treasure"]]
        status, result = _route(pathfinder, json.dumps(grid), (0, 0))
        assert status == 200, result
        assert _replay(pathfinder, grid, (0, 0), result["path"]) == []

    def test_numeric_parameters_accept_strings(self, pathfinder):
        """The sub-agent passes tool arguments as strings."""
        status, result = _route(
            pathfinder, COINS_AND_CHALLENGES, (0, 0), time_budget="6", lives="2"
        )
        assert status == 200, result
        assert result["estimated_time"] <= 6
