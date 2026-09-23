"""Seeded, data-only procedural world generation for PyZombieGame.

This module deliberately has no graphics dependency. A renderer can later turn
the terrain and locations here into tiles, sprites, or an isometric map.
"""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
import json
import random
import secrets
from pathlib import Path


class Terrain(str, Enum):
    WATER = "water"
    PLAINS = "plains"
    FOREST = "forest"
    HILLS = "hills"
    MOUNTAINS = "mountains"
    SWAMP = "swamp"


class Feature(str, Enum):
    RIVER = "river"
    ROAD = "road"


@dataclass
class Tile:
    x: int
    y: int
    elevation: float
    moisture: float
    terrain: Terrain
    features: set[Feature] = field(default_factory=set)


@dataclass
class Settlement:
    name: str
    x: int
    y: int
    size: str
    population: int = 0


@dataclass
class Farm:
    name: str
    x: int
    y: int


@dataclass
class PointOfInterest:
    name: str
    kind: str
    x: int
    y: int


@dataclass
class World:
    seed: int
    width: int
    height: int
    tiles: list[list[Tile]]
    cell_size_km: int = 5
    settlements: list[Settlement] = field(default_factory=list)
    farms: list[Farm] = field(default_factory=list)
    points_of_interest: list[PointOfInterest] = field(default_factory=list)

    def tile_at(self, x: int, y: int) -> Tile:
        """Return a tile by world-grid coordinates."""
        if not (0 <= x < self.width and 0 <= y < self.height):
            raise IndexError(f"Tile ({x}, {y}) is outside this world")
        return self.tiles[y][x]


def _smooth_noise(rng: random.Random, width: int, height: int, scale: int) -> list[list[float]]:
    """Create deterministic, smoothly varying values using a coarse grid."""
    grid_w = width // scale + 2
    grid_h = height // scale + 2
    lattice = [[rng.random() for _ in range(grid_w)] for _ in range(grid_h)]
    values: list[list[float]] = []
    for y in range(height):
        gy, fy = divmod(y, scale)
        fy /= scale
        row = []
        for x in range(width):
            gx, fx = divmod(x, scale)
            fx /= scale
            top = lattice[gy][gx] * (1 - fx) + lattice[gy][gx + 1] * fx
            bottom = lattice[gy + 1][gx] * (1 - fx) + lattice[gy + 1][gx + 1] * fx
            row.append(top * (1 - fy) + bottom * fy)
        values.append(row)
    return values


def _build_tiles(seed: int, width: int, height: int) -> list[list[Tile]]:
    # Separate streams keep terrain stable if we later add other random systems.
    elevation = _smooth_noise(random.Random(seed), width, height, 9)
    moisture = _smooth_noise(random.Random(seed ^ 0x5EED), width, height, 7)
    tiles: list[list[Tile]] = []
    for y in range(height):
        row = []
        for x in range(width):
            h, wet = elevation[y][x], moisture[y][x]
            if h < 0.25:
                terrain = Terrain.WATER
            elif h > 0.84:
                terrain = Terrain.MOUNTAINS
            elif wet > 0.84 and h < 0.55:
                terrain = Terrain.SWAMP
            elif wet > 0.38:
                terrain = Terrain.FOREST
            elif h > 0.67:
                terrain = Terrain.HILLS
            else:
                terrain = Terrain.PLAINS
            row.append(Tile(x, y, h, wet, terrain))
        tiles.append(row)
    return tiles


def _carve_river(world: World, rng: random.Random) -> None:
    """Trace a river from a high interior tile toward the nearest map edge."""
    candidates = [
        tile for row in world.tiles for tile in row
        if tile.terrain in (Terrain.HILLS, Terrain.MOUNTAINS)
        and world.width * 0.2 < tile.x < world.width * 0.8
        and world.height * 0.2 < tile.y < world.height * 0.8
    ]
    if not candidates:
        return
    current = max(candidates, key=lambda tile: tile.elevation)
    visited: set[tuple[int, int]] = set()
    while (current.x, current.y) not in visited:
        visited.add((current.x, current.y))
        current.features.add(Feature.RIVER)
        if current.terrain == Terrain.WATER:
            break
        neighbors = [
            world.tile_at(nx, ny)
            for nx, ny in ((current.x - 1, current.y), (current.x + 1, current.y),
                           (current.x, current.y - 1), (current.x, current.y + 1))
            if 0 <= nx < world.width and 0 <= ny < world.height
            and (nx, ny) not in visited
        ]
        if not neighbors:
            break
        # Prefer lower land, with a tiny seeded tie-breaker for natural bends.
        current = min(neighbors, key=lambda tile: tile.elevation + rng.random() * 0.035)


def _place_settlements(world: World) -> None:
    """Place a city and smaller communities across the large region."""
    plans = (
        ("Blackpine", "city", 0.50, 0.50),
        ("Westhaven", "town", 0.20, 0.28),
        ("Eastbank", "town", 0.80, 0.28),
        ("Pine Junction", "town", 0.20, 0.72),
        ("Mill Creek", "town", 0.80, 0.72),
        ("Northwood", "village", 0.50, 0.12),
        ("South Fork", "village", 0.50, 0.88),
        ("Raven's Pass", "village", 0.12, 0.50),
        ("Clearwater", "village", 0.88, 0.50),
    )
    river_tiles = [tile for row in world.tiles for tile in row if Feature.RIVER in tile.features]
    population_rng = random.Random(world.seed ^ 0xC17A)

    for name, size, fraction_x, fraction_y in plans:
        target_x = round((world.width - 1) * fraction_x)
        target_y = round((world.height - 1) * fraction_y)
        candidates = [
            tile for row in world.tiles for tile in row
            if tile.terrain in (Terrain.PLAINS, Terrain.FOREST)
            and Feature.RIVER not in tile.features
        ]
        if not candidates:
            continue

        def placement_score(tile: Tile) -> float:
            target_distance = abs(tile.x - target_x) + abs(tile.y - target_y)
            if not river_tiles:
                return target_distance
            river_distance = min(abs(tile.x - river.x) + abs(tile.y - river.y) for river in river_tiles)
            river_bonus = max(0, river_distance - 12) * 0.25
            forest_penalty = 2 if tile.terrain == Terrain.FOREST else 0
            return target_distance + river_bonus + forest_penalty

        candidates.sort(key=placement_score)
        minimum_spacing = {"city": 20, "town": 16, "village": 12}[size]
        site = next((
            tile for tile in candidates
            if all(abs(tile.x - other.x) + abs(tile.y - other.y) >= minimum_spacing
                   for other in world.settlements)
        ), None)
        if site is None:
            continue

        populations = {"city": (45000, 120000), "town": (5000, 20000), "village": (300, 1500)}
        population = population_rng.randint(*populations[size])
        world.settlements.append(Settlement(name, site.x, site.y, size, population))


def _place_farms(world: World) -> None:
    """Place farms on plains within a reasonable distance of settlements."""
    rng = random.Random(world.seed ^ 0xFA12)
    for settlement in world.settlements:
        count, radius = {
            "city": (5, (8, 16)),
            "town": (3, (5, 11)),
            "village": (1, (3, 7)),
        }[settlement.size]
        placed = 0
        for _ in range(count * 12):
            if placed >= count:
                break
            dx = rng.randint(radius[0], radius[1]) * rng.choice((-1, 1))
            dy = rng.randint(radius[0], radius[1]) * rng.choice((-1, 1))
            x, y = settlement.x + dx, settlement.y + dy
            if not (0 <= x < world.width and 0 <= y < world.height):
                continue
            tile = world.tile_at(x, y)
            if tile.terrain != Terrain.PLAINS or Feature.RIVER in tile.features:
                continue
            if any(abs(x - farm.x) + abs(y - farm.y) < 5 for farm in world.farms):
                continue
            world.farms.append(Farm(f"{settlement.name} Farm {placed + 1}", x, y))
            placed += 1


def _connect_roads(world: World) -> None:
    """Connect settlements in order with simple low-cost grid paths."""
    towns = world.settlements
    for start, end in zip(towns, towns[1:]):
        x, y = start.x, start.y
        # Move along the axis with the larger remaining distance first.
        while (x, y) != (end.x, end.y):
            dx, dy = end.x - x, end.y - y
            if abs(dx) >= abs(dy) and dx:
                x += 1 if dx > 0 else -1
            elif dy:
                y += 1 if dy > 0 else -1
            tile = world.tile_at(x, y)
            if tile.terrain not in (Terrain.WATER, Terrain.MOUNTAINS):
                tile.features.add(Feature.ROAD)


def _place_landmarks(world: World, rng: random.Random) -> None:
    """Add isolated survival-horror locations using terrain and road rules."""
    roads = [tile for row in world.tiles for tile in row if Feature.ROAD in tile.features]
    all_tiles = [tile for row in world.tiles for tile in row]
    road_distance = [[world.width + world.height for _ in range(world.width)] for _ in range(world.height)]
    frontier = deque()
    for tile in roads:
        road_distance[tile.y][tile.x] = 0
        frontier.append((tile.x, tile.y))
    while frontier:
        x, y = frontier.popleft()
        next_distance = road_distance[y][x] + 1
        for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
            if (0 <= nx < world.width and 0 <= ny < world.height
                    and next_distance < road_distance[ny][nx]):
                road_distance[ny][nx] = next_distance
                frontier.append((nx, ny))
    occupied = {(site.x, site.y) for site in world.settlements}
    occupied.update((farm.x, farm.y) for farm in world.farms)

    def near_road(tile: Tile, distance: int = 4) -> bool:
        return road_distance[tile.y][tile.x] <= distance

    def far_from_settlements(tile: Tile, distance: int = 8) -> bool:
        return all(abs(tile.x - site.x) + abs(tile.y - site.y) >= distance for site in world.settlements)

    definitions = (
        ("abandoned_gas_station", "Abandoned Gas Station", 4,
         lambda t: Feature.ROAD in t.features and far_from_settlements(t, 10)),
        ("abandoned_motel", "Abandoned Motel", 2,
         lambda t: Feature.ROAD in t.features and far_from_settlements(t, 8)),
        ("roadside_diner", "Roadside Diner", 3,
         lambda t: Feature.ROAD in t.features and far_from_settlements(t, 8)),
        ("roadblock", "Abandoned Roadblock", 4,
         lambda t: Feature.ROAD in t.features and far_from_settlements(t, 12)),
        ("ranger_station", "Ranger Station", 3,
         lambda t: t.terrain == Terrain.FOREST and near_road(t, 5) and far_from_settlements(t, 10)),
        ("lumber_mill", "Abandoned Lumber Mill", 2,
         lambda t: t.terrain == Terrain.FOREST and near_road(t, 5) and far_from_settlements(t, 12)),
        ("research_outpost", "Research Outpost", 2,
         lambda t: t.terrain in (Terrain.FOREST, Terrain.HILLS) and far_from_settlements(t, 22)),
        ("hunting_cabin", "Abandoned Cabin", 8,
         lambda t: t.terrain == Terrain.FOREST and not near_road(t, 3) and far_from_settlements(t, 10)),
    )

    for kind, label, count, rule in definitions:
        candidates = [tile for tile in all_tiles if rule(tile) and (tile.x, tile.y) not in occupied]
        rng.shuffle(candidates)
        placed = 0
        for tile in candidates:
            if any(abs(tile.x - x) + abs(tile.y - y) < 7 for x, y in occupied):
                continue
            name = f"{label} {placed + 1}"
            world.points_of_interest.append(PointOfInterest(name, kind, tile.x, tile.y))
            occupied.add((tile.x, tile.y))
            placed += 1
            if placed >= count:
                break


def generate_world(seed: int, width: int = 120, height: int = 240, cell_size_km: int = 5) -> World:
    """Generate a repeatable world from a seed and map dimensions."""
    if width < 12 or height < 12:
        raise ValueError("World dimensions must be at least 12 by 12")
    world = World(seed, width, height, _build_tiles(seed, width, height), cell_size_km)
    _carve_river(world, random.Random(seed ^ 0xA11CE))
    _place_settlements(world)
    _place_farms(world)
    _connect_roads(world)
    _place_landmarks(world, random.Random(seed ^ 0xB01D))
    return world


def render_ascii(world: World, max_width: int = 80, max_height: int = 40) -> str:
    """Return a compact top-down preview for debugging before graphics exist."""
    symbols = {
        Terrain.WATER: "~", Terrain.PLAINS: ".", Terrain.FOREST: "T",
        Terrain.HILLS: "^", Terrain.MOUNTAINS: "M", Terrain.SWAMP: ",",
    }
    step_x = max(1, (world.width + max_width - 1) // max_width)
    step_y = max(1, (world.height + max_height - 1) // max_height)
    settlement_marks = {
        (site.x // step_x, site.y // step_y): "C" if site.size == "city" else "S"
        for site in world.settlements
    }
    farm_marks = {(farm.x // step_x, farm.y // step_y) for farm in world.farms}
    poi_symbols = {
        "abandoned_gas_station": "G", "abandoned_motel": "m",
        "roadside_diner": "d", "roadblock": "!", "ranger_station": "r",
        "lumber_mill": "l", "research_outpost": "?", "hunting_cabin": "c",
    }
    poi_marks = {
        (site.x // step_x, site.y // step_y): poi_symbols.get(site.kind, "?")
        for site in world.points_of_interest
    }
    rows = []
    for y in range(0, world.height, step_y):
        line = []
        for x in range(0, world.width, step_x):
            tile = world.tile_at(x, y)
            if (x // step_x, y // step_y) in settlement_marks:
                char = settlement_marks[(x // step_x, y // step_y)]
            elif (x // step_x, y // step_y) in farm_marks:
                char = "f"
            elif (x // step_x, y // step_y) in poi_marks:
                char = poi_marks[(x // step_x, y // step_y)]
            elif Feature.RIVER in tile.features:
                char = "="
            elif Feature.ROAD in tile.features:
                char = "+"
            else:
                char = symbols[tile.terrain]
            line.append(char)
        rows.append("".join(line))
    return "\n".join(rows)


def render_svg(world: World, tile_size: int = 6) -> str:
    """Render the world as a standalone, browser-viewable SVG map."""
    colors = {
        Terrain.WATER: "#416d91", Terrain.PLAINS: "#b6ad70",
        Terrain.FOREST: "#456b4a", Terrain.HILLS: "#8d815d",
        Terrain.MOUNTAINS: "#aaa69a", Terrain.SWAMP: "#68785b",
    }
    pad = 24
    map_width, map_height = world.width * tile_size, world.height * tile_size
    legend = (("#416d91", "Water"), ("#b6ad70", "Plains"), ("#456b4a", "Forest"),
              ("#8d815d", "Hills"), ("#aaa69a", "Mountains"), ("#68785b", "Swamp"),
              ("#67c7de", "River"), ("#e5d4a4", "Road"), ("#93a96c", "Farm"),
              ("#e4a83d", "City"), ("#f1eee6", "Town / village"), ("#e36b46", "POI"))
    legend_rows = (len(legend) + 4) // 5
    legend_height = legend_rows * 20 + 34
    svg_width = max(map_width + pad * 2, pad * 2 + 5 * 145)
    svg_height = map_height + pad * 2 + legend_height
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{svg_width}" height="{svg_height}" viewBox="0 0 {svg_width} {svg_height}">',
        '<rect width="100%" height="100%" fill="#171a20"/>',
        f'<text x="{pad}" y="22" fill="#f3eee3" font-family="sans-serif" font-size="16">PyZombieGame region · seed {world.seed} · {world.cell_size_km} km per cell</text>',
    ]
    left, top = pad, pad + 10
    for row in world.tiles:
        for tile in row:
            x, y = left + tile.x * tile_size, top + tile.y * tile_size
            parts.append(
                f'<rect x="{x}" y="{y}" width="{tile_size}" height="{tile_size}" '
                f'fill="{colors[tile.terrain]}" stroke="#171a20" stroke-opacity=".2" stroke-width=".5">'
                f'<title>{tile.terrain.value} at ({tile.x}, {tile.y}) · elevation {tile.elevation:.2f} · moisture {tile.moisture:.2f}</title></rect>'
            )
            if Feature.RIVER in tile.features:
                inset = tile_size * 0.32
                parts.append(f'<rect x="{x + inset:.1f}" y="{y + inset:.1f}" width="{tile_size - inset * 2:.1f}" height="{tile_size - inset * 2:.1f}" rx="3" fill="#67c7de"/>')
            if Feature.ROAD in tile.features:
                inset = tile_size * 0.4
                parts.append(f'<rect x="{x + inset:.1f}" y="{y + inset:.1f}" width="{tile_size - inset * 2:.1f}" height="{tile_size - inset * 2:.1f}" rx="2" fill="#e5d4a4"/>')
    for site in world.points_of_interest:
        cx, cy = left + (site.x + 0.5) * tile_size, top + (site.y + 0.5) * tile_size
        parts.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{tile_size * .34:.1f}" fill="#e36b46" stroke="#171a20"><title>{site.name} ({site.kind})</title></circle>')
    for farm in world.farms:
        x, y = left + farm.x * tile_size, top + farm.y * tile_size
        parts.append(f'<rect x="{x + tile_size * .18:.1f}" y="{y + tile_size * .18:.1f}" width="{tile_size * .64:.1f}" height="{tile_size * .64:.1f}" fill="#93a96c" stroke="#24301c"><title>{farm.name}</title></rect>')
    for settlement in world.settlements:
        cx, cy = left + (settlement.x + 0.5) * tile_size, top + (settlement.y + 0.5) * tile_size
        color = "#e4a83d" if settlement.size == "city" else "#f1eee6"
        radius = tile_size * (0.46 if settlement.size == "city" else 0.34)
        parts.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{radius:.1f}" fill="{color}" stroke="#28231d" stroke-width="1"><title>{settlement.name} · {settlement.size} · population {settlement.population:,}</title></circle>')
        if settlement.size == "city":
            parts.append(f'<text x="{cx + tile_size * .55:.1f}" y="{cy + 4:.1f}" fill="#fff" font-family="sans-serif" font-size="12" stroke="#171a20" stroke-width="3" paint-order="stroke">{settlement.name}</text>')
    legend_y = top + map_height + 26
    for index, (color, label) in enumerate(legend):
        row, column = divmod(index, 5)
        x, y = pad + column * 145, legend_y + row * 20
        parts.append(f'<rect x="{x}" y="{y - 11}" width="12" height="12" rx="2" fill="{color}"/>')
        parts.append(f'<text x="{x + 17}" y="{y}" fill="#eee" font-family="sans-serif" font-size="11">{label}</text>')
    parts.append('</svg>')
    return "\n".join(parts)


def save_game(save_dir: Path, slot_number: int, seed: int,
              character: dict[str, object] | None = None,
              save_name: str | None = None) -> Path:
    """Write a world seed into its own numbered save file."""
    save_dir.mkdir(parents=True, exist_ok=True)
    path = save_dir / f"slot_{slot_number}.json"
    previous = {}
    if path.exists():
        try:
            previous = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            previous = {}
    chosen_name = (save_name or str(previous.get("save_name", ""))).strip()
    if not chosen_name:
        chosen_name = f"Save Slot {slot_number}"
    data = {"save_version": 4, "slot_number": slot_number,
            "world_seed": seed, "save_name": chosen_name}
    if character is not None:
        data["character"] = character
    elif isinstance(previous.get("character"), dict):
        data["character"] = previous["character"]
    temporary_path = path.with_suffix(".tmp")
    temporary_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    temporary_path.replace(path)
    return path


def load_save_name(path: Path, slot_number: int) -> str:
    """Return a friendly label for a save, including older unnamed slots."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        name = str(data.get("save_name", "")).strip()
    except (OSError, json.JSONDecodeError):
        name = ""
    return name or f"Save Slot {slot_number}"


def load_game(path: Path) -> int:
    """Load a world's seed from a save file."""
    data = json.loads(path.read_text(encoding="utf-8"))
    return int(data["world_seed"])


def load_character(path: Path) -> dict[str, object]:
    """Load basic player details, providing defaults for older save files."""
    data = json.loads(path.read_text(encoding="utf-8"))
    character = data.get("character", {})
    stored_ammo = max(0, int(character.get("ammo_9mm", 18)))
    if "handgun_loaded" in character:
        handgun_loaded = max(0, min(6, int(character["handgun_loaded"])))
        ammo_9mm = stored_ammo
    else:
        # Older saves stored all handgun rounds in ammo_9mm without a magazine.
        handgun_loaded = min(6, stored_ammo)
        ammo_9mm = max(0, stored_ammo - handgun_loaded)
    return {
        "name": str(character.get("name", "Survivor")),
        "x": float(character.get("x", 31)),
        "y": float(character.get("y", 19)),
        "health": int(character.get("health", 100)),
        "inventory": character.get("inventory", {"Flashlight": 1, "First Aid Kit": 2}),
        "area_version": int(character.get("area_version", 0)),
        "inside_building": character.get("inside_building"),
        "outside_x": character.get("outside_x"),
        "outside_y": character.get("outside_y"),
        "flashlight_on": bool(character.get("flashlight_on", False)),
        "facing_x": int(character.get("facing_x", 0)),
        "facing_y": int(character.get("facing_y", -1)),
        "game_time_minutes": float(character.get("game_time_minutes", 8 * 60)),
        "region_x": int(character.get("region_x", 60)),
        "region_y": int(character.get("region_y", 120)),
        "equipped_weapon": str(character.get("equipped_weapon", "Kitchen Knife")),
        "ammo_9mm": ammo_9mm,
        "handgun_loaded": handgun_loaded,
        "defeated_zombies": character.get("defeated_zombies", []),
        "collected_items": character.get("collected_items", []),
    }


def list_save_slots(save_dir: Path, legacy_save_path: Path) -> list[tuple[int, Path, int]]:
    """Return valid numbered saves, migrating the old single save if needed."""
    save_dir.mkdir(parents=True, exist_ok=True)
    paths = list(save_dir.glob("slot_*.json"))
    if not paths and legacy_save_path.exists():
        # Preserve the user's existing single save as slot 1; leave the old
        # file untouched as a backup.
        save_game(save_dir, 1, load_game(legacy_save_path))
        paths = list(save_dir.glob("slot_*.json"))

    slots = []
    for path in paths:
        try:
            slot_number = int(path.stem.removeprefix("slot_"))
            slots.append((slot_number, path, load_game(path)))
        except (ValueError, KeyError, OSError, json.JSONDecodeError):
            continue
    return sorted(slots, key=lambda slot: slot[0])


def next_slot_number(save_dir: Path) -> int:
    """Choose the next unused numeric save slot."""
    used = []
    for path in save_dir.glob("slot_*.json"):
        try:
            used.append(int(path.stem.removeprefix("slot_")))
        except ValueError:
            continue
    return max(used, default=0) + 1


def show_world(seed: int, save_path: Path, pause: bool = False) -> None:
    """Rebuild the saved world and refresh its text and SVG previews."""
    world = generate_world(seed=seed)
    preview_path = Path(__file__).with_name("world_preview.svg")
    preview_path.write_text(render_svg(world), encoding="utf-8")
    map_width_km = world.width * world.cell_size_km
    map_height_km = world.height * world.cell_size_km
    print(f"\nWorld seed: {world.seed}")
    print(f"Province-scale map: {map_width_km} x {map_height_km} km, with {world.cell_size_km} km per cell")
    print("Legend: ~ water  . plains  T forest  ^ hills  M mountains  , swamp")
    print("         = river  + road  C city  S town/village  f farm")
    print("         G gas station  m motel  d diner  ! roadblock")
    print("         r ranger station  l lumber mill  ? research outpost  c cabin")
    print(render_ascii(world))
    print("Settlements:")
    for settlement in world.settlements:
        print(f"  {settlement.name}: {settlement.size}, population about {settlement.population:,}")
    print(f"Farms: {len(world.farms)}")
    print("Remote locations:")
    for location in world.points_of_interest:
        print(f"  {location.name} at ({location.x}, {location.y})")
    print(f"Visual map: {preview_path}")
    print(f"Current game saved in: {save_path}")
    if pause:
        input("\nPress Enter to return to the main menu...")


def main_menu(save_dir: Path, legacy_save_path: Path) -> None:
    """Show a terminal menu for choosing a save or starting another game."""
    while True:
        slots = list_save_slots(save_dir, legacy_save_path)
        print("\n" + "=" * 42)
        print("              PY ZOMBIE GAME")
        print("=" * 42)
        if slots:
            print("Saved games:")
            for slot_number, _, seed in slots:
                print(f"[{slot_number}] Save Slot {slot_number}  (world seed {seed})")
        else:
            print("No saved games yet.")
        print("[N] New game (creates another save slot)")
        print("[Q] Quit")
        choice = input("\nChoose an option: ").strip().lower()

        selected_slot = next((slot for slot in slots if str(slot[0]) == choice), None)
        if selected_slot is not None:
            slot_number, save_path, seed = selected_slot
            print(f"Loading Save Slot {slot_number}.")
            show_world(seed, save_path, pause=True)
        elif choice == "n":
            seed = secrets.randbits(32)
            slot_number = next_slot_number(save_dir)
            save_path = save_game(save_dir, slot_number, seed)
            print(f"New game started in Save Slot {slot_number}.")
            show_world(seed, save_path, pause=True)
        elif choice == "q":
            print("Goodbye!")
            return
        else:
            print("Please choose one of the listed options.")


if __name__ == "__main__":
    project_dir = Path(__file__).parent
    save_dir = project_dir / "saves"
    legacy_save_path = project_dir / "savegame.json"
    parser = argparse.ArgumentParser(description="Play PyZombieGame's procedural world prototype.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--new-game", action="store_true",
                      help="start a new random world and save it as the current game")
    mode.add_argument("--seed", type=int,
                      help="start or restore a world with this exact seed")
    args = parser.parse_args()

    if args.new_game or args.seed is not None:
        # Carry the previous single save into slot 1 before allocating a new one.
        list_save_slots(save_dir, legacy_save_path)

    if args.seed is not None:
        seed = args.seed
        slot_number = next_slot_number(save_dir)
        save_path = save_game(save_dir, slot_number, seed)
        show_world(seed, save_path)
    elif args.new_game:
        seed = secrets.randbits(32)
        slot_number = next_slot_number(save_dir)
        save_path = save_game(save_dir, slot_number, seed)
        show_world(seed, save_path)
    else:
        main_menu(save_dir, legacy_save_path)
