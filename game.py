"""Windowed Pygame launcher and simple procedural region viewer."""

from __future__ import annotations

import random
import secrets
import math
import json
import heapq
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pygame

import world as game_world


SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 800
FPS = 60

INK = (19, 24, 26)
PANEL = (29, 37, 38)
PANEL_LIGHT = (42, 52, 51)
PAPER = (227, 223, 207)
MUTED = (163, 174, 164)
ACCENT = (184, 112, 67)
ACCENT_HOVER = (211, 139, 85)

TERRAIN_COLORS = {
    game_world.Terrain.WATER: (45, 91, 112),
    game_world.Terrain.PLAINS: (135, 128, 84),
    game_world.Terrain.FOREST: (30, 65, 48),
    game_world.Terrain.HILLS: (94, 92, 69),
    game_world.Terrain.MOUNTAINS: (119, 119, 111),
    game_world.Terrain.SWAMP: (68, 83, 63),
}

PROJECT_DIR = Path(__file__).parent
SAVE_DIR = PROJECT_DIR / "saves"
LEGACY_SAVE_PATH = PROJECT_DIR / "savegame.json"
MENU_MUSIC_PATH = PROJECT_DIR / "assets" / "music" / "MainMenu.mp3"
FILE_SCREEN_MUSIC_PATH = PROJECT_DIR / "assets" / "music" / "SafePlace.mp3"
SETTINGS_PATH = PROJECT_DIR / "settings.json"
LOCAL_WIDTH = 192
LOCAL_HEIGHT = 128
LOCAL_TILE_SIZE = 15
LOCAL_TILE_METERS = 2
LOCAL_AREA_VERSION = 5
DEFAULT_INVENTORY = {
    "Flashlight": 1, "First Aid Kit": 2, "Handgun": 1, "Kitchen Knife": 1,
}
ITEM_DESCRIPTIONS = {
    "Flashlight": "A hand-held light. Enter toggles its beam.",
    "First Aid Kit": "Restores 35 health when used.",
    "Handgun": "A 9mm sidearm with a six-round magazine. Press R to reload from spare ammunition.",
    "Kitchen Knife": "A close-range weapon. Space swings toward the way you face.",
    "9mm Ammo": "Rounds for the handgun.",
}


@dataclass
class Character:
    name: str = "Survivor"
    x: float = 106
    y: float = 68
    health: int = 100
    inventory: dict[str, int] = field(default_factory=lambda: DEFAULT_INVENTORY.copy())
    area_version: int = LOCAL_AREA_VERSION
    inside_building: str | None = None
    outside_x: float | None = None
    outside_y: float | None = None
    flashlight_on: bool = False
    facing_x: int = 0
    facing_y: int = -1
    game_time_minutes: float = 8 * 60
    region_x: int = 60
    region_y: int = 120
    equipped_weapon: str = "Kitchen Knife"
    ammo_9mm: int = 12  # reserve ammunition
    handgun_loaded: int = 6
    defeated_zombies: list[str] = field(default_factory=list)
    collected_items: list[str] = field(default_factory=list)

    def to_save_data(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class LocalArea:
    name: str
    tiles: list[list[str]]
    labels: list[tuple[str, int, int]]
    buildings: list["Building"]
    spawn_x: int
    spawn_y: int
    biome: game_world.Terrain = game_world.Terrain.FOREST

    def can_walk(self, x: int, y: int) -> bool:
        return (0 <= x < LOCAL_WIDTH and 0 <= y < LOCAL_HEIGHT
                and self.tiles[y][x] not in {"tree", "building", "farmhouse", "shed", "pump"})


def can_occupy(area: LocalArea | Interior, x: float, y: float) -> bool:
    """Check the survivor's small footprint at a continuous map position."""
    radius = 0.22
    for offset_x in (-radius, radius):
        for offset_y in (-radius, radius):
            tile_x = math.floor(x + offset_x + 0.5)
            tile_y = math.floor(y + offset_y + 0.5)
            if not area.can_walk(tile_x, tile_y):
                return False
    return True


def reachable_walkable_tiles(area: LocalArea, origin_x: float,
                             origin_y: float) -> set[tuple[int, int]]:
    """Return walkable scene cells connected to the player's current position."""
    origin_cell = (math.floor(origin_x + 0.5), math.floor(origin_y + 0.5))
    origin = origin_cell
    if not (0 <= origin[0] < LOCAL_WIDTH and 0 <= origin[1] < LOCAL_HEIGHT
            and can_occupy(area, float(origin[0]), float(origin[1]))):
        origin = next(((x, y) for radius in range(1, 5)
                       for y in range(max(0, origin_cell[1] - radius),
                                      min(LOCAL_HEIGHT, origin_cell[1] + radius + 1))
                       for x in range(max(0, origin_cell[0] - radius),
                                      min(LOCAL_WIDTH, origin_cell[0] + radius + 1))
                       if max(abs(x - origin_cell[0]), abs(y - origin_cell[1])) == radius
                       and can_occupy(area, float(x), float(y))), None)
    if origin is None:
        return set()
    reachable = {origin}
    queue = deque([origin])
    while queue:
        x, y = queue.popleft()
        for neighbor in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            nx, ny = neighbor
            if (neighbor not in reachable and 0 <= nx < LOCAL_WIDTH and 0 <= ny < LOCAL_HEIGHT
                    and can_occupy(area, float(nx), float(ny))):
                reachable.add(neighbor)
                queue.append(neighbor)
    return reachable


def zombie_can_see_player(area: LocalArea, zombie_x: float, zombie_y: float,
                          player_x: float, player_y: float, max_distance: float = 20.0) -> bool:
    """Check range and sightline between the zombie and the player."""
    dx, dy = player_x - zombie_x, player_y - zombie_y
    distance = math.hypot(dx, dy)
    if distance > max_distance:
        return False
    steps = max(1, math.ceil(distance * 4))
    previous_cell = None
    for step in range(1, steps):
        progress = step / steps
        cell = (math.floor(zombie_x + dx * progress + 0.5),
                math.floor(zombie_y + dy * progress + 0.5))
        if cell != previous_cell:
            if not area.can_walk(*cell):
                return False
            previous_cell = cell
    return True


def find_zombie_path(area: LocalArea, start: tuple[int, int],
                     goal: tuple[int, int]) -> list[tuple[int, int]]:
    """Find a short four-way route through walkable cells using A*."""
    if not can_occupy(area, float(goal[0]), float(goal[1])):
        return []
    frontier = [(abs(goal[0] - start[0]) + abs(goal[1] - start[1]), 0, start)]
    came_from: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    costs = {start: 0}
    while frontier:
        _, cost, current = heapq.heappop(frontier)
        if cost != costs.get(current):
            continue
        if current == goal:
            route = []
            while current != start:
                route.append(current)
                current = came_from[current]
            route.reverse()
            return route
        x, y = current
        for neighbor in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            nx, ny = neighbor
            if (0 <= nx < LOCAL_WIDTH and 0 <= ny < LOCAL_HEIGHT
                    and can_occupy(area, float(nx), float(ny))):
                new_cost = cost + 1
                if new_cost < costs.get(neighbor, math.inf):
                    costs[neighbor] = new_cost
                    came_from[neighbor] = current
                    priority = new_cost + abs(goal[0] - nx) + abs(goal[1] - ny)
                    heapq.heappush(frontier, (priority, new_cost, neighbor))
    return []


@dataclass(frozen=True)
class Building:
    name: str
    kind: str
    door_x: int
    door_y: int
    door_side: str = "south"


@dataclass
class Zombie:
    zombie_id: str
    x: float
    y: float
    health: int = 100
    attack_timer: float = 0.0
    alerted: bool = False
    path_timer: float = 0.0
    path_target: tuple[int, int] | None = None
    path: list[tuple[int, int]] = field(default_factory=list)


@dataclass
class DamagePopup:
    x: float
    y: float
    text: str
    timer: float = 0.9


@dataclass
class Pickup:
    pickup_id: str
    item: str
    x: float
    y: float
    quantity: int = 1


@dataclass
class Interior:
    name: str
    kind: str
    tiles: list[list[str]]
    door_x: int
    door_y: int

    @property
    def width(self) -> int:
        return len(self.tiles[0])

    @property
    def height(self) -> int:
        return len(self.tiles)

    def can_walk(self, x: int, y: int) -> bool:
        return (0 <= x < self.width and 0 <= y < self.height
                and self.tiles[y][x] not in {"wall", "shelf", "counter", "bed",
                                             "table", "chair", "fridge", "dresser",
                                             "fireplace", "booth", "register"})


def draw_text(surface: pygame.Surface, font: pygame.font.Font, text: str,
              color: tuple[int, int, int], position: tuple[int, int]) -> None:
    surface.blit(font.render(text, True, color), position)


def fit_text(font: pygame.font.Font, text: str, max_width: int) -> str:
    """Trim a UI label to fit, preserving a visible ellipsis when needed."""
    if font.size(text)[0] <= max_width:
        return text
    while text and font.size(text + "…")[0] > max_width:
        text = text[:-1]
    return text + "…"


def draw_button(surface: pygame.Surface, rect: pygame.Rect, label: str,
                font: pygame.font.Font, mouse_position: tuple[int, int],
                enabled: bool = True, focused: bool = False) -> None:
    hovered = enabled and rect.collidepoint(mouse_position)
    if focused and enabled:
        pygame.draw.rect(surface, PAPER, rect.inflate(6, 6), width=2, border_radius=9)
    color = ACCENT_HOVER if hovered else ACCENT
    if not enabled:
        color = (67, 73, 68)
    pygame.draw.rect(surface, color, rect, border_radius=7)
    pygame.draw.rect(surface, (225, 193, 153), rect, width=1, border_radius=7)
    label_image = font.render(label, True, PAPER if enabled else MUTED)
    surface.blit(label_image, label_image.get_rect(center=rect.center))


def create_forest_backdrop() -> tuple[
    pygame.Surface,
    list[tuple[pygame.Surface, float]],
    list[tuple[pygame.Surface, int, float]],
    list[tuple[pygame.Surface, int, float]],
]:
    """Build reusable parallax tree, cloud, and fog layers for the menu."""
    surface = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
    for y in range(SCREEN_HEIGHT):
        shade = max(0, min(28, int(y / SCREEN_HEIGHT * 28)))
        pygame.draw.line(surface, (19 + shade // 3, 29 + shade // 2, 34 + shade),
                         (0, y), (SCREEN_WIDTH, y))

    tree_line = int(SCREEN_HEIGHT * 0.62)
    surface.fill((21, 29, 28), (0, tree_line + 55, SCREEN_WIDTH, SCREEN_HEIGHT))
    shade = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
    shade.fill((7, 11, 14, 78))
    surface.blit(shade, (0, 0))

    cloud_layers: list[tuple[pygame.Surface, int, float]] = []
    for layer, (cloud_y, speed, seed) in enumerate(((0, 4.0, 619), (42, 8.0, 853))):
        rng = random.Random(seed)
        clouds = pygame.Surface((SCREEN_WIDTH * 2, 350), pygame.SRCALPHA)
        for _ in range(34 if layer == 0 else 26):
            x = rng.randrange(-220, SCREEN_WIDTH + 220)
            y = rng.randrange(5, 240)
            width = rng.randrange(190, 520)
            height = rng.randrange(45, 145)
            shade_value = rng.randrange(24, 47) if layer == 0 else rng.randrange(32, 57)
            alpha = rng.randrange(35, 72) if layer == 0 else rng.randrange(28, 58)
            cloud_color = (shade_value, shade_value + 5, shade_value + 9, alpha)
            pygame.draw.ellipse(clouds, cloud_color, (x, y, width, height))
            pygame.draw.ellipse(clouds, cloud_color,
                                (x + width // 5, y - height // 2, width // 2, height))
        cloud_tile = clouds.subsurface((0, 0, SCREEN_WIDTH, 350)).copy()
        clouds.blit(cloud_tile, (SCREEN_WIDTH, 0))
        cloud_layers.append((clouds, cloud_y, speed))

    tree_layers: list[tuple[pygame.Surface, float]] = []
    for layer, count, speed in ((0, 38, 7.0), (1, 32, 17.0), (2, 26, 29.0)):
        rng = random.Random(81 + layer * 19)
        tree_surface = pygame.Surface((SCREEN_WIDTH * 2, SCREEN_HEIGHT), pygame.SRCALPHA)
        base_y = tree_line + layer * 58
        for _ in range(count):
            x = rng.randrange(-35, SCREEN_WIDTH + 35)
            height = rng.randrange(155, 310) if layer == 0 else rng.randrange(105, 240)
            half_width = int(height * rng.uniform(0.25, 0.42))
            green = rng.randrange(35, 57) - layer * 5
            tree_color = (green - 5, green + 8, green - 3, 255)
            pygame.draw.polygon(tree_surface, tree_color,
                                ((x, base_y - height), (x - half_width, base_y),
                                 (x + half_width, base_y)))
            pygame.draw.rect(tree_surface, (38, 38, 31, 255),
                             (x - 4, base_y - 3, 8, 26))
        tree_tile = tree_surface.subsurface((0, 0, SCREEN_WIDTH, SCREEN_HEIGHT)).copy()
        tree_surface.blit(tree_tile, (SCREEN_WIDTH, 0))
        tree_layers.append((tree_surface, speed))

    fog_layers: list[tuple[pygame.Surface, int, float]] = []
    for layer, (fog_y, speed, seed) in enumerate((
            (tree_line - 36, 11.0, 240), (tree_line + 48, 21.0, 481))):
        rng = random.Random(seed)
        fog = pygame.Surface((SCREEN_WIDTH * 2, 230), pygame.SRCALPHA)
        for _ in range(75):
            x = rng.randrange(-180, SCREEN_WIDTH + 180)
            y = rng.randrange(15, 210)
            width = rng.randrange(110, 340)
            height = rng.randrange(22, 74)
            alpha = rng.randrange(7, 20) if layer == 0 else rng.randrange(9, 24)
            pygame.draw.ellipse(fog, (175, 193, 192, alpha), (x, y, width, height))
        fog_tile = fog.subsurface((0, 0, SCREEN_WIDTH, 230)).copy()
        fog.blit(fog_tile, (SCREEN_WIDTH, 0))
        fog_layers.append((fog, fog_y, speed))
    return surface, tree_layers, cloud_layers, fog_layers


def draw_forest_backdrop(surface: pygame.Surface,
                         base: pygame.Surface,
                         tree_layers: list[tuple[pygame.Surface, float]],
                         cloud_layers: list[tuple[pygame.Surface, int, float]],
                         fog_layers: list[tuple[pygame.Surface, int, float]],
                         elapsed: float,
                         flash_alpha: int = 0,
                         lightning_points: list[tuple[int, int]] | None = None) -> None:
    surface.blit(base, (0, 0))
    for clouds, y, speed in cloud_layers:
        offset = int(elapsed * speed) % SCREEN_WIDTH
        surface.blit(clouds, (0, y), pygame.Rect(offset, 0, SCREEN_WIDTH, clouds.get_height()))
    for trees, speed in tree_layers:
        offset = int(elapsed * speed) % SCREEN_WIDTH
        surface.blit(trees, (0, 0), pygame.Rect(offset, 0, SCREEN_WIDTH, SCREEN_HEIGHT))
    for fog, y, speed in fog_layers:
        offset = int(elapsed * speed) % SCREEN_WIDTH
        surface.blit(fog, (0, y), pygame.Rect(offset, 0, SCREEN_WIDTH, fog.get_height()))
    if flash_alpha:
        flash = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
        flash.fill((201, 219, 233, flash_alpha))
        surface.blit(flash, (0, 0))
        if lightning_points and len(lightning_points) > 1:
            pygame.draw.lines(surface, (151, 185, 213), False, lightning_points, 8)
            pygame.draw.lines(surface, (237, 246, 255), False, lightning_points, 3)


def create_new_game(save_name: str | None = None) -> tuple[int, Path, int]:
    slot_number = game_world.next_slot_number(SAVE_DIR)
    seed = secrets.randbits(32)
    character = Character()
    save_path = game_world.save_game(SAVE_DIR, slot_number, seed,
                                     character.to_save_data(), save_name)
    return slot_number, save_path, seed


def generate_local_area(
    seed: int,
    biome: game_world.Terrain = game_world.Terrain.FOREST,
    landmark: tuple[str, str] | None = None,
    place_name: str | None = None,
    farm_site: bool = False,
) -> LocalArea:
    """Build a distinct local chunk from its regional terrain and landmark."""
    rng = random.Random(seed ^ 0x0A7A)
    tree_chance = {
        game_world.Terrain.FOREST: 0.52,
        game_world.Terrain.PLAINS: 0.13,
        game_world.Terrain.HILLS: 0.34,
        game_world.Terrain.MOUNTAINS: 0.25,
        game_world.Terrain.SWAMP: 0.41,
        game_world.Terrain.WATER: 0.08,
    }[biome]
    tiles = [["tree" if rng.random() < tree_chance else "grass"
              for _ in range(LOCAL_WIDTH)] for _ in range(LOCAL_HEIGHT)]

    def paint(x1: int, y1: int, x2: int, y2: int, tile: str) -> None:
        for y in range(max(0, y1), min(LOCAL_HEIGHT, y2)):
            for x in range(max(0, x1), min(LOCAL_WIDTH, x2)):
                tiles[y][x] = tile

    # Main roads stay aligned at chunk edges; side sites and spurs are seeded.
    paint(0, 62, LOCAL_WIDTH, 70, "road")
    paint(38, 0, 44, LOCAL_HEIGHT, "road")

    quadrants = [
        ((6, 27, 8, 32), [("Ranger Station", "cabin"), ("Abandoned Cabin", "cabin")]),
        ((50, 168, 8, 32), [("Abandoned Gas Station", "gas_station"),
                            ("Pine Rest Motel", "motel"), ("Research Outpost", "cabin")]),
        ((6, 27, 76, 96), [("Hunting Cabin", "cabin"), ("Abandoned Cabin", "cabin")]),
        ((50, 158, 76, 96), [("Roadside Diner", "diner"),
                             ("Abandoned Motel", "motel"), ("Lumber Mill", "cabin")]),
    ]
    if landmark:
        name, kind = landmark
        room_kind = {
            "abandoned_gas_station": "gas_station", "abandoned_motel": "motel",
            "roadside_diner": "diner", "ranger_station": "cabin",
            "lumber_mill": "cabin", "research_outpost": "cabin",
            "hunting_cabin": "cabin", "roadblock": "cabin",
        }.get(kind, "cabin")
        chosen = [(rng.randrange(4), (name, room_kind))]
    else:
        count = rng.randint(1, 2) if biome == game_world.Terrain.FOREST else rng.randint(2, 3)
        quadrant_indexes = rng.sample(range(4), count)
        chosen = [(index, rng.choice(quadrants[index][1])) for index in quadrant_indexes]

    buildings: list[Building] = []
    labels: list[tuple[str, int, int]] = []
    for quadrant_index, (name, kind) in chosen:
        (min_x, max_x, min_y, max_y), _ = quadrants[quadrant_index]
        width, height = {
            "gas_station": (26, 16), "motel": (30, 19),
            "diner": (27, 18), "cabin": (20, 14),
        }[kind]
        x = rng.randint(min_x, max_x - width)
        y = rng.randint(min_y, max_y - height)
        door_x = x + width // 2
        door_side = "south" if y < 62 else "north"
        door_y = y + height if door_side == "south" else y - 1
        paint(x - 5, y - 4, x + width + 5, y + height + 5, "lot")
        paint(x, y, x + width, y + height, "building")
        # Driveways join each lot to the nearest main road.
        if door_side == "south":
            paint(door_x - 1, door_y, door_x + 2, 63, "road")
        else:
            paint(door_x - 1, 70, door_x + 2, door_y, "road")
        building_name = name if not any(item.name == name for item in buildings) else f"{name} {len(buildings) + 1}"
        buildings.append(Building(building_name, kind, door_x, door_y, door_side))
        tiles[door_y][door_x] = "door"
        approach_y = door_y + 1 if door_side == "south" else door_y - 1
        for approach_x in range(door_x - 1, door_x + 2):
            if tiles[approach_y][approach_x] in {"tree", "pump", "building", "farmhouse", "shed"}:
                tiles[approach_y][approach_x] = "lot"
        labels.append((building_name.upper(), x, y - 2))
        if kind == "gas_station":
            for pump_x in (x + width + 2, x + width + 5):
                paint(pump_x, door_y - 2, pump_x + 2, door_y + 2, "pump")

    def add_small_structure(name: str, kind: str, width: int, height: int,
                            tile_kind: str, attempts: int = 80) -> bool:
        for _ in range(attempts):
            x = rng.randint(5, LOCAL_WIDTH - width - 6)
            side = rng.choice(("north", "south"))
            y = rng.randint(5, 45) if side == "south" else rng.randint(74, LOCAL_HEIGHT - height - 6)
            lot = (x - 2, y - 2, x + width + 2, y + height + 3)
            if any(tiles[py][px] not in {"tree", "grass"}
                   for py in range(lot[1], lot[3])
                   for px in range(lot[0], lot[2])):
                continue
            door_x = x + width // 2
            door_y = y + height if side == "south" else y - 1
            paint(*lot, "lot")
            paint(x, y, x + width, y + height, tile_kind)
            if side == "south":
                paint(door_x - 1, door_y, door_x + 2, 63, "road")
            else:
                paint(door_x - 1, 70, door_x + 2, door_y, "road")
            approach_y = door_y + 1 if side == "south" else door_y - 1
            for approach_x in range(door_x - 1, door_x + 2):
                if tiles[approach_y][approach_x] in {"tree", "pump", "building", "farmhouse", "shed"}:
                    tiles[approach_y][approach_x] = "lot"
            buildings.append(Building(name, kind, door_x, door_y, side))
            tiles[door_y][door_x] = "door"
            labels.append((name.upper(), x, y - 2))
            return True
        return False

    # Scattered outbuildings make the forest feel inhabited between major sites.
    if farm_site or (biome == game_world.Terrain.PLAINS and rng.random() < 0.3):
        add_small_structure("Farmhouse", "farmhouse", 19, 14, "farmhouse")
    elif biome in (game_world.Terrain.FOREST, game_world.Terrain.HILLS) and rng.random() < 0.12:
        add_small_structure("Remote Farmhouse", "farmhouse", 19, 14, "farmhouse")
    shed_count = rng.randint(2, 4) + (rng.randint(2, 3) if farm_site else 0)
    for shed_index in range(shed_count):
        add_small_structure(f"Storage Shed {shed_index + 1}", "shed", 11, 9, "shed")

    debris_count = rng.randint(45, 120)
    for _ in range(debris_count):
        x, y = rng.randrange(LOCAL_WIDTH), rng.randrange(LOCAL_HEIGHT)
        if tiles[y][x] == "grass":
            tiles[y][x] = "rubble" if rng.random() < 0.7 else "puddle"

    if place_name:
        area_name = place_name
    elif landmark:
        area_name = landmark[0]
    else:
        area_name = f"{biome.value.title()} Backroads"
    return LocalArea(area_name, tiles, labels, buildings, 106, 68, biome)


def local_region_seed(world_seed: int, region_x: int, region_y: int) -> int:
    """Give every regional grid cell a stable, distinct local scene seed."""
    return world_seed ^ (region_x * 73856093) ^ (region_y * 19349663)


def generate_region_area(world: game_world.World, region_x: int, region_y: int) -> LocalArea:
    """Create the terrain and point-of-interest scene for a regional map cell."""
    tile = world.tile_at(region_x, region_y)
    point = next((item for item in world.points_of_interest
                  if (item.x, item.y) == (region_x, region_y)), None)
    settlement = next((item for item in world.settlements
                       if (item.x, item.y) == (region_x, region_y)), None)
    farm = next((item for item in world.farms
                 if (item.x, item.y) == (region_x, region_y)), None)
    landmark = (point.name, point.kind) if point else None
    place_name = (point.name if point else settlement.name if settlement else
                  farm.name if farm else f"{tile.terrain.value.title()} Backroads · {region_x},{region_y}")
    return generate_local_area(local_region_seed(world.seed, region_x, region_y),
                               tile.terrain, landmark, place_name, farm is not None)


def generate_zombies(area: LocalArea, seed: int, region_x: int, region_y: int,
                     defeated: list[str],
                     anchor: tuple[float, float] | None = None) -> list[Zombie]:
    """Place infected around the survivor so they are visible in the current scene."""
    rng = random.Random(seed ^ 0x20B1E)
    zombies: list[Zombie] = []
    origin_x, origin_y = anchor or (float(area.spawn_x), float(area.spawn_y))
    target_count = rng.randint(4, 6)
    reachable = reachable_walkable_tiles(area, origin_x, origin_y)
    candidates = [
        (x, y) for y in range(LOCAL_HEIGHT) for x in range(LOCAL_WIDTH)
        if (area.tiles[y][x] in {"grass", "rubble", "puddle", "lot"}
            and 5 <= math.hypot(x - origin_x, y - origin_y) <= 18
            and (x, y) in reachable)
    ]
    rng.shuffle(candidates)
    for index in range(target_count):
        zombie_id = f"{region_x},{region_y}:{index}"
        if zombie_id in defeated:
            continue
        candidate_index = next((candidate_index for candidate_index, (x, y) in enumerate(candidates)
                                if all(math.hypot(x - other.x, y - other.y) >= 5
                                       for other in zombies)), None)
        if candidate_index is None:
            break
        x, y = candidates.pop(candidate_index)
        zombies.append(Zombie(zombie_id, float(x), float(y)))
    return zombies


def draw_zombie(surface: pygame.Surface, center: tuple[int, int], scale: int = LOCAL_TILE_SIZE,
                health: int = 100) -> None:
    """Draw a small original infected sprite from simple shapes."""
    x, y = center
    size = max(12, scale * 2)
    body_rect = pygame.Rect(x - size // 2, y - size // 2, size, size)
    pygame.draw.ellipse(surface, (33, 42, 31), body_rect)
    pygame.draw.ellipse(surface, (102, 133, 72),
                        (x - size // 3, y - size // 2, max(5, size * 2 // 3),
                         max(6, size * 2 // 3)))
    pygame.draw.ellipse(surface, (196, 158, 105), body_rect, max(1, size // 12))
    pygame.draw.line(surface, (95, 112, 69), (x - size // 2, y),
                     (x + size // 2, y + size // 4), max(2, size // 5))
    pygame.draw.circle(surface, (255, 70, 45), (x - size // 8, y - size // 4),
                       max(2, size // 9))
    pygame.draw.circle(surface, (255, 167, 63), (x + size // 7, y - size // 4),
                       max(2, size // 10))
    bar_width = max(24, size + 4)
    bar_x, bar_y = x - bar_width // 2, y - size // 2 - 8
    ratio = max(0.0, min(1.0, health / 100))
    bar_color = (105, 196, 102) if ratio > 0.5 else (222, 182, 73) if ratio > 0.25 else (215, 70, 58)
    pygame.draw.rect(surface, (18, 19, 18), (bar_x - 1, bar_y - 1, bar_width + 2, 6))
    pygame.draw.rect(surface, (62, 38, 36), (bar_x, bar_y, bar_width, 4))
    if ratio > 0:
        pygame.draw.rect(surface, bar_color,
                         (bar_x, bar_y, max(1, round(bar_width * ratio)), 4))


def draw_pickup(surface: pygame.Surface, center: tuple[int, int], item: str,
                scale: int = LOCAL_TILE_SIZE) -> None:
    """Draw a small illuminated scavenging marker with an item-specific glyph."""
    x, y = center
    size = max(14, min(26, scale))
    pygame.draw.circle(surface, (22, 24, 20), (x, y + 2), size // 2 + 3)
    pygame.draw.circle(surface, (220, 190, 101), (x, y), size // 2 + 3, 2)
    color = {
        "First Aid Kit": (191, 67, 56), "9mm Ammo": (213, 181, 81),
        "Handgun": (112, 127, 123), "Kitchen Knife": (186, 190, 169),
        "Flashlight": (201, 171, 78),
    }.get(item, (154, 164, 117))
    icon = pygame.Rect(x - size // 3, y - size // 3, size * 2 // 3, size * 2 // 3)
    pygame.draw.rect(surface, color, icon, border_radius=2)
    if item == "First Aid Kit":
        pygame.draw.rect(surface, (239, 229, 204), (x - 1, y - size // 4, 3, size // 2))
        pygame.draw.rect(surface, (239, 229, 204), (x - size // 4, y - 1, size // 2, 3))
    elif item == "9mm Ammo":
        pygame.draw.line(surface, (249, 226, 152), (x - 3, y - size // 4),
                         (x - 3, y + size // 4), 2)
        pygame.draw.line(surface, (249, 226, 152), (x + 3, y - size // 4),
                         (x + 3, y + size // 4), 2)
    elif item == "Kitchen Knife":
        pygame.draw.line(surface, (244, 240, 213), (x - size // 4, y + size // 4),
                         (x + size // 4, y - size // 4), 2)
    elif item == "Flashlight":
        pygame.draw.line(surface, (241, 231, 177), (x - size // 4, y + size // 4),
                         (x + size // 4, y - size // 4), max(2, size // 6))
    else:
        pygame.draw.line(surface, (229, 231, 218), (x - size // 4, y - 2),
                         (x + size // 4, y - 2), 2)
        pygame.draw.line(surface, (72, 77, 72), (x, y - 1), (x, y + size // 4), 2)


def make_local_surface(area: LocalArea) -> pygame.Surface:
    """Draw the small exploration area using simple colored shapes."""
    colors = {
        "grass": (57, 83, 57), "tree": (26, 55, 42),
        "road": (64, 69, 66), "lot": (93, 91, 72),
        "building": (79, 82, 79), "farmhouse": (117, 83, 55),
        "shed": (105, 66, 48), "pump": (156, 69, 49),
        "rubble": (91, 83, 67), "puddle": (54, 86, 96),
        "door": (115, 82, 49),
    }
    colors["grass"] = {
        game_world.Terrain.WATER: (67, 92, 72),
        game_world.Terrain.PLAINS: (126, 121, 77),
        game_world.Terrain.FOREST: (57, 83, 57),
        game_world.Terrain.HILLS: (100, 92, 68),
        game_world.Terrain.MOUNTAINS: (111, 109, 101),
        game_world.Terrain.SWAMP: (68, 82, 63),
    }[area.biome]
    image = pygame.Surface((LOCAL_WIDTH * LOCAL_TILE_SIZE, LOCAL_HEIGHT * LOCAL_TILE_SIZE))
    for y, row in enumerate(area.tiles):
        for x, tile in enumerate(row):
            left, top = x * LOCAL_TILE_SIZE, y * LOCAL_TILE_SIZE
            pygame.draw.rect(image, colors[tile],
                             (left, top, LOCAL_TILE_SIZE, LOCAL_TILE_SIZE))
            if tile == "tree":
                pygame.draw.circle(image, (34, 72, 48),
                                   (left + 7, top + 7), 6)
                pygame.draw.circle(image, (44, 87, 54),
                                   (left + 10, top + 9), 4)
            elif tile == "road" and y in (65, 66):
                if x % 12 in (3, 4, 5, 6, 7, 8):
                    pygame.draw.line(image, (177, 151, 104),
                                     (left + 2, top + 7), (left + 13, top + 7), 2)
            elif tile in {"building", "farmhouse", "shed"}:
                structure_color = {"building": (86, 89, 86),
                                   "farmhouse": (143, 103, 66),
                                   "shed": (126, 75, 51)}[tile]
                roof_color = {"building": (111, 110, 100),
                              "farmhouse": (165, 116, 68),
                              "shed": (149, 85, 55)}[tile]
                pygame.draw.rect(image, structure_color,
                                 (left + 1, top + 1, LOCAL_TILE_SIZE - 2, LOCAL_TILE_SIZE - 2))
                pygame.draw.line(image, roof_color,
                                 (left + 2, top + 3), (left + LOCAL_TILE_SIZE - 3, top + 3), 1)
            elif tile == "pump":
                pygame.draw.rect(image, (185, 81, 53),
                                 (left + 4, top + 3, 7, 10), border_radius=2)
            elif tile == "door":
                pygame.draw.rect(image, (190, 151, 88),
                                 (left + 2, top + 1, LOCAL_TILE_SIZE - 4, LOCAL_TILE_SIZE - 1))
                pygame.draw.circle(image, (42, 37, 30),
                                   (left + LOCAL_TILE_SIZE - 5, top + 8), 1)
    return image


def generate_interior(seed: int, building: Building) -> Interior:
    """Generate a deterministic floor plan and furnishings for one building."""
    rng = random.Random(seed + sum((index + 1) * ord(char)
                                   for index, char in enumerate(building.name)))
    sizes = {
        "gas_station": (30, 20), "motel": (38, 26),
        "diner": (32, 23), "cabin": (22, 18),
        "farmhouse": (28, 20), "shed": (16, 12),
    }
    width, height = sizes[building.kind]
    tiles = [["floor" for _ in range(width)] for _ in range(height)]
    for x in range(width):
        tiles[0][x] = tiles[height - 1][x] = "wall"
    for y in range(height):
        tiles[y][0] = tiles[y][width - 1] = "wall"

    def place(x: int, y: int, tile: str) -> None:
        if 1 <= x < width - 1 and 1 <= y < height - 1 and tiles[y][x] == "floor":
            tiles[y][x] = tile

    if building.kind == "gas_station":
        for x in range(7, width - 7, 5):
            for y in range(7, 14):
                place(x, y, "shelf")
        for x in range(5, width - 5):
            place(x, 4, "counter")
        place(width - 5, 5, "register")
        place(3, 3, "fridge")
        place(4, 3, "fridge")
    elif building.kind == "diner":
        for x in range(4, width - 4):
            place(x, 3, "counter")
        place(width - 6, 4, "register")
        for y in (8, 14):
            for x in (6, 14, 22):
                place(x, y, "table")
                place(x - 1, y, "chair")
                place(x + 1, y, "chair")
                place(x, y - 1, "chair")
                place(x, y + 1, "chair")
        for x in range(4, width - 4, 6):
            place(x, 19, "booth")
    elif building.kind == "motel":
        hall_x = width // 2
        for y in range(2, height - 2):
            if y not in (6, 12, 18):
                place(hall_x - 1, y, "wall")
                place(hall_x + 1, y, "wall")
        for y in (4, 9, 15, 21):
            for x in range(2, width - 2):
                if abs(x - hall_x) > 2:
                    place(x, y, "wall")
            for x in (4, hall_x + 4):
                place(x, y - 2, "bed")
                place(x + 2, y - 2, "dresser")
    elif building.kind == "farmhouse":
        place(4, 4, "bed")
        place(6, 4, "dresser")
        place(width - 5, 4, "fridge")
        place(width // 2, 8, "table")
        place(width // 2 - 1, 8, "chair")
        place(width // 2 + 1, 8, "chair")
        place(width - 5, height - 5, "fireplace")
    elif building.kind == "shed":
        for y in range(3, height - 3, 3):
            place(3, y, "shelf")
            place(width - 4, y, "shelf")
        place(width // 2, 4, "table")
        place(width // 2 - 1, 4, "shelf")
    else:  # cabin
        split_x = width // 2
        for y in range(3, height - 3):
            if y not in (8, 9):
                place(split_x, y, "wall")
        place(4, 4, "bed")
        place(6, 4, "dresser")
        place(width - 5, 4, "fireplace")
        place(width - 7, 10, "table")
        place(width - 8, 10, "chair")
        place(width - 6, 10, "chair")

    # Small deterministic clutter makes each room feel lived in or abandoned.
    clutter = {"gas_station": "shelf", "motel": "dresser",
               "diner": "chair", "cabin": "chair",
               "farmhouse": "chair", "shed": "shelf"}[building.kind]
    for _ in range(max(2, width // 8)):
        place(rng.randrange(2, width - 2), rng.randrange(2, height - 3), clutter)

    door_x, door_y = width // 2, height - 1
    tiles[door_y][door_x] = "door"
    return Interior(building.name, building.kind, tiles, door_x, door_y)


def make_interior_surface(interior: Interior, tile_size: int) -> pygame.Surface:
    """Render the generated floor plan with recognizable furniture silhouettes."""
    colors = {
        "floor": (74, 69, 58), "wall": (45, 46, 43), "door": (166, 119, 67),
        "shelf": (88, 66, 45), "counter": (105, 73, 48), "register": (62, 80, 76),
        "fridge": (126, 133, 126), "bed": (74, 89, 93), "dresser": (90, 65, 48),
        "table": (111, 75, 48), "chair": (101, 69, 47),
        "fireplace": (65, 51, 43), "booth": (111, 71, 59),
    }
    image = pygame.Surface((interior.width * tile_size, interior.height * tile_size))
    for y, row in enumerate(interior.tiles):
        for x, tile in enumerate(row):
            rect = pygame.Rect(x * tile_size, y * tile_size, tile_size, tile_size)
            color = colors[tile]
            if tile == "floor" and (x + y) % 2:
                color = (78, 72, 60)
            pygame.draw.rect(image, color, rect)
            if tile == "wall":
                pygame.draw.line(image, (65, 65, 59), rect.topleft,
                                 (rect.right - 1, rect.top), max(1, tile_size // 14))
            elif tile == "door":
                pygame.draw.rect(image, (196, 145, 79), rect.inflate(-tile_size // 5, -tile_size // 8))
            elif tile in {"table", "counter", "booth", "dresser", "shelf"}:
                pygame.draw.rect(image, tuple(min(255, c + 18) for c in color),
                                 rect.inflate(-tile_size // 5, -tile_size // 5),
                                 border_radius=max(1, tile_size // 12))
            elif tile == "bed":
                pygame.draw.rect(image, (114, 125, 119), rect.inflate(-tile_size // 6, -tile_size // 6))
                pygame.draw.rect(image, (173, 170, 147),
                                 (rect.x + tile_size // 6, rect.y + tile_size // 6,
                                  tile_size // 3, tile_size // 3))
            elif tile == "fireplace":
                pygame.draw.rect(image, (28, 29, 27), rect.inflate(-tile_size // 5, -tile_size // 5))
                pygame.draw.circle(image, (186, 89, 44), rect.center, max(2, tile_size // 5))
            elif tile == "register":
                pygame.draw.rect(image, (174, 161, 128), rect.inflate(-tile_size // 3, -tile_size // 3))
    return image


def _make_pickup(rng: random.Random, pickup_id: str, x: int, y: int,
                 weights: tuple[tuple[str, int], ...]) -> Pickup:
    item = rng.choices([entry[0] for entry in weights],
                       weights=[entry[1] for entry in weights], k=1)[0]
    quantity = rng.randint(5, 10) if item == "9mm Ammo" else 1
    return Pickup(pickup_id, item, float(x), float(y), quantity)


def generate_outdoor_pickups(area: LocalArea, seed: int, region_x: int, region_y: int,
                             collected: list[str]) -> list[Pickup]:
    """Seed roadside and wilderness supplies, with building-specific stashes."""
    rng = random.Random(seed ^ 0x1007)
    pickups: list[Pickup] = []
    weights_by_kind = {
        "gas_station": (("9mm Ammo", 6), ("First Aid Kit", 3), ("Handgun", 1),
                        ("Flashlight", 1)),
        "motel": (("First Aid Kit", 4), ("9mm Ammo", 3), ("Flashlight", 2),
                  ("Kitchen Knife", 1)),
        "diner": (("First Aid Kit", 3), ("Kitchen Knife", 3), ("9mm Ammo", 2),
                  ("Flashlight", 1)),
        "cabin": (("Kitchen Knife", 4), ("First Aid Kit", 2), ("9mm Ammo", 1),
                  ("Flashlight", 1)),
        "farmhouse": (("Kitchen Knife", 3), ("First Aid Kit", 2), ("9mm Ammo", 2),
                      ("Flashlight", 1)),
        "shed": (("Kitchen Knife", 3), ("9mm Ammo", 3), ("First Aid Kit", 1)),
    }
    walkable = {"grass", "rubble", "puddle", "lot", "road"}

    def free_spot(x: int, y: int) -> bool:
        return (0 <= x < LOCAL_WIDTH and 0 <= y < LOCAL_HEIGHT
                and area.tiles[y][x] in walkable and can_occupy(area, x, y)
                and all(math.hypot(x - pickup.x, y - pickup.y) >= 4 for pickup in pickups))

    for index, building in enumerate(area.buildings):
        door_out_y = building.door_y + (1 if building.door_side == "south" else -1)
        door_out_x = building.door_x
        candidates = [(x, y) for y in range(max(0, door_out_y - 7), min(LOCAL_HEIGHT, door_out_y + 8))
                      for x in range(max(0, door_out_x - 7), min(LOCAL_WIDTH, door_out_x + 8))
                      if 2 <= math.hypot(x - door_out_x, y - door_out_y) <= 7
                      and free_spot(x, y)]
        if candidates:
            x, y = rng.choice(candidates)
            weights = weights_by_kind.get(building.kind, weights_by_kind["cabin"])
            pickup = _make_pickup(rng, f"outside:{region_x},{region_y}:building:{index}",
                                  x, y, weights)
            pickups.append(pickup)

    # Make a small, findable cache near the region's standard arrival point.
    near_spawn = [(x, y) for y in range(max(0, area.spawn_y - 10), min(LOCAL_HEIGHT, area.spawn_y + 11))
                  for x in range(max(0, area.spawn_x - 10), min(LOCAL_WIDTH, area.spawn_x + 11))
                  if 5 <= math.hypot(x - area.spawn_x, y - area.spawn_y) <= 10 and free_spot(x, y)]
    rng.shuffle(near_spawn)
    wild_weights = (("9mm Ammo", 4), ("First Aid Kit", 2), ("Kitchen Knife", 2),
                    ("Flashlight", 1))
    for index, (x, y) in enumerate(near_spawn[:2]):
        pickup = _make_pickup(rng, f"outside:{region_x},{region_y}:cache:{index}",
                              x, y, wild_weights)
        pickups.append(pickup)

    remote = [(x, y) for y in range(LOCAL_HEIGHT) for x in range(LOCAL_WIDTH)
              if area.tiles[y][x] in walkable and can_occupy(area, x, y) and free_spot(x, y)]
    rng.shuffle(remote)
    for x, y in remote:
        if sum(pickup.pickup_id.startswith(f"outside:{region_x},{region_y}:wild:")
               for pickup in pickups) >= 4:
            break
        if any(math.hypot(x - pickup.x, y - pickup.y) < 12 for pickup in pickups):
            continue
        wild_index = sum(pickup.pickup_id.startswith(f"outside:{region_x},{region_y}:wild:")
                         for pickup in pickups)
        pickup = _make_pickup(rng, f"outside:{region_x},{region_y}:wild:{wild_index}",
                              x, y, wild_weights)
        pickups.append(pickup)
    return [pickup for pickup in pickups if pickup.pickup_id not in collected]


def generate_interior_pickups(interior: Interior, seed: int, region_x: int, region_y: int,
                              collected: list[str]) -> list[Pickup]:
    """Place deterministic loot on free floor tiles, weighted for the room type."""
    rng = random.Random(seed ^ sum((i + 1) * ord(c) for i, c in enumerate(interior.name)) ^ 0x1007)
    weights_by_kind = {
        "gas_station": (("9mm Ammo", 6), ("First Aid Kit", 3), ("Handgun", 1),
                        ("Flashlight", 1)),
        "motel": (("First Aid Kit", 4), ("9mm Ammo", 3), ("Flashlight", 2),
                  ("Kitchen Knife", 1)),
        "diner": (("First Aid Kit", 3), ("Kitchen Knife", 3), ("9mm Ammo", 2)),
        "cabin": (("Kitchen Knife", 4), ("First Aid Kit", 2), ("9mm Ammo", 1)),
        "farmhouse": (("First Aid Kit", 3), ("Kitchen Knife", 3), ("9mm Ammo", 2)),
        "shed": (("Kitchen Knife", 4), ("9mm Ammo", 3), ("First Aid Kit", 1)),
    }
    candidates = [(x, y) for y, row in enumerate(interior.tiles) for x, tile in enumerate(row)
                  if tile == "floor" and math.hypot(x - interior.door_x, y - interior.door_y) > 2]
    rng.shuffle(candidates)
    pickups = []
    for index, (x, y) in enumerate(candidates[:rng.randint(2, 4)]):
        pickup_id = f"inside:{region_x},{region_y}:{interior.name}:{index}"
        weights = weights_by_kind.get(interior.kind, weights_by_kind["cabin"])
        pickups.append(_make_pickup(rng, pickup_id, x, y, weights))
    return [pickup for pickup in pickups if pickup.pickup_id not in collected]


def character_from_save(save_path: Path, area: LocalArea) -> Character:
    data = game_world.load_character(save_path)
    if int(data.get("area_version", 0)) != LOCAL_AREA_VERSION:
        data["x"], data["y"] = area.spawn_x, area.spawn_y
        data["inside_building"] = None
    inside_building = data.get("inside_building")
    if inside_building not in {building.name for building in area.buildings}:
        if inside_building is not None:
            data["x"], data["y"] = area.spawn_x, area.spawn_y
        inside_building = None
    raw_inventory = data.get("inventory", DEFAULT_INVENTORY)
    inventory: dict[str, int] = {}
    if isinstance(raw_inventory, dict):
        for item, quantity in raw_inventory.items():
            canonical = "First Aid Kit" if str(item).casefold() == "first aid kit" else str(item)
            if int(quantity) > 0:
                inventory[canonical] = inventory.get(canonical, 0) + int(quantity)
    elif isinstance(raw_inventory, list):
        for item in raw_inventory:
            canonical = "First Aid Kit" if str(item).casefold() == "first aid kit" else str(item)
            inventory[canonical] = inventory.get(canonical, 0) + 1
    inventory.setdefault("Kitchen Knife", 1)
    inventory.setdefault("Handgun", 1)
    defeated_raw = data.get("defeated_zombies", [])
    defeated_zombies = [str(item) for item in defeated_raw] if isinstance(defeated_raw, list) else []
    collected_raw = data.get("collected_items", [])
    collected_items = [str(item) for item in collected_raw] if isinstance(collected_raw, list) else []
    character = Character(
        name=str(data["name"]),
        x=max(0.0, min(LOCAL_WIDTH - 1.0, float(data["x"]))),
        y=max(0.0, min(LOCAL_HEIGHT - 1.0, float(data["y"]))),
        health=max(0, min(100, int(data["health"]))),
        inventory=inventory,
        area_version=LOCAL_AREA_VERSION,
        inside_building=inside_building,
        outside_x=(float(data["outside_x"]) if data.get("outside_x") is not None else area.spawn_x),
        outside_y=(float(data["outside_y"]) if data.get("outside_y") is not None else area.spawn_y),
        flashlight_on=bool(data.get("flashlight_on", False)),
        facing_x=int(data.get("facing_x", 0)),
        facing_y=int(data.get("facing_y", -1)),
        game_time_minutes=max(0.0, float(data.get("game_time_minutes", 8 * 60))),
        region_x=max(0, int(data.get("region_x", 60))),
        region_y=max(0, int(data.get("region_y", 120))),
        equipped_weapon=(str(data.get("equipped_weapon", "Kitchen Knife"))
                         if data.get("equipped_weapon") in {"Kitchen Knife", "Handgun"}
                         else "Kitchen Knife"),
        ammo_9mm=max(0, int(data.get("ammo_9mm", 12))),
        handgun_loaded=max(0, min(6, int(data.get("handgun_loaded", 6)))),
        defeated_zombies=defeated_zombies,
        collected_items=collected_items,
    )
    if character.inside_building is None and not can_occupy(area, character.x, character.y):
        character.x, character.y = area.spawn_x, area.spawn_y
    return character


def save_character(save: tuple[int, Path, int], character: Character) -> None:
    slot_number, _, seed = save
    game_world.save_game(SAVE_DIR, slot_number, seed, character.to_save_data())


def draw_character(surface: pygame.Surface, center: tuple[int, int],
                   flashlight_on: bool = False,
                   facing: tuple[int, int] = (0, -1), scale: int = LOCAL_TILE_SIZE,
                   weapon: str = "Kitchen Knife") -> None:
    """Draw a small original survivor sprite from basic shapes."""
    x, y = center
    if flashlight_on:
        direction_x, direction_y = facing
        direction_length = math.hypot(direction_x, direction_y) or 1
        direction_x, direction_y = direction_x / direction_length, direction_y / direction_length
        perpendicular = (-direction_y, direction_x)
        beam_start = (x + direction_x * scale, y + direction_y * scale)
        beam_tip = (x + direction_x * scale * 8, y + direction_y * scale * 8)
        beam_half_width = scale * 3
        beam = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
        pygame.draw.polygon(beam, (246, 224, 156, 35), (
            (round(beam_start[0] + perpendicular[0] * beam_half_width),
             round(beam_start[1] + perpendicular[1] * beam_half_width)),
            (round(beam_start[0] - perpendicular[0] * beam_half_width),
             round(beam_start[1] - perpendicular[1] * beam_half_width)),
            (round(beam_tip[0] - perpendicular[0] * beam_half_width * 1.6),
             round(beam_tip[1] - perpendicular[1] * beam_half_width * 1.6)),
            (round(beam_tip[0] + perpendicular[0] * beam_half_width * 1.6),
             round(beam_tip[1] + perpendicular[1] * beam_half_width * 1.6)),
        ))
        surface.blit(beam, (0, 0))
    pygame.draw.ellipse(surface, (19, 24, 23), (x - 7, y + 5, 14, 6))
    pygame.draw.rect(surface, (93, 119, 96), (x - 5, y - 3, 10, 12), border_radius=3)
    pygame.draw.circle(surface, (197, 169, 139), (x, y - 7), 4)
    face_direction = pygame.Vector2(facing)
    if face_direction.length_squared() == 0:
        face_direction.update(0, -1)
    face_direction = face_direction.normalize()
    face_center = pygame.Vector2(x, y - 7) + face_direction * 2
    eye_side = pygame.Vector2(-face_direction.y, face_direction.x) * 1.5
    pygame.draw.circle(surface, (48, 43, 39), face_center + eye_side, 1)
    pygame.draw.circle(surface, (48, 43, 39), face_center - eye_side, 1)
    pygame.draw.line(surface, (36, 42, 39), (x - 2, y + 8), (x - 4, y + 12), 2)
    pygame.draw.line(surface, (36, 42, 39), (x + 2, y + 8), (x + 4, y + 12), 2)
    direction = pygame.Vector2(facing)
    if direction.length_squared() == 0:
        direction.update(0, -1)
    direction = direction.normalize()
    hand = pygame.Vector2(x, y) + direction * (scale * 0.22)
    tip = hand + direction * (scale * (0.82 if weapon == "Handgun" else 0.95))
    if weapon == "Handgun":
        pygame.draw.line(surface, (39, 44, 43), hand, tip, max(2, scale // 5))
        pygame.draw.line(surface, (92, 93, 84), hand + direction * 2,
                         tip - direction * 2, max(1, scale // 9))
        pygame.draw.line(surface, (49, 50, 47), hand,
                         hand - pygame.Vector2(-direction.y, direction.x) * (scale * 0.33),
                         max(2, scale // 6))
    else:
        pygame.draw.line(surface, (91, 99, 91), hand, tip, max(2, scale // 5))
        pygame.draw.line(surface, (174, 178, 156), hand + direction * (scale * 0.25),
                         tip, max(1, scale // 12))


def draw_inventory_icon(surface: pygame.Surface, rect: pygame.Rect, item: str) -> None:
    """Draw compact original pixel-style item art for the inventory slots."""
    pygame.draw.rect(surface, (9, 22, 88), rect)
    pygame.draw.rect(surface, (81, 103, 179), rect, 1)
    if item == "First Aid Kit":
        case = pygame.Rect(rect.centerx - 16, rect.centery - 14, 32, 28)
        pygame.draw.rect(surface, (205, 207, 190), case)
        pygame.draw.rect(surface, (112, 118, 113), case, 2)
        pygame.draw.rect(surface, (157, 49, 43), (rect.centerx - 4, rect.centery - 9, 8, 18))
        pygame.draw.rect(surface, (157, 49, 43),
                         (rect.centerx - 9, rect.centery - 4, 18, 8))
        pygame.draw.rect(surface, (225, 222, 194), (rect.centerx - 7, rect.centery - 19, 14, 5))
    elif item == "Flashlight":
        pygame.draw.polygon(surface, (220, 197, 115), (
            (rect.x + 9, rect.centery - 12), (rect.x + 28, rect.centery - 6),
            (rect.x + 28, rect.centery + 6), (rect.x + 9, rect.centery + 12)))
        pygame.draw.rect(surface, (48, 55, 58), (rect.x + 25, rect.centery - 5, 21, 10))
        pygame.draw.rect(surface, (108, 115, 108), (rect.x + 40, rect.centery - 8, 8, 16))
        pygame.draw.line(surface, (193, 190, 162), (rect.x + 29, rect.centery),
                         (rect.x + 43, rect.centery), 2)
    elif item == "Handgun":
        pygame.draw.rect(surface, (104, 112, 110), (rect.x + 8, rect.centery - 10, 38, 9))
        pygame.draw.rect(surface, (151, 155, 144), (rect.x + 35, rect.centery - 7, 13, 4))
        pygame.draw.polygon(surface, (67, 58, 48), (
            (rect.x + 18, rect.centery - 2), (rect.x + 31, rect.centery - 2),
            (rect.x + 27, rect.centery + 14), (rect.x + 17, rect.centery + 12)))
    elif item == "Kitchen Knife":
        pygame.draw.polygon(surface, (194, 198, 183), (
            (rect.x + 8, rect.centery), (rect.x + 37, rect.centery - 8),
            (rect.x + 44, rect.centery), (rect.x + 37, rect.centery + 6)))
        pygame.draw.rect(surface, (100, 66, 43), (rect.x + 5, rect.centery - 3, 12, 7))
    elif item == "9mm Ammo":
        for index in range(3):
            pygame.draw.rect(surface, (184, 157, 83),
                             (rect.x + 12 + index * 12, rect.centery - 11, 8, 24))
            pygame.draw.rect(surface, (209, 198, 156),
                             (rect.x + 12 + index * 12, rect.centery - 11, 8, 5))
    else:
        pygame.draw.rect(surface, (151, 137, 95), rect.inflate(-16, -16))


def inventory_items_for(character: Character) -> list[str]:
    items = sorted(item for item, count in character.inventory.items() if count > 0)
    if character.ammo_9mm > 0:
        items.append("9mm Ammo")
    return items


def make_map_surface(world: game_world.World) -> pygame.Surface:
    """Create a native-resolution terrain image; the window scales it later."""
    image = pygame.Surface((world.width, world.height))
    for row in world.tiles:
        for tile in row:
            color = TERRAIN_COLORS[tile.terrain]
            if game_world.Feature.RIVER in tile.features:
                color = (74, 145, 161)
            elif game_world.Feature.ROAD in tile.features:
                color = (176, 153, 112)
            image.set_at((tile.x, tile.y), color)
    return image


def open_world(save: tuple[int, Path, int]) -> tuple[game_world.World, pygame.Surface]:
    _, _, seed = save
    world = game_world.generate_world(seed)
    return world, make_map_surface(world)


def run_game() -> None:
    global SCREEN_WIDTH, SCREEN_HEIGHT
    pygame.init()
    pygame.joystick.init()
    pygame.display.set_caption("PyZombieGame — Blackpine Region")
    display_info = pygame.display.Info()
    SCREEN_WIDTH, SCREEN_HEIGHT = display_info.current_w, display_info.current_h
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.FULLSCREEN)
    # Check the mixer itself here; each screen loads its own track on transition,
    # so a missing menu file cannot prevent the save-screen music from playing.
    music_available = pygame.mixer.get_init() is not None
    active_music_path: Path | None = None
    clock = pygame.time.Clock()
    title_font = pygame.font.SysFont("dejavusans", 42, bold=True)
    heading_font = pygame.font.SysFont("dejavusans", 25, bold=True)
    body_font = pygame.font.SysFont("dejavusans", 18)
    small_font = pygame.font.SysFont("dejavusans", 14)
    (menu_background, menu_tree_layers,
     menu_cloud_layers, menu_fog_layers) = create_forest_backdrop()

    saves = game_world.list_save_slots(SAVE_DIR, LEGACY_SAVE_PATH)
    state = "menu"
    selected_save: tuple[int, Path, int] | None = None
    current_world: game_world.World | None = None
    map_surface: pygame.Surface | None = None
    local_area: LocalArea | None = None
    local_surface: pygame.Surface | None = None
    local_minimap_surface: pygame.Surface | None = None
    current_interior: Interior | None = None
    current_interior_surface: pygame.Surface | None = None
    outdoor_pickups: list[Pickup] = []
    interior_pickups: list[Pickup] = []
    character: Character | None = None
    zombies: list[Zombie] = []
    damage_popups: list[DamagePopup] = []
    locked_zombie_id: str | None = None
    lock_on_enabled = False
    sprint_toggled = False
    controller = None
    default_controller_bindings = {
        "attack": 0, "interact": 1, "reload": 2, "flashlight": 3,
        "lock_on": 4, "inventory": 7, "sprint": 8,
    }
    default_keyboard_bindings = {
        "move_up": pygame.K_w, "move_down": pygame.K_s,
        "move_left": pygame.K_a, "move_right": pygame.K_d,
        "attack": pygame.K_SPACE, "interact": pygame.K_e, "reload": pygame.K_r,
        "flashlight": pygame.K_f, "lock_on": pygame.K_q,
        "inventory": pygame.K_i, "sprint_toggle": pygame.K_TAB,
    }
    controller_bindings = default_controller_bindings.copy()
    keyboard_bindings = default_keyboard_bindings.copy()
    try:
        saved_bindings = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
        if isinstance(saved_bindings, dict):
            saved_controller = saved_bindings.get("controller", saved_bindings)
            saved_keyboard = saved_bindings.get("keyboard", {})
            for action in controller_bindings:
                if isinstance(saved_controller, dict) and action in saved_controller:
                    button = int(saved_controller[action])
                    if 0 <= button <= 32:
                        controller_bindings[action] = button
            if isinstance(saved_keyboard, dict):
                for action in keyboard_bindings:
                    if action in saved_keyboard:
                        key = int(saved_keyboard[action])
                        if 0 <= key <= 2047:
                            keyboard_bindings[action] = key
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        pass
    controller_action_names = {
        "attack": "Attack", "interact": "Interact / Pick Up", "reload": "Reload",
        "flashlight": "Flashlight", "lock_on": "Lock On", "sprint": "Sprint Toggle",
        "inventory": "Inventory",
    }
    keyboard_action_names = {
        "move_up": "Move Up", "move_down": "Move Down",
        "move_left": "Move Left", "move_right": "Move Right",
        "attack": "Attack", "interact": "Interact / Pick Up", "reload": "Reload",
        "flashlight": "Flashlight", "lock_on": "Lock On",
        "inventory": "Inventory", "sprint_toggle": "Sprint Toggle",
    }
    controller_button_names = {
        0: "A / Cross", 1: "B / Circle", 2: "X / Square", 3: "Y / Triangle",
        4: "LB / L1", 5: "RB / R1", 6: "Back / Select", 7: "Start",
        8: "L3", 9: "R3",
    }
    settings_selected = 0
    remap_selected = 0
    remapping_action: str | None = None
    remapping_key_action: str | None = None
    inventory_axis_direction = 0
    inventory_axis_timer = 0.0
    weapon_cooldown = 0.0
    reload_feedback = ""
    reload_feedback_timer = 0.0
    inventory_open = False
    inventory_selected = 0
    save_scroll = 0
    menu_selected = 0 if saves else 1
    name_input = ""
    name_replace_on_type = True
    name_entry_mode = "new"
    name_return_state = "menu"
    name_target: tuple[int, Path, int] | None = None
    delete_target: tuple[int, Path, int] | None = None
    save_selected = 0
    save_action_focus = 0  # load, rename, delete
    world_selected = 0  # explore, main menu
    region_selection_x, region_selection_y = 60, 120
    saved_region_x, saved_region_y = 60, 120
    menu_scene_time = 0.0
    storm_rng = random.Random(secrets.randbits(32))
    lightning_cooldown = storm_rng.uniform(4.0, 10.0)
    lightning_remaining = 0.0
    lightning_points: list[tuple[int, int]] = []
    running = True

    if pygame.joystick.get_count() > 0:
        try:
            controller = pygame.joystick.Joystick(0)
            controller.init()
        except pygame.error:
            controller = None

    menu_buttons = {
        "continue": pygame.Rect(SCREEN_WIDTH // 2 - 150, SCREEN_HEIGHT // 2 + 35, 300, 54),
        "new": pygame.Rect(SCREEN_WIDTH // 2 - 150, SCREEN_HEIGHT // 2 + 101, 300, 54),
        "settings": pygame.Rect(SCREEN_WIDTH // 2 - 150, SCREEN_HEIGHT // 2 + 167, 300, 54),
        "quit": pygame.Rect(SCREEN_WIDTH // 2 - 150, SCREEN_HEIGHT // 2 + 233, 300, 54),
    }
    back_button = pygame.Rect(50, 35, 120, 42)
    confirm_yes = pygame.Rect(SCREEN_WIDTH // 2 - 170, SCREEN_HEIGHT // 2 + 45, 150, 52)
    confirm_no = pygame.Rect(SCREEN_WIDTH // 2 + 20, SCREEN_HEIGHT // 2 + 45, 150, 52)
    menu_button = pygame.Rect(SCREEN_WIDTH - 50 - 150, 32, 150, 42)
    explore_button = pygame.Rect(menu_button.x - 175, 32, 160, 42)
    map_viewport = pygame.Rect(55, 105, min(500, int(SCREEN_WIDTH * 0.40)),
                               min(640, SCREEN_HEIGHT - 160))
    local_viewport = pygame.Rect(0, 0, SCREEN_WIDTH, SCREEN_HEIGHT)

    def activate_menu_choice(action: str) -> None:
        nonlocal state, save_scroll, save_selected, save_action_focus, name_input, name_entry_mode
        nonlocal name_replace_on_type
        nonlocal name_target, name_return_state, running
        nonlocal settings_selected
        if action == "continue" and saves:
            state = "saves"
            save_scroll = 0
            save_selected = 0
            save_action_focus = 0
        elif action == "new":
            name_entry_mode = "new"
            name_target = None
            name_return_state = "menu"
            name_input = f"New World {game_world.next_slot_number(SAVE_DIR)}"
            name_replace_on_type = True
            pygame.key.start_text_input()
            state = "name_entry"
        elif action == "settings":
            settings_selected = 0
            state = "settings"
        elif action == "quit":
            running = False

    def save_controller_bindings() -> None:
        SETTINGS_PATH.write_text(json.dumps({"controller": controller_bindings,
                                             "keyboard": keyboard_bindings}, indent=2) + "\n",
                                 encoding="utf-8")

    def assign_controller_button(action: str, button: int) -> None:
        previous_button = controller_bindings[action]
        for other_action, current_button in controller_bindings.items():
            if other_action != action and current_button == button:
                controller_bindings[other_action] = previous_button
        controller_bindings[action] = button
        save_controller_bindings()

    def controller_button_label(button: int) -> str:
        return controller_button_names.get(button, f"Button {button}")

    def keyboard_key_label(key: int) -> str:
        return pygame.key.name(key).upper()

    def assign_keyboard_key(action: str, key: int) -> None:
        previous_key = keyboard_bindings[action]
        for other_action, current_key in keyboard_bindings.items():
            if other_action != action and current_key == key:
                keyboard_bindings[other_action] = previous_key
        keyboard_bindings[action] = key
        save_controller_bindings()

    def reset_controller_defaults() -> None:
        controller_bindings.clear()
        controller_bindings.update(default_controller_bindings)
        save_controller_bindings()

    def reset_keyboard_defaults() -> None:
        keyboard_bindings.clear()
        keyboard_bindings.update(default_keyboard_bindings)
        save_controller_bindings()

    menu_actions = ("continue", "new", "settings", "quit")
    settings_actions = ("controller", "keyboard", "back")
    remappable_actions = tuple(controller_action_names)
    keyboard_remappable_actions = tuple(keyboard_action_names)

    def remap_row_rect(index: int, keyboard: bool = False) -> pygame.Rect:
        panel_height = min(760, SCREEN_HEIGHT - 40)
        panel_top = (SCREEN_HEIGHT - panel_height) // 2
        row_count = (len(keyboard_remappable_actions) if keyboard
                     else len(remappable_actions)) + 1
        row_step = max(30, min(44, (panel_height - 172) // row_count))
        row_height = max(26, min(38, row_step - 6))
        return pygame.Rect(SCREEN_WIDTH // 2 - 350, panel_top + 102 + index * row_step,
                           700, row_height)
    save_actions = ("load", "rename", "delete")

    def load_saved_world(save: tuple[int, Path, int]) -> None:
        nonlocal selected_save, current_world, map_surface, character, state, world_selected
        nonlocal region_selection_x, region_selection_y, saved_region_x, saved_region_y
        selected_save = save
        current_world, map_surface = open_world(save)
        character = None
        saved = game_world.load_character(save[1])
        saved_region_x = max(0, min(current_world.width - 1,
                                    int(saved.get("region_x", current_world.width // 2))))
        saved_region_y = max(0, min(current_world.height - 1,
                                    int(saved.get("region_y", current_world.height // 2))))
        region_selection_x = max(0, min(current_world.width - 1,
                                        saved_region_x))
        region_selection_y = max(0, min(current_world.height - 1,
                                        saved_region_y))
        world_selected = 0
        state = "world"

    def enter_explore_area() -> None:
        nonlocal local_area, local_surface, local_minimap_surface, character, zombies
        nonlocal outdoor_pickups, interior_pickups
        nonlocal locked_zombie_id, lock_on_enabled
        nonlocal current_interior, current_interior_surface, state
        if current_world is None or selected_save is None:
            return
        previous = game_world.load_character(selected_save[1])
        previous_x = int(previous.get("region_x", current_world.width // 2))
        previous_y = int(previous.get("region_y", current_world.height // 2))
        region_changed = (previous_x, previous_y) != (region_selection_x, region_selection_y)
        region_seed = local_region_seed(current_world.seed, region_selection_x,
                                        region_selection_y)
        local_area = generate_region_area(current_world, region_selection_x,
                                          region_selection_y)
        local_surface = make_local_surface(local_area)
        local_minimap_surface = pygame.transform.smoothscale(local_surface, (390, 260))
        character = character_from_save(selected_save[1], local_area)
        if region_changed:
            travel_km = math.hypot(region_selection_x - previous_x,
                                   region_selection_y - previous_y) * current_world.cell_size_km
            character.game_time_minutes += max(10, travel_km * 5)
            character.x, character.y = float(local_area.spawn_x), float(local_area.spawn_y)
            character.inside_building = None
            character.outside_x = character.outside_y = None
        anchor = ((character.outside_x, character.outside_y)
                  if character.inside_building is not None
                  and character.outside_x is not None and character.outside_y is not None
                  else (character.x, character.y))
        zombies = generate_zombies(local_area, region_seed, region_selection_x,
                                   region_selection_y, character.defeated_zombies, anchor)
        locked_zombie_id = None
        lock_on_enabled = False
        outdoor_pickups = generate_outdoor_pickups(
            local_area, region_seed, region_selection_x, region_selection_y,
            character.collected_items)
        character.region_x, character.region_y = region_selection_x, region_selection_y
        current_interior = None
        current_interior_surface = None
        interior_pickups = []
        if character.inside_building is not None:
            building = next((item for item in local_area.buildings
                             if item.name == character.inside_building), None)
            if building is not None:
                current_interior = generate_interior(region_seed, building)
                interior_pickups = generate_interior_pickups(
                    current_interior, region_seed, character.region_x, character.region_y,
                    character.collected_items)
                if not can_occupy(current_interior, character.x, character.y):
                    character.x = current_interior.door_x
                    character.y = current_interior.door_y - 1
                tile_size = max(1, min(40, SCREEN_WIDTH // current_interior.width,
                                       SCREEN_HEIGHT // current_interior.height))
                current_interior_surface = make_interior_surface(current_interior, tile_size)
        save_character(selected_save, character)
        state = "local"

    def travel_to_adjacent_chunk(dx: int, dy: int) -> bool:
        """Seamlessly load the neighboring region cell at a connected road edge."""
        nonlocal local_area, local_surface, local_minimap_surface, zombies
        nonlocal outdoor_pickups, interior_pickups
        nonlocal locked_zombie_id
        nonlocal region_selection_x, region_selection_y
        if current_world is None or character is None or selected_save is None:
            return False
        new_x, new_y = character.region_x + dx, character.region_y + dy
        if not (0 <= new_x < current_world.width and 0 <= new_y < current_world.height):
            return False

        new_seed = local_region_seed(current_world.seed, new_x, new_y)
        local_area = generate_region_area(current_world, new_x, new_y)
        local_surface = make_local_surface(local_area)
        local_minimap_surface = pygame.transform.smoothscale(local_surface, (390, 260))
        # The main roads cross every scene, so adjacent scenes join at a clear road.
        if dx:
            character.x = float(LOCAL_WIDTH - 1.5 if dx < 0 else 1.5)
            character.y = 66.0
        else:
            character.x = 41.0
            character.y = float(LOCAL_HEIGHT - 1.5 if dy < 0 else 1.5)
        zombies = generate_zombies(local_area, new_seed, new_x, new_y,
                                   character.defeated_zombies, (character.x, character.y))
        damage_popups.clear()
        locked_zombie_id = None
        outdoor_pickups = generate_outdoor_pickups(
            local_area, new_seed, new_x, new_y, character.collected_items)
        interior_pickups = []
        character.region_x, character.region_y = new_x, new_y
        character.game_time_minutes += 10
        region_selection_x, region_selection_y = new_x, new_y
        save_character(selected_save, character)
        return True

    def attack_with_equipped_weapon() -> None:
        nonlocal weapon_cooldown, reload_feedback, reload_feedback_timer
        if character is None or current_interior is not None or weapon_cooldown > 0:
            return
        weapon = character.equipped_weapon
        if weapon == "Handgun" and character.handgun_loaded <= 0:
            reload_feedback = "MAGAZINE EMPTY · PRESS R TO RELOAD"
            reload_feedback_timer = 1.8
            return
        weapon_cooldown = 0.48 if weapon == "Kitchen Knife" else 0.62
        if weapon == "Handgun":
            character.handgun_loaded -= 1
        facing_length = math.hypot(character.facing_x, character.facing_y) or 1.0
        candidates: list[tuple[float, Zombie]] = []
        for zombie in zombies:
            dx, dy = zombie.x - character.x, zombie.y - character.y
            distance = math.hypot(dx, dy)
            reach = 1.8 if weapon == "Kitchen Knife" else 12.0
            alignment = ((dx * character.facing_x + dy * character.facing_y)
                         / (distance * facing_length)) if distance else 1.0
            side_distance = abs(dx * character.facing_y - dy * character.facing_x) / facing_length
            max_side = 0.9 if weapon == "Kitchen Knife" else 0.45 + distance * 0.035
            if distance <= reach and alignment >= (0.15 if weapon == "Kitchen Knife" else 0.94) \
                    and side_distance <= max_side:
                candidates.append((distance, zombie))
        if candidates:
            target = min(candidates, key=lambda pair: pair[0])[1]
            damage = 45 if weapon == "Kitchen Knife" else 60
            target.health = max(0, target.health - damage)
            damage_popups.append(DamagePopup(target.x, target.y, f"-{damage}"))
            reload_feedback = f"{weapon.upper()} HIT  ·  {damage} DAMAGE"
            reload_feedback_timer = 1.1
            if target.health <= 0:
                reload_feedback = f"ZOMBIE DEFEATED  ·  {damage} DAMAGE"
                character.defeated_zombies.append(target.zombie_id)
                zombies.remove(target)
        else:
            reload_feedback = "GUNSHOT MISSED" if weapon == "Handgun" else "KNIFE SWING MISSED"
            reload_feedback_timer = 0.9
        if selected_save is not None:
            save_character(selected_save, character)

    def reload_handgun() -> None:
        nonlocal reload_feedback, reload_feedback_timer
        if character is None or character.equipped_weapon != "Handgun":
            return
        missing_rounds = 6 - character.handgun_loaded
        if missing_rounds <= 0:
            reload_feedback = "MAGAZINE FULL"
        elif character.ammo_9mm <= 0:
            reload_feedback = "NO SPARE 9MM AMMO"
        else:
            loaded = min(missing_rounds, character.ammo_9mm)
            character.handgun_loaded += loaded
            character.ammo_9mm -= loaded
            reload_feedback = f"RELOADED · {character.handgun_loaded}/6"
            if selected_save is not None:
                save_character(selected_save, character)
        reload_feedback_timer = 1.8

    def activate_world_choice(choice: int) -> None:
        nonlocal state, saves
        if choice == 0:
            enter_explore_area()
        else:
            saves = game_world.list_save_slots(SAVE_DIR, LEGACY_SAVE_PATH)
            state = "menu"

    def begin_save_rename(save: tuple[int, Path, int]) -> None:
        nonlocal name_entry_mode, name_target, name_return_state, name_input, state
        nonlocal name_replace_on_type
        name_entry_mode = "rename"
        name_target = save
        name_return_state = "saves"
        name_input = game_world.load_save_name(save[1], save[0])
        name_replace_on_type = True
        pygame.key.start_text_input()
        state = "name_entry"

    def begin_save_delete(save: tuple[int, Path, int]) -> None:
        nonlocal delete_target, state
        delete_target = save
        state = "delete_confirm"

    while running:
        if controller is None and pygame.joystick.get_count() > 0:
            try:
                controller = pygame.joystick.Joystick(0)
                controller.init()
            except pygame.error:
                controller = None
        if saves:
            save_selected = min(save_selected, len(saves) - 1)
            save_scroll = min(save_scroll, max(0, len(saves) - 7))
        else:
            save_selected = save_scroll = 0
        if state == "menu" and not saves and menu_selected == 0:
            menu_selected = 1
        delta_time = min(clock.get_time() / 1000.0, 0.05)
        nearby_pickup: Pickup | None = None
        nearby_distance = float("inf")
        if state == "menu":
            menu_scene_time += delta_time
            if lightning_remaining > 0:
                lightning_remaining = max(0.0, lightning_remaining - delta_time)
                if lightning_remaining == 0:
                    lightning_points = []
            else:
                lightning_cooldown -= delta_time
                if lightning_cooldown <= 0:
                    bolt_x = storm_rng.randrange(max(40, SCREEN_WIDTH // 12),
                                                 max(41, SCREEN_WIDTH - SCREEN_WIDTH // 12))
                    bolt_y = 0
                    lightning_points = [(bolt_x, bolt_y)]
                    while bolt_y < SCREEN_HEIGHT * 0.63:
                        bolt_y += storm_rng.randint(28, 58)
                        bolt_x = max(8, min(SCREEN_WIDTH - 8,
                                            bolt_x + storm_rng.randint(-34, 34)))
                        lightning_points.append((bolt_x, min(bolt_y, SCREEN_HEIGHT)))
                    lightning_remaining = 0.16
                    lightning_cooldown = storm_rng.uniform(5.0, 15.0)
        if state == "local" and character is not None:
            # One complete in-game day takes twelve minutes of active play.
            previous_hour = int(character.game_time_minutes // 60)
            character.game_time_minutes += delta_time * 120
            if int(character.game_time_minutes // 60) != previous_hour and selected_save is not None:
                save_character(selected_save, character)
            weapon_cooldown = max(0.0, weapon_cooldown - delta_time)
            reload_feedback_timer = max(0.0, reload_feedback_timer - delta_time)
            for popup in damage_popups:
                popup.timer -= delta_time
            damage_popups[:] = [popup for popup in damage_popups if popup.timer > 0]
        mouse_position = pygame.mouse.get_pos()
        for event in pygame.event.get():
            if event.type == pygame.JOYDEVICEREMOVED:
                controller = None
            elif event.type == pygame.JOYBUTTONDOWN:
                if state == "controller_remap" and remapping_action is not None:
                    assign_controller_button(remapping_action, event.button)
                    remapping_action = None
                elif state == "keyboard_remap" and remapping_key_action is not None:
                    # Controller input cannot be used as a keyboard binding; cancel the listen state.
                    remapping_key_action = None
                else:
                    if state == "local":
                        action = next((name for name, button in controller_bindings.items()
                                       if button == event.button), None)
                        key = (keyboard_bindings.get("sprint_toggle") if action == "sprint"
                               else keyboard_bindings.get(action))
                        if inventory_open and action == "interact":
                            key = pygame.K_ESCAPE
                    else:
                        key = {0: pygame.K_RETURN, 1: pygame.K_ESCAPE,
                               7: pygame.K_RETURN}.get(event.button)
                    if key is not None:
                        pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=key))
            elif event.type == pygame.JOYHATMOTION:
                hat_x, hat_y = event.value
                if state == "local" and inventory_open:
                    key = (pygame.K_UP if hat_y > 0 else pygame.K_DOWN if hat_y < 0
                           else pygame.K_LEFT if hat_x < 0 else pygame.K_RIGHT if hat_x > 0 else None)
                    if key is not None:
                        pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=key))
                elif state != "local":
                    key = (pygame.K_LEFT if hat_x < 0 else pygame.K_RIGHT if hat_x > 0
                           else pygame.K_UP if hat_y > 0 else pygame.K_DOWN if hat_y < 0 else None)
                    if key is not None:
                        pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=key))
            elif event.type == pygame.QUIT:
                if selected_save is not None and character is not None:
                    save_character(selected_save, character)
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                if state == "controller_remap":
                    if remapping_action is not None:
                        remapping_action = None
                    else:
                        state = "settings"
                elif state == "keyboard_remap":
                    if remapping_key_action is not None:
                        remapping_key_action = None
                    else:
                        state = "settings"
                elif state == "settings":
                    state = "menu"
                elif state == "name_entry":
                    pygame.key.stop_text_input()
                    state = name_return_state
                    name_target = None
                elif state == "delete_confirm":
                    state = "saves"
                    delete_target = None
                elif state == "local":
                    if inventory_open:
                        inventory_open = False
                    else:
                        if selected_save is not None and character is not None:
                            save_character(selected_save, character)
                        state = "world"
                elif state != "menu":
                    state = "menu"
            elif state == "name_entry" and event.type == pygame.TEXTINPUT:
                if name_replace_on_type:
                    name_input = event.text[:32]
                    name_replace_on_type = False
                else:
                    name_input = (name_input + event.text)[:32]
            elif state == "name_entry" and event.type == pygame.KEYDOWN:
                if event.key == pygame.K_BACKSPACE:
                    if name_replace_on_type:
                        name_input = ""
                        name_replace_on_type = False
                    name_input = name_input[:-1]
                elif event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                    chosen_name = name_input.strip()
                    if chosen_name:
                        pygame.key.stop_text_input()
                        if name_entry_mode == "new":
                            selected_save = create_new_game(chosen_name)
                            saves = game_world.list_save_slots(SAVE_DIR, LEGACY_SAVE_PATH)
                            current_world, map_surface = open_world(selected_save)
                            character = None
                            world_selected = 0
                            state = "world"
                        elif name_target is not None:
                            slot_number, path, seed = name_target
                            game_world.save_game(SAVE_DIR, slot_number, seed,
                                                 save_name=chosen_name)
                            saves = game_world.list_save_slots(SAVE_DIR, LEGACY_SAVE_PATH)
                            state = "saves"
                            name_target = None
                    else:
                        name_input = ""
            elif state == "delete_confirm" and event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_y, pygame.K_RETURN, pygame.K_KP_ENTER):
                    if delete_target is not None:
                        delete_target[1].unlink(missing_ok=True)
                        if selected_save is not None and selected_save[1] == delete_target[1]:
                            selected_save = None
                            character = None
                        saves = game_world.list_save_slots(SAVE_DIR, LEGACY_SAVE_PATH)
                    delete_target = None
                    state = "saves"
                elif event.key == pygame.K_n:
                    delete_target = None
                    state = "saves"
            elif state == "menu" and event.type == pygame.KEYDOWN and event.key in (
                    pygame.K_UP, pygame.K_w, pygame.K_DOWN, pygame.K_s):
                step = -1 if event.key in (pygame.K_UP, pygame.K_w) else 1
                menu_selected = (menu_selected + step) % len(menu_actions)
                while menu_selected == 0 and not saves:
                    menu_selected = (menu_selected + step) % len(menu_actions)
            elif state == "menu" and event.type == pygame.KEYDOWN and event.key in (
                    pygame.K_RETURN, pygame.K_KP_ENTER, pygame.K_SPACE):
                activate_menu_choice(menu_actions[menu_selected])
            elif state == "settings" and event.type == pygame.KEYDOWN and event.key in (
                    pygame.K_UP, pygame.K_w, pygame.K_DOWN, pygame.K_s):
                step = -1 if event.key in (pygame.K_UP, pygame.K_w) else 1
                settings_selected = (settings_selected + step) % len(settings_actions)
            elif state == "settings" and event.type == pygame.KEYDOWN and event.key in (
                    pygame.K_RETURN, pygame.K_KP_ENTER, pygame.K_SPACE):
                if settings_actions[settings_selected] == "controller":
                    remap_selected = 0
                    state = "controller_remap"
                elif settings_actions[settings_selected] == "keyboard":
                    remap_selected = 0
                    state = "keyboard_remap"
                else:
                    state = "menu"
            elif state == "controller_remap" and event.type == pygame.KEYDOWN and event.key in (
                    pygame.K_UP, pygame.K_w, pygame.K_DOWN, pygame.K_s) and remapping_action is None:
                step = -1 if event.key in (pygame.K_UP, pygame.K_w) else 1
                remap_selected = (remap_selected + step) % (len(remappable_actions) + 1)
            elif state == "controller_remap" and event.type == pygame.KEYDOWN and event.key in (
                    pygame.K_RETURN, pygame.K_KP_ENTER, pygame.K_SPACE) and remapping_action is None:
                if remap_selected == len(remappable_actions):
                    reset_controller_defaults()
                else:
                    remapping_action = remappable_actions[remap_selected]
            elif state == "keyboard_remap" and event.type == pygame.KEYDOWN and event.key in (
                    pygame.K_UP, pygame.K_w, pygame.K_DOWN, pygame.K_s) and remapping_key_action is None:
                step = -1 if event.key in (pygame.K_UP, pygame.K_w) else 1
                remap_selected = (remap_selected + step) % (len(keyboard_remappable_actions) + 1)
            elif state == "keyboard_remap" and event.type == pygame.KEYDOWN and remapping_key_action is not None:
                assign_keyboard_key(remapping_key_action, event.key)
                remapping_key_action = None
            elif state == "keyboard_remap" and event.type == pygame.KEYDOWN and event.key in (
                    pygame.K_RETURN, pygame.K_KP_ENTER, pygame.K_SPACE) and remapping_key_action is None:
                if remap_selected == len(keyboard_remappable_actions):
                    reset_keyboard_defaults()
                else:
                    remapping_key_action = keyboard_remappable_actions[remap_selected]
            elif state == "saves" and event.type == pygame.KEYDOWN and event.key in (
                    pygame.K_UP, pygame.K_w, pygame.K_DOWN, pygame.K_s):
                if saves:
                    step = -1 if event.key in (pygame.K_UP, pygame.K_w) else 1
                    save_selected = max(0, min(len(saves) - 1, save_selected + step))
                    if save_selected < save_scroll:
                        save_scroll = save_selected
                    elif save_selected >= save_scroll + 7:
                        save_scroll = save_selected - 6
                    save_action_focus = 0
            elif state == "saves" and event.type == pygame.KEYDOWN and event.key in (
                    pygame.K_LEFT, pygame.K_a, pygame.K_RIGHT, pygame.K_d, pygame.K_TAB):
                step = -1 if event.key in (pygame.K_LEFT, pygame.K_a) else 1
                save_action_focus = (save_action_focus + step) % len(save_actions)
            elif state == "saves" and event.type == pygame.KEYDOWN and saves and event.key in (
                    pygame.K_RETURN, pygame.K_KP_ENTER, pygame.K_SPACE, pygame.K_r, pygame.K_DELETE):
                save = saves[save_selected]
                action = ("rename" if event.key == pygame.K_r else
                          "delete" if event.key == pygame.K_DELETE else
                          save_actions[save_action_focus])
                if action == "load":
                    load_saved_world(save)
                elif action == "rename":
                    begin_save_rename(save)
                else:
                    begin_save_delete(save)
            elif state == "world" and event.type == pygame.KEYDOWN and event.key in (
                    pygame.K_LEFT, pygame.K_a, pygame.K_RIGHT, pygame.K_d,
                    pygame.K_UP, pygame.K_w, pygame.K_DOWN, pygame.K_s):
                dx = int(event.key in (pygame.K_RIGHT, pygame.K_d)) - int(
                    event.key in (pygame.K_LEFT, pygame.K_a))
                dy = int(event.key in (pygame.K_DOWN, pygame.K_s)) - int(
                    event.key in (pygame.K_UP, pygame.K_w))
                if current_world is not None:
                    region_selection_x = max(0, min(current_world.width - 1, region_selection_x + dx))
                    region_selection_y = max(0, min(current_world.height - 1, region_selection_y + dy))
                    world_selected = 0
            elif state == "world" and event.type == pygame.KEYDOWN and event.key == pygame.K_TAB:
                world_selected = 1 - world_selected
            elif state == "world" and event.type == pygame.KEYDOWN and event.key in (
                    pygame.K_RETURN, pygame.K_KP_ENTER, pygame.K_SPACE):
                activate_world_choice(world_selected)
            elif state == "local" and event.type == pygame.KEYDOWN and event.key == keyboard_bindings["inventory"]:
                inventory_open = not inventory_open
                inventory_selected = 0
            elif state == "local" and character is not None and event.type == pygame.KEYDOWN and event.key == keyboard_bindings["flashlight"]:
                character.flashlight_on = not character.flashlight_on
                if selected_save is not None:
                    save_character(selected_save, character)
            elif (state == "local" and event.type == pygame.KEYDOWN
                  and event.key == keyboard_bindings["sprint_toggle"] and not getattr(event, "repeat", False)):
                sprint_toggled = not sprint_toggled
            elif (state == "local" and not inventory_open and character is not None
                  and event.type == pygame.KEYDOWN and event.key == keyboard_bindings["lock_on"]):
                lock_on_enabled = not lock_on_enabled
                locked_zombie_id = None
                if lock_on_enabled and zombies:
                    nearest = min(zombies, key=lambda zombie: math.hypot(
                        zombie.x - character.x, zombie.y - character.y))
                    if math.hypot(nearest.x - character.x, nearest.y - character.y) <= 32:
                        locked_zombie_id = nearest.zombie_id
            elif state == "local" and inventory_open and event.type == pygame.KEYDOWN and character is not None and event.key in (
                    pygame.K_UP, pygame.K_w, pygame.K_DOWN, pygame.K_s,
                    keyboard_bindings["move_up"], keyboard_bindings["move_down"]):
                inventory_items = inventory_items_for(character)
                if inventory_items:
                    delta = -1 if event.key in (pygame.K_UP, pygame.K_w, keyboard_bindings["move_up"]) else 1
                    inventory_selected = (inventory_selected + delta) % len(inventory_items)
            elif state == "local" and inventory_open and event.type == pygame.KEYDOWN and event.key in (
                    pygame.K_RETURN, pygame.K_KP_ENTER, pygame.K_SPACE,
                    keyboard_bindings["attack"]) and character is not None:
                inventory_items = inventory_items_for(character)
                if inventory_items:
                    inventory_selected %= len(inventory_items)
                    item = inventory_items[inventory_selected]
                    if item == "Flashlight":
                        character.flashlight_on = not character.flashlight_on
                        save_character(selected_save, character)
                    elif item in {"Kitchen Knife", "Handgun"}:
                        character.equipped_weapon = item
                        save_character(selected_save, character)
                    elif item == "First Aid Kit" and character.health < 100:
                        character.health = min(100, character.health + 35)
                        character.inventory[item] -= 1
                        if character.inventory[item] <= 0:
                            del character.inventory[item]
                        inventory_selected = min(inventory_selected,
                                                 max(0, len(character.inventory) - 1))
                        save_character(selected_save, character)
            elif state == "local" and not inventory_open and event.type == pygame.KEYDOWN and event.key == keyboard_bindings["attack"]:
                attack_with_equipped_weapon()
            elif state == "local" and not inventory_open and event.type == pygame.KEYDOWN and event.key == keyboard_bindings["reload"]:
                reload_handgun()
            elif state == "local" and not inventory_open and event.type == pygame.KEYDOWN and character is not None and local_area is not None and event.key == keyboard_bindings["interact"]:
                scene_pickups = interior_pickups if current_interior is not None else outdoor_pickups
                pickup = min(scene_pickups,
                             key=lambda item: math.hypot(item.x - character.x,
                                                        item.y - character.y),
                             default=None)
                if pickup is not None and math.hypot(pickup.x - character.x,
                                                     pickup.y - character.y) <= 1.6:
                    if pickup.item == "9mm Ammo":
                        character.ammo_9mm += pickup.quantity
                    else:
                        character.inventory[pickup.item] = character.inventory.get(pickup.item, 0) + pickup.quantity
                    character.collected_items.append(pickup.pickup_id)
                    scene_pickups.remove(pickup)
                    if selected_save is not None:
                        save_character(selected_save, character)
                elif current_interior is not None:
                    if (abs(character.x - current_interior.door_x)
                            + abs(character.y - current_interior.door_y) <= 1):
                        building = next((item for item in local_area.buildings
                                         if item.name == current_interior.name), None)
                        character.inside_building = None
                        if building is not None:
                            character.x = float(building.door_x)
                            character.y = float(building.door_y + 1 if building.door_side == "south"
                                               else building.door_y - 1)
                        else:
                            character.x = float(local_area.spawn_x)
                            character.y = float(local_area.spawn_y)
                        if not can_occupy(local_area, character.x, character.y):
                            if building is not None:
                                character.x = float(building.door_x)
                                character.y = float(building.door_y + (1 if building.door_side == "south" else -1))
                        character.outside_x = character.outside_y = None
                        current_interior = None
                        current_interior_surface = None
                        interior_pickups = []
                        if selected_save is not None:
                            save_character(selected_save, character)
                else:
                    for building in local_area.buildings:
                        if (abs(character.x - building.door_x)
                                + abs(character.y - building.door_y) <= 1):
                            area_seed = local_region_seed(current_world.seed, character.region_x,
                                                          character.region_y)
                            current_interior = generate_interior(area_seed, building)
                            interior_pickups = generate_interior_pickups(
                                current_interior, area_seed, character.region_x,
                                character.region_y, character.collected_items)
                            character.inside_building = building.name
                            character.outside_x, character.outside_y = character.x, character.y
                            character.x = current_interior.door_x
                            character.y = current_interior.door_y - 1
                            tile_size = max(1, min(40, SCREEN_WIDTH // current_interior.width,
                                                   SCREEN_HEIGHT // current_interior.height))
                            current_interior_surface = make_interior_surface(current_interior, tile_size)
                            if selected_save is not None:
                                save_character(selected_save, character)
                            break
            elif state == "menu" and event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                for index, action in enumerate(menu_actions):
                    if menu_buttons[action].collidepoint(event.pos):
                        menu_selected = index
                        activate_menu_choice(action)
                        break
            elif state == "settings" and event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                panel = pygame.Rect(SCREEN_WIDTH // 2 - 390, SCREEN_HEIGHT // 2 - 230, 780, 460)
                settings_rows = [
                    pygame.Rect(panel.centerx - 230, panel.y + 190, 460, 54),
                    pygame.Rect(panel.centerx - 230, panel.y + 260, 460, 54),
                    pygame.Rect(panel.centerx - 230, panel.y + 330, 460, 54),
                ]
                for index, row in enumerate(settings_rows):
                    if row.collidepoint(event.pos):
                        settings_selected = index
                        if settings_actions[index] == "controller":
                            remap_selected = 0
                            state = "controller_remap"
                        elif settings_actions[index] == "keyboard":
                            remap_selected = 0
                            state = "keyboard_remap"
                        else:
                            state = "menu"
                        break
            elif state == "controller_remap" and event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if back_button.collidepoint(event.pos):
                    if remapping_action is not None:
                        remapping_action = None
                    else:
                        state = "settings"
                elif remapping_action is None:
                    for index, action in enumerate(remappable_actions):
                        if remap_row_rect(index).collidepoint(event.pos):
                            remap_selected = index
                            remapping_action = action
                            break
                    reset_rect = remap_row_rect(len(remappable_actions))
                    if reset_rect.collidepoint(event.pos):
                        remap_selected = len(remappable_actions)
                        reset_controller_defaults()
            elif state == "keyboard_remap" and event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if back_button.collidepoint(event.pos):
                    if remapping_key_action is not None:
                        remapping_key_action = None
                    else:
                        state = "settings"
                elif remapping_key_action is None:
                    for index, action in enumerate(keyboard_remappable_actions):
                        if remap_row_rect(index, keyboard=True).collidepoint(event.pos):
                            remap_selected = index
                            remapping_key_action = action
                            break
                    reset_rect = remap_row_rect(len(keyboard_remappable_actions), keyboard=True)
                    if reset_rect.collidepoint(event.pos):
                        remap_selected = len(keyboard_remappable_actions)
                        reset_keyboard_defaults()
            elif state == "saves" and event.type == pygame.MOUSEWHEEL:
                save_scroll = max(0, min(save_scroll - event.y, max(0, len(saves) - 7)))
                save_selected = max(save_scroll, min(save_selected, save_scroll + 6))
            elif state == "saves" and event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if back_button.collidepoint(event.pos):
                    state = "menu"
                else:
                    visible_saves = saves[save_scroll:save_scroll + 7]
                    for index, save in enumerate(visible_saves):
                        row = pygame.Rect(170, 205 + index * 66, SCREEN_WIDTH - 340, 54)
                        load_rect = pygame.Rect(row.right - 260, row.y + 7, 78, 40)
                        rename_rect = pygame.Rect(row.right - 174, row.y + 7, 78, 40)
                        delete_rect = pygame.Rect(row.right - 88, row.y + 7, 78, 40)
                        save_selected = save_scroll + index
                        if rename_rect.collidepoint(event.pos):
                            save_action_focus = 1
                            begin_save_rename(save)
                            break
                        elif delete_rect.collidepoint(event.pos):
                            save_action_focus = 2
                            begin_save_delete(save)
                            break
                        elif load_rect.collidepoint(event.pos):
                            save_action_focus = 0
                            load_saved_world(save)
                            break
                        elif row.collidepoint(event.pos):
                            save_action_focus = 0
                            load_saved_world(save)
                            break
            elif state == "delete_confirm" and event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if confirm_yes.collidepoint(event.pos) and delete_target is not None:
                    delete_target[1].unlink(missing_ok=True)
                    if selected_save is not None and selected_save[1] == delete_target[1]:
                        selected_save = None
                        character = None
                    saves = game_world.list_save_slots(SAVE_DIR, LEGACY_SAVE_PATH)
                    delete_target = None
                    state = "saves"
                elif confirm_no.collidepoint(event.pos):
                    delete_target = None
                    state = "saves"
            elif state == "world" and event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if current_world is None:
                    continue
                scale = min(map_viewport.width / current_world.width,
                            map_viewport.height / current_world.height)
                map_width = int(current_world.width * scale)
                map_height = int(current_world.height * scale)
                map_x = map_viewport.x + (map_viewport.width - map_width) // 2
                map_y = map_viewport.y + (map_viewport.height - map_height) // 2
                if (map_x <= event.pos[0] < map_x + map_width
                        and map_y <= event.pos[1] < map_y + map_height):
                    region_selection_x = max(0, min(current_world.width - 1,
                                                    int((event.pos[0] - map_x) / scale)))
                    region_selection_y = max(0, min(current_world.height - 1,
                                                    int((event.pos[1] - map_y) / scale)))
                    world_selected = 0
                elif explore_button.collidepoint(event.pos) and selected_save is not None:
                    world_selected = 0
                    activate_world_choice(world_selected)
                elif menu_button.collidepoint(event.pos):
                    world_selected = 1
                    activate_world_choice(world_selected)
        if state == "local" and inventory_open and character is not None:
            if controller is not None and controller.get_init() and controller.get_numaxes() > 1:
                try:
                    stick_y = controller.get_axis(1)
                    axis_direction = -1 if stick_y < -0.65 else 1 if stick_y > 0.65 else 0
                    inventory_axis_timer -= delta_time
                    if axis_direction and (axis_direction != inventory_axis_direction
                                           or inventory_axis_timer <= 0):
                        items = inventory_items_for(character)
                        if items:
                            inventory_selected = (inventory_selected + axis_direction) % len(items)
                        inventory_axis_direction = axis_direction
                        inventory_axis_timer = 0.22
                    elif not axis_direction:
                        inventory_axis_direction = 0
                        inventory_axis_timer = 0.0
                except pygame.error:
                    controller = None
        if state == "local" and not inventory_open and character is not None and local_area is not None:
            keys = pygame.key.get_pressed()
            direction_x = int(keys[keyboard_bindings["move_right"]] or keys[pygame.K_RIGHT]) - int(
                keys[keyboard_bindings["move_left"]] or keys[pygame.K_LEFT])
            direction_y = int(keys[keyboard_bindings["move_down"]] or keys[pygame.K_DOWN]) - int(
                keys[keyboard_bindings["move_up"]] or keys[pygame.K_UP])
            right_x = right_y = 0.0
            if controller is not None and controller.get_init():
                try:
                    left_x, left_y = controller.get_axis(0), controller.get_axis(1)
                    hat_x, hat_y = controller.get_hat(0) if controller.get_numhats() else (0, 0)
                    if math.hypot(left_x, left_y) > 0.2:
                        direction_x, direction_y = left_x, left_y
                    elif hat_x or hat_y:
                        direction_x, direction_y = hat_x, -hat_y
                    right_x = controller.get_axis(2) if controller.get_numaxes() > 2 else 0.0
                    right_y = controller.get_axis(3) if controller.get_numaxes() > 3 else 0.0
                except pygame.error:
                    controller = None
            direction_length = math.hypot(direction_x, direction_y)
            if not lock_on_enabled and direction_length == 0 and math.hypot(right_x, right_y) > 0.3:
                aim_length = math.hypot(right_x, right_y)
                character.facing_x = round(right_x * 100 / aim_length)
                character.facing_y = round(right_y * 100 / aim_length)
            if direction_length and delta_time:
                old_cell = (math.floor(character.x + 0.5), math.floor(character.y + 0.5))
                if not lock_on_enabled:
                    character.facing_x = round(direction_x * 100 / direction_length)
                    character.facing_y = round(direction_y * 100 / direction_length)
                move_speed = (6.0 if sprint_toggled or keys[pygame.K_LSHIFT]
                              or keys[pygame.K_RSHIFT] else 3.0)
                distance = move_speed * delta_time / direction_length
                active_map = current_interior if current_interior is not None else local_area
                next_x = character.x + direction_x * distance
                next_y = character.y + direction_y * distance
                moved_chunk = False
                if current_interior is None:
                    if next_x < 0.22 and direction_x < 0:
                        moved_chunk = travel_to_adjacent_chunk(-1, 0)
                    elif next_x >= LOCAL_WIDTH - 1.28 and direction_x > 0:
                        moved_chunk = travel_to_adjacent_chunk(1, 0)
                    elif next_y < 0.22 and direction_y < 0:
                        moved_chunk = travel_to_adjacent_chunk(0, -1)
                    elif next_y >= LOCAL_HEIGHT - 1.28 and direction_y > 0:
                        moved_chunk = travel_to_adjacent_chunk(0, 1)
                if not moved_chunk:
                    if can_occupy(active_map, next_x, character.y):
                        character.x = next_x
                    if can_occupy(active_map, character.x, next_y):
                        character.y = next_y
                new_cell = (math.floor(character.x + 0.5), math.floor(character.y + 0.5))
                if new_cell != old_cell and selected_save is not None:
                    save_character(selected_save, character)

            if current_interior is None:
                for zombie in zombies:
                    zombie.attack_timer = max(0.0, zombie.attack_timer - delta_time)
                    zombie.path_timer = max(0.0, zombie.path_timer - delta_time)
                    dx, dy = character.x - zombie.x, character.y - zombie.y
                    distance = math.hypot(dx, dy)
                    if zombie_can_see_player(local_area, zombie.x, zombie.y,
                                             character.x, character.y):
                        zombie.alerted = True
                    if zombie.alerted:
                        target_cell = (math.floor(character.x + 0.5),
                                       math.floor(character.y + 0.5))
                        if (zombie.path_timer <= 0 or zombie.path_target != target_cell
                                or not zombie.path):
                            zombie.path = find_zombie_path(
                                local_area,
                                (math.floor(zombie.x + 0.5), math.floor(zombie.y + 0.5)),
                                target_cell)
                            zombie.path_target = target_cell
                            zombie.path_timer = 0.35
                    if distance <= 0.72:
                        if zombie.alerted and zombie.attack_timer == 0:
                            character.health = max(0, character.health - 8)
                            zombie.attack_timer = 0.9
                            if selected_save is not None:
                                save_character(selected_save, character)
                    elif zombie.alerted and zombie.path:
                        waypoint_x, waypoint_y = zombie.path[0]
                        path_dx, path_dy = waypoint_x - zombie.x, waypoint_y - zombie.y
                        path_distance = math.hypot(path_dx, path_dy)
                        step = 1.05 * delta_time
                        if path_distance <= step:
                            zombie.x, zombie.y = float(waypoint_x), float(waypoint_y)
                            zombie.path.pop(0)
                        elif path_distance:
                            next_x = zombie.x + path_dx * step / path_distance
                            next_y = zombie.y + path_dy * step / path_distance
                            if can_occupy(local_area, next_x, zombie.y):
                                zombie.x = next_x
                            if can_occupy(local_area, zombie.x, next_y):
                                zombie.y = next_y
                if lock_on_enabled:
                    target = next((zombie for zombie in zombies
                                   if zombie.zombie_id == locked_zombie_id), None)
                    if target is None or math.hypot(target.x - character.x,
                                                    target.y - character.y) > 32:
                        nearby = [(math.hypot(zombie.x - character.x,
                                              zombie.y - character.y), zombie)
                                  for zombie in zombies]
                        nearby = [entry for entry in nearby if entry[0] <= 32]
                        target = min(nearby, key=lambda entry: entry[0])[1] if nearby else None
                        locked_zombie_id = target.zombie_id if target else None
                    if target is not None:
                        dx, dy = target.x - character.x, target.y - character.y
                        distance = math.hypot(dx, dy) or 1.0
                        character.facing_x = round(dx * 100 / distance)
                        character.facing_y = round(dy * 100 / distance)

        if music_available:
            desired_music = (MENU_MUSIC_PATH if state in {"menu", "settings", "controller_remap", "keyboard_remap"} else
                             FILE_SCREEN_MUSIC_PATH if state == "saves" else None)
            if desired_music != active_music_path:
                if desired_music is None:
                    pygame.mixer.music.stop()
                    active_music_path = None
                else:
                    try:
                        pygame.mixer.music.load(str(desired_music))
                        pygame.mixer.music.set_volume(0.45)
                        pygame.mixer.music.play(-1)
                        active_music_path = desired_music
                    except pygame.error:
                        active_music_path = desired_music

        if state == "menu":
            flash_alpha = round(112 * lightning_remaining / 0.16)
            draw_forest_backdrop(screen, menu_background, menu_tree_layers,
                                 menu_cloud_layers, menu_fog_layers, menu_scene_time,
                                 flash_alpha, lightning_points)
            title_image = title_font.render("PY ZOMBIE GAME", True, PAPER)
            screen.blit(title_image, title_image.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 - 105)))
            tagline = body_font.render("The roads end. The woods don't.", True, MUTED)
            screen.blit(tagline, tagline.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 - 43)))
            draw_button(screen, menu_buttons["continue"], "CONTINUE", body_font,
                        mouse_position, enabled=bool(saves), focused=menu_selected == 0)
            draw_button(screen, menu_buttons["new"], "NEW GAME", body_font, mouse_position,
                        focused=menu_selected == 1)
            draw_button(screen, menu_buttons["settings"], "SETTINGS", body_font, mouse_position,
                        focused=menu_selected == 2)
            draw_button(screen, menu_buttons["quit"], "QUIT", body_font, mouse_position,
                        focused=menu_selected == 3)
            if saves:
                save_text = small_font.render(f"{len(saves)} saved world(s)", True, MUTED)
                screen.blit(save_text, save_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 312)))
            else:
                save_text = small_font.render("No saved worlds yet", True, MUTED)
                screen.blit(save_text, save_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 312)))

        elif state == "settings":
            screen.fill(INK)
            panel = pygame.Rect(SCREEN_WIDTH // 2 - 390, SCREEN_HEIGHT // 2 - 230, 780, 460)
            pygame.draw.rect(screen, PANEL, panel, border_radius=10)
            pygame.draw.rect(screen, PANEL_LIGHT, panel, 2, border_radius=10)
            title_image = title_font.render("SETTINGS", True, PAPER)
            screen.blit(title_image, title_image.get_rect(center=(SCREEN_WIDTH // 2, panel.y + 75)))
            draw_text(screen, body_font, "Configure your controller", MUTED,
                      (panel.centerx - 106, panel.y + 125))
            settings_rows = [
                pygame.Rect(panel.centerx - 230, panel.y + 180, 460, 54),
                pygame.Rect(panel.centerx - 230, panel.y + 250, 460, 54),
                pygame.Rect(panel.centerx - 230, panel.y + 320, 460, 54),
            ]
            draw_button(screen, settings_rows[0], "CONTROLLER REMAPPING", body_font,
                        mouse_position, focused=settings_selected == 0)
            draw_button(screen, settings_rows[1], "KEYBOARD REMAPPING", body_font,
                        mouse_position, focused=settings_selected == 1)
            draw_button(screen, settings_rows[2], "BACK TO MAIN MENU", body_font,
                        mouse_position, focused=settings_selected == 2)
            draw_text(screen, small_font, "Use Up/Down and Enter, or click an option. Esc returns.",
                      MUTED, (panel.centerx - 184, panel.bottom - 55))

        elif state == "controller_remap":
            screen.fill(INK)
            panel_height = min(760, SCREEN_HEIGHT - 40)
            panel = pygame.Rect(SCREEN_WIDTH // 2 - 420,
                                (SCREEN_HEIGHT - panel_height) // 2, 840, panel_height)
            pygame.draw.rect(screen, PANEL, panel, border_radius=10)
            pygame.draw.rect(screen, PANEL_LIGHT, panel, 2, border_radius=10)
            draw_button(screen, back_button, "BACK", small_font, mouse_position)
            title_image = title_font.render("CONTROLLER REMAPPING", True, PAPER)
            screen.blit(title_image, title_image.get_rect(center=(SCREEN_WIDTH // 2, panel.y + 58)))
            for index, action in enumerate(remappable_actions):
                row = remap_row_rect(index)
                selected = index == remap_selected
                pygame.draw.rect(screen, PANEL_LIGHT if selected else PANEL,
                                 row, border_radius=6)
                pygame.draw.rect(screen, PAPER if selected else (93, 101, 91),
                                 row, 2 if selected else 1, border_radius=6)
                draw_text(screen, body_font, controller_action_names[action], PAPER,
                          (row.x + 18, row.y + 12))
                binding_text = ("PRESS A CONTROLLER BUTTON..."
                                if remapping_action == action
                                else controller_button_label(controller_bindings[action]))
                binding_image = body_font.render(binding_text, True,
                                                 (234, 195, 111) if remapping_action == action else MUTED)
                screen.blit(binding_image, binding_image.get_rect(midright=(row.right - 18, row.centery)))
            reset_index = len(remappable_actions)
            reset_row = remap_row_rect(reset_index)
            draw_button(screen, reset_row, "RESET TO DEFAULT CONTROLS", body_font,
                        mouse_position, focused=remap_selected == reset_index)
            status = ("Press any controller button to bind this action. Esc cancels."
                      if remapping_action is not None else
                      "Select a row to remap. The bottom option resets all controls.")
            if controller is None and remapping_action is None:
                status = "No controller detected. Connect one, then choose an action to remap."
            draw_text(screen, small_font, fit_text(small_font, status, panel.width - 40),
                      (234, 195, 111) if remapping_action else MUTED,
                      (panel.x + 22, panel.bottom - 52))

        elif state == "keyboard_remap":
            screen.fill(INK)
            panel_height = min(760, SCREEN_HEIGHT - 40)
            panel = pygame.Rect(SCREEN_WIDTH // 2 - 420,
                                (SCREEN_HEIGHT - panel_height) // 2, 840, panel_height)
            pygame.draw.rect(screen, PANEL, panel, border_radius=10)
            pygame.draw.rect(screen, PANEL_LIGHT, panel, 2, border_radius=10)
            draw_button(screen, back_button, "BACK", small_font, mouse_position)
            title_image = title_font.render("KEYBOARD REMAPPING", True, PAPER)
            screen.blit(title_image, title_image.get_rect(center=(SCREEN_WIDTH // 2, panel.y + 58)))
            for index, action in enumerate(keyboard_remappable_actions):
                row = remap_row_rect(index, keyboard=True)
                selected = index == remap_selected
                pygame.draw.rect(screen, PANEL_LIGHT if selected else PANEL, row, border_radius=6)
                pygame.draw.rect(screen, PAPER if selected else (93, 101, 91), row,
                                 2 if selected else 1, border_radius=6)
                draw_text(screen, body_font, keyboard_action_names[action], PAPER,
                          (row.x + 18, row.y + 12))
                binding_text = ("PRESS A KEY..." if remapping_key_action == action
                                else keyboard_key_label(keyboard_bindings[action]))
                binding_image = body_font.render(binding_text, True,
                                                 (234, 195, 111) if remapping_key_action == action else MUTED)
                screen.blit(binding_image, binding_image.get_rect(midright=(row.right - 18, row.centery)))
            reset_index = len(keyboard_remappable_actions)
            reset_row = remap_row_rect(reset_index, keyboard=True)
            draw_button(screen, reset_row, "RESET TO DEFAULT CONTROLS", body_font,
                        mouse_position, focused=remap_selected == reset_index)
            status = ("Press a key to assign it. Esc cancels listening."
                      if remapping_key_action is not None else
                      "Select a row to remap. The bottom option resets all controls.")
            draw_text(screen, small_font, fit_text(small_font, status, panel.width - 40),
                      (234, 195, 111) if remapping_key_action else MUTED,
                      (panel.x + 22, panel.bottom - 52))

        elif state == "name_entry":
            screen.fill(INK)
            title = "NAME YOUR SAVE" if name_entry_mode == "new" else "RENAME SAVE"
            title_image = title_font.render(title, True, PAPER)
            screen.blit(title_image, title_image.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 - 110)))
            draw_text(screen, body_font, "Type a name, then press Enter.", MUTED,
                      (SCREEN_WIDTH // 2 - 155, SCREEN_HEIGHT // 2 - 48))
            entry_rect = pygame.Rect(SCREEN_WIDTH // 2 - 260, SCREEN_HEIGHT // 2 - 5, 520, 58)
            pygame.draw.rect(screen, PANEL_LIGHT, entry_rect, border_radius=6)
            pygame.draw.rect(screen, ACCENT, entry_rect, width=2, border_radius=6)
            shown_name = fit_text(body_font, name_input, entry_rect.width - 32)
            draw_text(screen, body_font, shown_name, PAPER, (entry_rect.x + 16, entry_rect.y + 18))
            draw_text(screen, small_font, "Enter saves  ·  Esc cancels", MUTED,
                      (SCREEN_WIDTH // 2 - 115, entry_rect.bottom + 20))

        elif state == "delete_confirm":
            screen.fill(INK)
            title_image = title_font.render("DELETE SAVE?", True, PAPER)
            screen.blit(title_image, title_image.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 - 95)))
            if delete_target is not None:
                save_label = game_world.load_save_name(delete_target[1], delete_target[0])
                prompt = f"Permanently delete {save_label}?"
                prompt_image = body_font.render(fit_text(body_font, prompt, SCREEN_WIDTH - 120), True, MUTED)
                screen.blit(prompt_image, prompt_image.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 - 35)))
            draw_button(screen, confirm_yes, "DELETE", body_font, mouse_position)
            draw_button(screen, confirm_no, "CANCEL", body_font, mouse_position)
            draw_text(screen, small_font, "Enter / Y confirms  ·  Esc / N cancels", MUTED,
                      (SCREEN_WIDTH // 2 - 145, confirm_no.bottom + 18))

        elif state == "saves":
            screen.fill(INK)
            draw_button(screen, back_button, "BACK", small_font, mouse_position)
            title_image = title_font.render("SAVED WORLDS", True, PAPER)
            screen.blit(title_image, title_image.get_rect(center=(SCREEN_WIDTH // 2, 105)))
            subtitle = "Up/Down: saves  ·  Left/Right: Load, Rename, Delete  ·  Enter: select"
            subtitle_image = small_font.render(subtitle, True, MUTED)
            screen.blit(subtitle_image, subtitle_image.get_rect(center=(SCREEN_WIDTH // 2, 157)))
            draw_text(screen, small_font, "R: rename  ·  Delete: remove  ·  Esc: back", MUTED,
                      (SCREEN_WIDTH // 2 - 146, 178))
            visible_saves = saves[save_scroll:save_scroll + 7]
            for index, (slot_number, path, seed) in enumerate(visible_saves):
                row = pygame.Rect(170, 205 + index * 66, SCREEN_WIDTH - 340, 54)
                absolute_index = save_scroll + index
                hovered = row.collidepoint(mouse_position)
                row_color = PANEL_LIGHT if hovered or absolute_index == save_selected else PANEL
                pygame.draw.rect(screen, row_color, row, border_radius=6)
                if absolute_index == save_selected:
                    pygame.draw.rect(screen, PAPER, row, width=2, border_radius=6)
                save_name = game_world.load_save_name(path, slot_number)
                draw_text(screen, body_font, fit_text(body_font, save_name, row.width - 300),
                          PAPER, (row.x + 16, row.y + 5))
                draw_text(screen, small_font, f"Slot {slot_number}  ·  seed {seed}", MUTED,
                          (row.x + 16, row.y + 32))
                load_rect = pygame.Rect(row.right - 260, row.y + 7, 78, 40)
                rename_rect = pygame.Rect(row.right - 174, row.y + 7, 78, 40)
                delete_rect = pygame.Rect(row.right - 88, row.y + 7, 78, 40)
                focused = absolute_index == save_selected
                draw_button(screen, load_rect, "LOAD", small_font, mouse_position,
                            focused=focused and save_action_focus == 0)
                draw_button(screen, rename_rect, "RENAME", small_font, mouse_position,
                            focused=focused and save_action_focus == 1)
                draw_button(screen, delete_rect, "DELETE", small_font, mouse_position,
                            focused=focused and save_action_focus == 2)
            if not saves:
                message = body_font.render("No saved worlds found.", True, MUTED)
                screen.blit(message, message.get_rect(center=(SCREEN_WIDTH // 2, 260)))
            if len(saves) > 7:
                draw_text(screen, small_font, f"Showing {save_scroll + 1}-{min(save_scroll + 7, len(saves))} of {len(saves)}", MUTED, (535, 680))

        elif state == "world" and current_world is not None and map_surface is not None:
            screen.fill(INK)
            pygame.draw.rect(screen, PANEL, (24, 20, SCREEN_WIDTH - 48, SCREEN_HEIGHT - 40), border_radius=10)
            draw_text(screen, heading_font, "BLACKPINE REGION", PAPER, (55, 43))
            draw_text(screen, small_font,
                      f"Seed {current_world.seed}  ·  {current_world.width * current_world.cell_size_km} x {current_world.height * current_world.cell_size_km} km region",
                      MUTED, (55, 75))
            draw_button(screen, explore_button, "GO TO LOCATION", small_font, mouse_position,
                        focused=world_selected == 0)
            draw_button(screen, menu_button, "MAIN MENU", small_font, mouse_position,
                        focused=world_selected == 1)

            pygame.draw.rect(screen, (12, 17, 18), map_viewport, border_radius=6)
            scale = min(map_viewport.width / current_world.width,
                        map_viewport.height / current_world.height)
            map_width = int(current_world.width * scale)
            map_height = int(current_world.height * scale)
            map_x = map_viewport.x + (map_viewport.width - map_width) // 2
            map_y = map_viewport.y + (map_viewport.height - map_height) // 2
            scaled_map = pygame.transform.scale(map_surface, (map_width, map_height))
            screen.blit(scaled_map, (map_x, map_y))

            def marker_position(x: int, y: int) -> tuple[int, int]:
                return map_x + int((x + 0.5) * scale), map_y + int((y + 0.5) * scale)

            for farm in current_world.farms:
                pygame.draw.rect(screen, (157, 181, 103), (*marker_position(farm.x, farm.y), 4, 4))
            for location in current_world.points_of_interest:
                pygame.draw.circle(screen, (220, 99, 67), marker_position(location.x, location.y), 3)
            for settlement in current_world.settlements:
                marker_color = (232, 174, 87) if settlement.size == "city" else (237, 232, 216)
                marker_size = 7 if settlement.size == "city" else 5
                pygame.draw.circle(screen, marker_color,
                                   marker_position(settlement.x, settlement.y), marker_size)

            selected_marker = marker_position(region_selection_x, region_selection_y)
            selected_rect = pygame.Rect(
                map_x + int(region_selection_x * scale),
                map_y + int(region_selection_y * scale),
                max(3, round(scale)), max(3, round(scale)))
            pygame.draw.rect(screen, (255, 238, 155), selected_rect, 2)
            pygame.draw.circle(screen, (255, 238, 155), selected_marker, 4)

            side_x = 610
            draw_text(screen, heading_font, "SETTLEMENTS", PAPER, (side_x, 120))
            y = 158
            for settlement in current_world.settlements:
                draw_text(screen, small_font,
                          f"{settlement.name}  ·  {settlement.size}  ·  {settlement.population:,}",
                          MUTED, (side_x, y))
                y += 23
            draw_text(screen, heading_font, "ROADSIDE & WOODLAND SITES", PAPER, (side_x, y + 8))
            y += 46
            for location in current_world.points_of_interest[:13]:
                label = location.name.replace("Abandoned ", "")
                draw_text(screen, small_font, f"• {label}", MUTED, (side_x, y))
                y += 21
            draw_text(screen, small_font,
                      f"{len(current_world.farms)} farms  ·  {len(current_world.points_of_interest)} remote sites",
                      (157, 181, 103), (side_x, 656))
            legend = (
                (TERRAIN_COLORS[game_world.Terrain.FOREST], "Forest"),
                (TERRAIN_COLORS[game_world.Terrain.PLAINS], "Plains"),
                (TERRAIN_COLORS[game_world.Terrain.WATER], "Water"),
                ((176, 153, 112), "Road"), ((157, 181, 103), "Farm"),
                ((232, 174, 87), "City"), ((237, 232, 216), "Town"),
                ((220, 99, 67), "Site"),
            )
            for index, (color, label) in enumerate(legend):
                x = side_x + index * 75
                pygame.draw.rect(screen, color, (x, 685, 10, 10))
                draw_text(screen, small_font, label, MUTED, (x + 13, 682))
            selection_tile = current_world.tile_at(region_selection_x, region_selection_y)
            site = next((item.name for item in current_world.points_of_interest
                         if (item.x, item.y) == (region_selection_x, region_selection_y)), None)
            settlement = next((item.name for item in current_world.settlements
                               if (item.x, item.y) == (region_selection_x, region_selection_y)), None)
            farm = next((item.name for item in current_world.farms
                         if (item.x, item.y) == (region_selection_x, region_selection_y)), None)
            destination = site or settlement or farm or selection_tile.terrain.value.title()
            selection_label = (
                f"Destination: {destination}  ·  grid {region_selection_x}, {region_selection_y}  ·  "
                f"{math.hypot(region_selection_x - saved_region_x, region_selection_y - saved_region_y) * current_world.cell_size_km:.0f} km from last stop")
            draw_text(screen, small_font,
                      fit_text(small_font, selection_label, SCREEN_WIDTH - side_x - 36),
                      (255, 238, 155), (side_x, 710))
            draw_text(screen, small_font,
                      "Arrows/WASD: pick region  ·  Enter: explore  ·  Tab: main menu  ·  Esc: back",
                      MUTED, (side_x, 738))

            if map_x <= mouse_position[0] < map_x + map_width and map_y <= mouse_position[1] < map_y + map_height:
                tile_x = min(current_world.width - 1, int((mouse_position[0] - map_x) / scale))
                tile_y = min(current_world.height - 1, int((mouse_position[1] - map_y) / scale))
                tile = current_world.tile_at(tile_x, tile_y)
                pygame.draw.rect(screen, PANEL_LIGHT, (40, 754, 520, 25), border_radius=4)
                draw_text(screen, small_font,
                          f"{tile.terrain.value.title()} · grid {tile_x}, {tile_y} · {current_world.cell_size_km} km cell",
                          PAPER, (52, 758))

        elif state == "local" and current_world is not None and local_area is not None and local_surface is not None and character is not None:
            # The exploration view uses every display pixel; the minimap floats over it.
            if current_interior is not None and current_interior_surface is not None:
                screen.fill((28, 27, 24))
                tile_size = current_interior_surface.get_width() // current_interior.width
                interior_x = (SCREEN_WIDTH - current_interior_surface.get_width()) // 2
                interior_y = (SCREEN_HEIGHT - current_interior_surface.get_height()) // 2
                screen.blit(current_interior_surface, (interior_x, interior_y))
                player_center = (round(interior_x + character.x * tile_size + tile_size // 2),
                                 round(interior_y + character.y * tile_size + tile_size // 2))
                draw_character(screen, player_center, character.flashlight_on,
                               (character.facing_x, character.facing_y), tile_size,
                               character.equipped_weapon)
                outside_x = character.outside_x if character.outside_x is not None else local_area.spawn_x
                outside_y = character.outside_y if character.outside_y is not None else local_area.spawn_y
                camera_x = round(outside_x * LOCAL_TILE_SIZE + LOCAL_TILE_SIZE // 2 - local_viewport.width // 2)
                camera_y = round(outside_y * LOCAL_TILE_SIZE + LOCAL_TILE_SIZE // 2 - local_viewport.height // 2)
                camera_x = max(0, min(camera_x, local_surface.get_width() - local_viewport.width))
                camera_y = max(0, min(camera_y, local_surface.get_height() - local_viewport.height))
            else:
                camera_x = max(0, min(round(character.x * LOCAL_TILE_SIZE + LOCAL_TILE_SIZE // 2
                                             - local_viewport.width // 2),
                                      local_surface.get_width() - local_viewport.width))
                camera_y = max(0, min(round(character.y * LOCAL_TILE_SIZE + LOCAL_TILE_SIZE // 2
                                             - local_viewport.height // 2),
                                      local_surface.get_height() - local_viewport.height))
                camera_rect = pygame.Rect(camera_x, camera_y, local_viewport.width, local_viewport.height)
                screen.blit(local_surface, local_viewport.topleft, camera_rect)
                for zombie in zombies:
                    zombie_center = (round(zombie.x * LOCAL_TILE_SIZE + LOCAL_TILE_SIZE // 2 - camera_x),
                                     round(zombie.y * LOCAL_TILE_SIZE + LOCAL_TILE_SIZE // 2 - camera_y))
                    if local_viewport.collidepoint(zombie_center):
                        draw_zombie(screen, zombie_center, health=zombie.health)
                        if lock_on_enabled and zombie.zombie_id == locked_zombie_id:
                            pygame.draw.circle(screen, (248, 195, 95), zombie_center,
                                               LOCAL_TILE_SIZE * 2, 2)
                for popup in damage_popups:
                    popup_x = round(popup.x * LOCAL_TILE_SIZE + LOCAL_TILE_SIZE // 2 - camera_x)
                    popup_y = round(popup.y * LOCAL_TILE_SIZE + LOCAL_TILE_SIZE // 2 - camera_y
                                    - (0.9 - popup.timer) * 26)
                    if local_viewport.collidepoint((popup_x, popup_y)):
                        alpha = min(255, round(255 * min(1.0, popup.timer / 0.25)))
                        label = small_font.render(popup.text, True, (255, 229, 137))
                        shadow = small_font.render(popup.text, True, (27, 21, 18))
                        label.set_alpha(alpha)
                        shadow.set_alpha(alpha)
                        screen.blit(shadow, shadow.get_rect(center=(popup_x + 1, popup_y + 1)))
                        screen.blit(label, label.get_rect(center=(popup_x, popup_y)))
                player_center = (
                    round(character.x * LOCAL_TILE_SIZE + LOCAL_TILE_SIZE // 2 - camera_x),
                    round(character.y * LOCAL_TILE_SIZE + LOCAL_TILE_SIZE // 2 - camera_y),
                )
                draw_character(screen, player_center, character.flashlight_on,
                               (character.facing_x, character.facing_y),
                               weapon=character.equipped_weapon)

                # Shift the wilderness palette through dusk, night, and dawn.
                hour = (character.game_time_minutes / 60) % 24
                if 16 <= hour < 18:
                    darkness = round(150 * (hour - 16) / 2)
                elif 6 <= hour < 8:
                    darkness = round(150 * (8 - hour) / 2)
                elif hour >= 18 or hour < 6:
                    darkness = 150
                else:
                    darkness = 0
                if darkness:
                    night_layer = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
                    night_layer.fill((5, 9, 24, darkness))
                    if character.flashlight_on:
                        # Leave a forward-facing corridor visible in the beam.
                        direction = pygame.Vector2(character.facing_x, character.facing_y)
                        if direction.length_squared() == 0:
                            direction.update(0, -1)
                        direction = direction.normalize()
                        side = pygame.Vector2(-direction.y, direction.x)
                        origin = pygame.Vector2(player_center)
                        tip = origin + direction * 250
                        beam_points = [origin + side * 15, tip + side * 92,
                                       tip - side * 92, origin - side * 15]
                        pygame.draw.polygon(night_layer, (5, 9, 24, 0),
                                            [(round(point.x), round(point.y)) for point in beam_points])
                    screen.blit(night_layer, (0, 0))

            scene_pickups = interior_pickups if current_interior is not None else outdoor_pickups
            nearby_pickup = None
            nearby_distance = float("inf")
            for pickup in scene_pickups:
                if current_interior is not None:
                    pickup_center = (round(interior_x + pickup.x * tile_size + tile_size // 2),
                                     round(interior_y + pickup.y * tile_size + tile_size // 2))
                    icon_scale = tile_size
                else:
                    pickup_center = (round(pickup.x * LOCAL_TILE_SIZE + LOCAL_TILE_SIZE // 2 - camera_x),
                                     round(pickup.y * LOCAL_TILE_SIZE + LOCAL_TILE_SIZE // 2 - camera_y))
                    icon_scale = LOCAL_TILE_SIZE
                if local_viewport.collidepoint(pickup_center):
                    draw_pickup(screen, pickup_center, pickup.item, icon_scale)
                distance = math.hypot(pickup.x - character.x, pickup.y - character.y)
                if distance < nearby_distance:
                    nearby_pickup, nearby_distance = pickup, distance

            # Small in-world clock; the time is part of this save slot.
            time_panel = pygame.Rect(24, 24, 320, 68)
            pygame.draw.rect(screen, (15, 21, 22, 205), time_panel, border_radius=5)
            pygame.draw.rect(screen, (140, 149, 136), time_panel, width=1, border_radius=5)
            total_minutes = int(character.game_time_minutes) % (24 * 60)
            hour24, minute = divmod(total_minutes, 60)
            display_hour = hour24 % 12 or 12
            period = "AM" if hour24 < 12 else "PM"
            location_status = (f"DAY {int(character.game_time_minutes // 1440) + 1:02d}  ·  "
                               f"GRID {character.region_x}, {character.region_y}")
            draw_text(screen, small_font,
                      fit_text(small_font, location_status, time_panel.width - 24),
                      MUTED, (time_panel.x + 12, time_panel.y + 7))
            draw_text(screen, body_font, f"{display_hour:02d}:{minute:02d} {period}",
                      PAPER, (time_panel.x + 12, time_panel.y + 22))
            draw_text(screen, small_font,
                      fit_text(small_font, local_area.name, time_panel.width - 24),
                      (190, 198, 179), (time_panel.x + 112, time_panel.y + 28))
            if current_interior is None:
                infected_label = f"INFECTED: {len(zombies)}"
                draw_text(screen, small_font, infected_label,
                          (224, 116, 86) if zombies else MUTED,
                          (time_panel.x + 12, time_panel.bottom - 20))

            if current_interior is None:
                mini_width = min(300, max(220, int(SCREEN_WIDTH * 0.22)))
                mini_height = int(mini_width * 2 / 3)
                mini_padding = 9
                mini_rect = pygame.Rect(SCREEN_WIDTH - mini_width - 24, 24,
                                       mini_width, mini_height)
                mini_panel = mini_rect.inflate(mini_padding * 2, mini_padding * 2)
                panel_surface = pygame.Surface(mini_panel.size, pygame.SRCALPHA)
                panel_surface.fill((15, 21, 22, 205))
                screen.blit(panel_surface, mini_panel.topleft)
                if local_minimap_surface is not None:
                    if local_minimap_surface.get_size() != mini_rect.size:
                        local_minimap_surface = pygame.transform.smoothscale(local_surface, mini_rect.size)
                    screen.blit(local_minimap_surface, mini_rect.topleft)
                    view_x = camera_x / local_surface.get_width() * mini_rect.width
                    view_y = camera_y / local_surface.get_height() * mini_rect.height
                    view_w = local_viewport.width / local_surface.get_width() * mini_rect.width
                    view_h = local_viewport.height / local_surface.get_height() * mini_rect.height
                    pygame.draw.rect(screen, (227, 223, 207),
                                     (mini_rect.x + round(view_x), mini_rect.y + round(view_y),
                                      max(1, round(view_w)), max(1, round(view_h))), 1)
                    pygame.draw.rect(screen, (140, 149, 136), mini_rect, 1)

        if state == "local" and inventory_open and character is not None:
            shade = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
            shade.fill((0, 0, 0, 215))
            screen.blit(shade, (0, 0))
            panel_width = min(1120, SCREEN_WIDTH - 52)
            panel_height = min(780, SCREEN_HEIGHT - 44)
            panel = pygame.Rect((SCREEN_WIDTH - panel_width) // 2,
                                (SCREEN_HEIGHT - panel_height) // 2,
                                panel_width, panel_height)
            metal = (87, 84, 61)
            metal_light = (142, 132, 91)
            metal_dark = (37, 40, 37)
            blue = (8, 20, 78)
            blue_edge = (65, 83, 151)
            pygame.draw.rect(screen, metal_dark, panel)
            pygame.draw.rect(screen, metal_light, panel, width=3)
            inner_panel = panel.inflate(-12, -12)
            pygame.draw.rect(screen, metal, inner_panel, width=5)
            pygame.draw.rect(screen, (13, 16, 18), inner_panel.inflate(-10, -10))

            # Ribbed console header with tab-like labels.
            header = pygame.Rect(panel.x + 20, panel.y + 18, panel.width - 40, 48)
            pygame.draw.rect(screen, (24, 28, 29), header)
            pygame.draw.line(screen, metal_light, header.topleft, header.topright, 2)
            pygame.draw.line(screen, metal_dark, header.bottomleft, header.bottomright, 3)
            for x in range(header.x + 8, header.right - 8, 14):
                pygame.draw.line(screen, (39, 43, 42), (x, header.y + 4),
                                 (x, header.bottom - 4), 1)
            tab_x = header.x + 16
            for label, selected in (("STATUS", False), ("EQUIPMENT", False), ("ITEM", True)):
                tab_width = max(88, small_font.size(label)[0] + 24)
                tab = pygame.Rect(tab_x, header.y + 8, tab_width, header.height - 12)
                pygame.draw.rect(screen, blue if selected else metal_dark, tab)
                pygame.draw.rect(screen, blue_edge if selected else metal_light, tab, 1)
                tab_image = small_font.render(label, True, PAPER if selected else MUTED)
                screen.blit(tab_image, tab_image.get_rect(center=tab.center))
                tab_x = tab.right + 8
            title_image = heading_font.render("SURVIVAL INVENTORY", True, PAPER)
            screen.blit(title_image, title_image.get_rect(midright=(header.right - 14, header.centery)))

            content_y = header.bottom + 14
            content_height = int(panel.height * 0.56)
            left_width = int((panel.width - 58) * 0.41)
            left_rect = pygame.Rect(panel.x + 24, content_y, left_width, content_height)
            right_rect = pygame.Rect(left_rect.right + 14, content_y,
                                     panel.right - left_rect.right - 38, content_height)

            # Status card and compact portrait, inspired by the reference's CRT panel.
            pygame.draw.rect(screen, metal, left_rect)
            pygame.draw.rect(screen, metal_light, left_rect, 2)
            status_head = pygame.Rect(left_rect.x + 8, left_rect.y + 8, left_rect.width - 16, 28)
            pygame.draw.rect(screen, blue, status_head)
            draw_text(screen, small_font, "SURVIVOR STATUS", PAPER,
                      (status_head.x + 8, status_head.y + 6))
            portrait = pygame.Rect(left_rect.x + 14, status_head.bottom + 12, 102, 118)
            pygame.draw.rect(screen, (6, 15, 63), portrait)
            pygame.draw.rect(screen, blue_edge, portrait, 2)
            # Original low-resolution face silhouette, drawn from simple shapes.
            pygame.draw.ellipse(screen, (61, 42, 34), (portrait.x + 22, portrait.y + 13, 58, 77))
            pygame.draw.ellipse(screen, (190, 145, 111), (portrait.x + 28, portrait.y + 25, 47, 58))
            pygame.draw.polygon(screen, (53, 39, 35), (
                (portrait.x + 27, portrait.y + 45), (portrait.x + 30, portrait.y + 22),
                (portrait.x + 51, portrait.y + 11), (portrait.x + 74, portrait.y + 28),
                (portrait.x + 71, portrait.y + 42), (portrait.x + 62, portrait.y + 29),
                (portrait.x + 44, portrait.y + 32)))
            pygame.draw.line(screen, (60, 48, 43), (portrait.x + 42, portrait.y + 51),
                             (portrait.x + 47, portrait.y + 51), 2)
            pygame.draw.line(screen, (60, 48, 43), (portrait.x + 60, portrait.y + 51),
                             (portrait.x + 65, portrait.y + 51), 2)
            pygame.draw.rect(screen, (72, 84, 73),
                             (portrait.x + 24, portrait.y + 83, 54, 30))

            stats_x = portrait.right + 14
            stats_width = left_rect.right - stats_x - 14
            draw_text(screen, small_font, character.name.upper(), PAPER,
                      (stats_x, portrait.y + 3))
            draw_text(screen, small_font, "CONDITION", MUTED, (stats_x, portrait.y + 29))
            draw_text(screen, body_font, f"{character.health:03d} / 100", PAPER,
                      (stats_x, portrait.y + 48))
            health_bar = pygame.Rect(stats_x, portrait.y + 77, stats_width, 14)
            pygame.draw.rect(screen, (36, 39, 37), health_bar)
            if character.health:
                fill_width = int(health_bar.width * character.health / 100)
                pygame.draw.rect(screen, (160, 86, 68),
                                 (health_bar.x, health_bar.y, fill_width, health_bar.height))
            light_status = "ON" if character.flashlight_on else "OFF"
            draw_text(screen, small_font, f"LIGHT  {light_status}",
                      (205, 190, 120), (stats_x, portrait.y + 98))

            graph = pygame.Rect(left_rect.x + 14, portrait.bottom + 12,
                                left_rect.width - 28, max(54, int(content_height * 0.23)))
            pygame.draw.rect(screen, (4, 15, 29), graph)
            pygame.draw.rect(screen, blue_edge, graph, 1)
            for grid_y in range(graph.y + 12, graph.bottom, 14):
                pygame.draw.line(screen, (31, 49, 58), (graph.x + 2, grid_y),
                                 (graph.right - 2, grid_y), 1)
            waveform = []
            for step in range(24):
                px = graph.x + 5 + step * (graph.width - 10) / 23
                pulse = (step * 7 + character.health // 10) % 13
                py = graph.centery - (12 if pulse in (4, 5) else 4 if pulse in (3, 6) else 0)
                waveform.append((round(px), round(py)))
            pygame.draw.lines(screen, (66, 176, 118), False, waveform, 2)
            draw_text(screen, small_font, "VITALS", (89, 183, 137),
                      (graph.right - 55, graph.bottom - 20))

            equipment_y = graph.bottom + 12
            equipment = pygame.Rect(left_rect.x + 14, equipment_y,
                                    left_rect.width - 28, max(48, left_rect.bottom - equipment_y - 12))
            pygame.draw.rect(screen, (54, 57, 44), equipment)
            pygame.draw.rect(screen, metal_light, equipment, 1)
            draw_text(screen, small_font, "EQUIPPED", PAPER,
                      (equipment.x + 9, equipment.y + 7))
            if character.equipped_weapon in {"Kitchen Knife", "Handgun"}:
                icon_rect = pygame.Rect(equipment.x + 10, equipment.y + 25, 42, 36)
                draw_inventory_icon(screen, icon_rect, character.equipped_weapon)
                weapon_label = character.equipped_weapon
                if character.equipped_weapon == "Handgun":
                    weapon_label += (f" · {character.handgun_loaded}/6 loaded"
                                     f" · {character.ammo_9mm} spare")
                draw_text(screen, small_font,
                          fit_text(small_font, weapon_label, equipment.width - 76), PAPER,
                          (icon_rect.right + 8, icon_rect.y + 10))
            if character.inventory.get("Flashlight", 0):
                light_y = equipment.y + 68
                icon_rect = pygame.Rect(equipment.x + 10, equipment.y + 25, 42, 36)
                icon_rect.y = min(light_y, equipment.bottom - 38)
                draw_inventory_icon(screen, icon_rect, "Flashlight")
                draw_text(screen, small_font, "Flashlight", PAPER,
                          (icon_rect.right + 10, icon_rect.y + 10))
                draw_text(screen, small_font, light_status, (205, 190, 120),
                          (equipment.right - 42, icon_rect.y + 10))

            # Deep-blue item list with selectable slots.
            pygame.draw.rect(screen, metal, right_rect)
            pygame.draw.rect(screen, metal_light, right_rect, 2)
            list_head = pygame.Rect(right_rect.x + 8, right_rect.y + 8,
                                    right_rect.width - 16, 30)
            pygame.draw.rect(screen, blue, list_head)
            inventory_items = inventory_items_for(character)
            inventory_total = (sum(character.inventory.values()) + character.ammo_9mm
                               + character.handgun_loaded)
            draw_text(screen, small_font, f"ITEM LIST  ·  {inventory_total} ITEMS",
                      PAPER, (list_head.x + 9, list_head.y + 8))
            list_area = pygame.Rect(right_rect.x + 8, list_head.bottom + 8,
                                    right_rect.width - 16, right_rect.bottom - list_head.bottom - 16)
            pygame.draw.rect(screen, blue, list_area)
            row_height = 74
            visible_rows = max(1, list_area.height // row_height)
            scroll_first = max(0, min(inventory_selected - visible_rows + 1,
                                      len(inventory_items) - visible_rows))
            if not inventory_items:
                draw_text(screen, body_font, "NO ITEMS", PAPER,
                          (list_area.x + 18, list_area.y + 18))
            for index in range(scroll_first, min(len(inventory_items), scroll_first + visible_rows)):
                item = inventory_items[index]
                row = pygame.Rect(list_area.x + 7,
                                  list_area.y + 7 + (index - scroll_first) * row_height,
                                  list_area.width - 14, row_height - 6)
                pygame.draw.rect(screen, (7, 17, 65), row)
                if index == inventory_selected:
                    pygame.draw.rect(screen, (225, 194, 103), row, 2)
                else:
                    pygame.draw.rect(screen, blue_edge, row, 1)
                icon_rect = pygame.Rect(row.x + 8, row.y + 7, 56, 54)
                draw_inventory_icon(screen, icon_rect, item)
                draw_text(screen, body_font, fit_text(body_font, item, row.width - 160),
                          PAPER, (icon_rect.right + 12, row.y + 10))
                quantity_text = (f"SPARE  {character.ammo_9mm:02d}"
                                 if item == "9mm Ammo" else
                                 f"LOADED  {character.handgun_loaded}/6"
                                 if item == "Handgun" else
                                 f"QTY  {character.inventory[item]:02d}")
                draw_text(screen, small_font, quantity_text, MUTED,
                          (icon_rect.right + 12, row.y + 39))
                if item == "Flashlight" and character.flashlight_on:
                    draw_text(screen, small_font, "ON", (221, 196, 115),
                              (row.right - 42, row.y + 20))

            # Selection details occupy their own lower console window.
            detail_y = right_rect.bottom + 12
            detail_rect = pygame.Rect(panel.x + 24, detail_y,
                                      panel.width - 48, panel.bottom - detail_y - 56)
            pygame.draw.rect(screen, (57, 55, 41), detail_rect)
            pygame.draw.rect(screen, metal_light, detail_rect, 2)
            detail_inner = detail_rect.inflate(-10, -10)
            pygame.draw.rect(screen, (5, 13, 46), detail_inner)
            if inventory_items:
                inventory_selected %= len(inventory_items)
                chosen_item = inventory_items[inventory_selected]
                description = ITEM_DESCRIPTIONS.get(chosen_item, "A carried item.")
                draw_text(screen, heading_font, chosen_item.upper(), PAPER,
                          (detail_inner.x + 14, detail_inner.y + 10))
                draw_text(screen, body_font, description, MUTED,
                          (detail_inner.x + 14, detail_inner.y + 46))
                action = ("ENTER  ·  TOGGLE LIGHT" if chosen_item == "Flashlight"
                          else "ENTER  ·  EQUIP" if chosen_item in {"Handgun", "Kitchen Knife"}
                          else "AMMUNITION" if chosen_item == "9mm Ammo"
                          else "ENTER  ·  USE ITEM")
                draw_text(screen, small_font, action, (211, 187, 112),
                          (detail_inner.right - 190, detail_inner.y + 15))
            else:
                draw_text(screen, body_font, "Nothing selected.", MUTED,
                          (detail_inner.x + 14, detail_inner.y + 18))

            draw_text(screen, small_font,
                      "UP/DOWN  SELECT     ENTER  USE/EQUIP     F  LIGHT     I / ESC  CLOSE",
                      MUTED, (panel.x + 26, panel.bottom - 31))

        if state == "local" and character is not None and current_interior is None and not inventory_open:
            combat_panel = pygame.Rect(24, SCREEN_HEIGHT - 72,
                                       min(900, SCREEN_WIDTH - 48), 46)
            pygame.draw.rect(screen, (15, 21, 22), combat_panel, border_radius=5)
            pygame.draw.rect(screen, (140, 149, 136), combat_panel, 1, border_radius=5)
            sprint_label = "SPRINT ON" if sprint_toggled else "TAB/L3 SPRINT"
            combat_text = (f"GUN {character.handgun_loaded}/6 +{character.ammo_9mm}  ·  SPACE/A FIRE  ·  R/X RELOAD  ·  Q/LB LOCK  ·  F/Y LIGHT  ·  {sprint_label}"
                           if character.equipped_weapon == "Handgun"
                           else f"KNIFE  ·  SPACE/A ATTACK  ·  Q/LB LOCK  ·  F/Y LIGHT  ·  {sprint_label}")
            draw_text(screen, small_font, fit_text(small_font, combat_text, combat_panel.width - 22), PAPER,
                      (combat_panel.x + 12, combat_panel.y + 15))
            if reload_feedback_timer > 0 and reload_feedback:
                feedback_image = small_font.render(reload_feedback, True, (235, 193, 116))
                screen.blit(feedback_image,
                            feedback_image.get_rect(midbottom=(SCREEN_WIDTH // 2,
                                                               SCREEN_HEIGHT - 84)))

        if (state == "local" and character is not None and not inventory_open
                and current_world is not None and local_area is not None
                and nearby_pickup is not None and nearby_distance <= 1.6):
            pickup_name = (f"{nearby_pickup.quantity} 9MM ROUNDS"
                           if nearby_pickup.item == "9mm Ammo"
                           else nearby_pickup.item.upper())
            prompt = f"E  ·  PICK UP {pickup_name}"
            prompt_rect = pygame.Rect(0, SCREEN_HEIGHT - 126, min(520, SCREEN_WIDTH - 40), 40)
            prompt_rect.centerx = SCREEN_WIDTH // 2
            pygame.draw.rect(screen, (15, 21, 22), prompt_rect, border_radius=5)
            pygame.draw.rect(screen, (220, 190, 101), prompt_rect, 1, border_radius=5)
            prompt_image = small_font.render(fit_text(small_font, prompt, prompt_rect.width - 24),
                                             True, PAPER)
            screen.blit(prompt_image, prompt_image.get_rect(center=prompt_rect.center))

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()


if __name__ == "__main__":
    run_game()
