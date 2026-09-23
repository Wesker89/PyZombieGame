# PyZombieGame

A Python RPG project. The first layer is a data-only world generator, so game
rules and world data can be built before choosing how the game is drawn.

The procedural overview map is 120 by 240 cells, with each cell representing
about 5 km. This gives it a province-scale footprint of roughly 600 by 1,200
km. It is an abstract rectangular region rather than an exact Alberta outline;
later, cities can open into detailed local maps.

## Run the windowed game prototype

Activate the project environment, then run:

```bash
source .venv/bin/activate
pip install -r requirements.txt
python game.py
```

`game.py` opens in fullscreen with a main menu. Continue opens the saved-world
list; New Game asks for a save name before creating the slot; Settings offers
controller and keyboard remapping; Quit closes the window. Each remapping page
has a Reset to Default Controls option at the bottom. In the map view,
the overview displays terrain, roads, settlements, farms, and remote sites.
The main menu has a slowly scrolling forest with drifting fog, storm clouds, and
occasional random lightning flashes.
`music/MainMenu.mp3` loops on the main menu; `music/SafePlace.mp3` loops on the
saved-world file screen. Music stops on other screens.
Use **Up/Down** and **Enter** to navigate the main menu, or click an option.
In the saved-world list, use **Up/Down** to choose a save, **Left/Right** to
choose Load, Rename, or Delete, and **Enter** to activate; **R** and **Delete**
are shortcuts. On the region overview, use **Left/Right** and **Enter** to
choose Explore Area or Main Menu. On the region map, click a grid cell or use
the arrow keys to choose where to go; press **Enter** to explore it. The
selection shows the terrain or named destination and distance from your last
stop. Each stop creates a different seeded forest-road area, and travel advances
the in-game clock. Press **Tab** to focus Main Menu, or **Esc** to go back.

Choose **Go to Location** to control the survivor in a 384 by 256 m forest-road scene.
Move smoothly with **WASD** or the **arrow keys**; combine keys for diagonal
movement. Hold **Shift** to run while held, or press **Tab** to toggle sprint.
Click the controller's left stick (**L3**) to toggle sprint. Trees, buildings,
and fuel pumps block movement. The camera follows
the survivor and scrolls across the larger area. Follow the main roads to the
edge of a scene to move seamlessly into the neighboring chunk; your grid
coordinates update as you travel. Chunks now use the regional terrain and
generate varied roadside layouts; named map sites appear when you travel to them.
Farm locations can generate a farmhouse and extra outbuildings, while sheds
appear across rural chunks. You can enter both and search their procedural interiors.
the main road is 16 m wide, with roadside buildings and lots scaled to match.
Exploration fills the screen. A compact map overlay in the upper-right shows
the region and outlines the area currently visible as the camera follows you.
The outdoor lighting shifts through dawn, day, dusk, and night. A clock shows
the current in-game day and time; a full day takes about twelve minutes of play.
Your time is saved with each world, and the flashlight lights a path after dark.
Press **E** beside a building door to enter; press **E** at the interior doorway
to leave. Gas stations, motels, diners, and cabins each generate a different
seeded floor plan and furnishings. Scavenge outdoors and inside for supplies:
gas stations tend to have ammunition, motels and diners medical supplies, and
sheds, farms, and cabins practical survival gear. Loot is tied to each area's
seed and stays collected in that save. Walk up to a glowing item and press **E**
to pick it up.
Press **I** to open the inventory. Select an item with **Up/Down** and press
**Enter** to use a first aid kit or toggle the flashlight. Press **F** anytime
while exploring to toggle the flashlight directly; its beam follows your
movement direction. The survivor starts with a kitchen knife and
a handgun with 18 rounds: six loaded and twelve spare. Select a weapon and
press **Enter** to equip it; press **Space** outdoors to attack in the direction
you face, **R** to reload, and **Q** to lock onto the nearest zombie. With lock-on
active, aim tracks that zombie until it is defeated or moves out of range.
Open **Settings → Controller Remapping** from the main menu, select an action,
then press the controller button to bind it. **Keyboard Remapping** works the
same way: select an action and press the key you want. Both screens have a
**Reset to Default Controls** option at the bottom, and mappings are saved locally.
Controllers are supported: left stick moves, right stick aims, **A** attacks,
**B** interacts, **X** reloads, **Y** toggles the flashlight, **LB** locks on,
**Start** opens the inventory, and **L3** toggles sprint by default. The D-pad
navigates menus. Zombies spawn in areas connected to the survivor by walkable
ground. They notice and pursue the survivor when they have a clear view, finding
routes around obstacles, and attack at close range. Their outline,
eyes, and health bars are visible in daylight and after dark; the status panel
shows how many are nearby. Defeated zombies stay gone in that save, and weapon
choice and remaining ammunition are saved too.
The character's position, health, inventory, and in-game time are stored in that
save slot; position is saved automatically as you walk.

The original terminal menu is still available with `python world.py`.

Each file in `saves/` stores one named world's seed and the survivor's current
state.
Name a world when creating it, or use **Rename** and **Delete** in the saved-world
list. Deleting asks for confirmation. Continuing rebuilds the same procedural
world, and quitting leaves all other saves in place. The earlier `savegame.json`
save is automatically copied into Save Slot 1 the first time the menu runs; the
original file is kept as a backup.
Running `python world.py` also refreshes `world_preview.svg`, which you can
open in VS Code or a browser.

World generation uses dense forest as the dominant wilderness terrain around a
central city and smaller communities. It scatters farms and survival-horror
locations along roads and deeper in the woods, including abandoned gas
stations, motels, diners, ranger stations, lumber mills, cabins, roadblocks,
and remote research outposts. Hover over markers in the SVG preview for their
names and types.

For a quick direct start, `python world.py --new-game` creates another random
save, or `python world.py --seed 42` creates a save using that exact seed.
