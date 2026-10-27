"""
S2R CARLA -> nuScenes LiDAR Dataset Collector
"""

import carla
import os
import sys
import time
import queue
import math
import random
import signal
import logging
import datetime
import numpy as np
import json
import glob
import argparse
import faulthandler
import threading

# CONSTANTS & CONFIGURATION
# Scans per town
TOWN_PLAN = [
    ("Town10HD_Opt", 70_000),
    ("Town03", 70_000),
    ("Town05", 40_000),
    ("Town02", 20_000),
]

TOTAL_SCANS = sum(q for _, q in TOWN_PLAN)  # 200,000

# LiDAR (Velodyne HDL32E)
# Spec: 1080 pts/ring × 32 ch = 34,560 pts/scan.  1.39M spec
# counts dual returns; CARLA ray-cast has no dual-return (1 ray = 1 point),
# Therefore, we set points_per_second to 691,200 to achieve ~34,560 pts/scan at 20 Hz.

LIDAR_CFG = {
    "channels": "32",
    "points_per_second": "691200",
    "rotation_frequency": "20",
    "upper_fov": "10.67",
    "lower_fov": "-30.67",
    "range": "100.0",
    "sensor_tick": "0.0",  # fire every sim frame
    "dropoff_general_rate": "0.10",  # 10% random beam dropoff
    "dropoff_intensity_limit": "0.0",
    "dropoff_zero_intensity": "0.05",  # 5% extra dropout for zero-intensity returns
    "atmosphere_attenuation_rate": "0.0",  # handled by weather presets
    "noise_stddev": "0.02",  # Gaussian range noise in metres
}
LIDAR_POS = carla.Location(x=0.0, y=0.0, z=1.84)

# structural planes under different point-density distributions.
WEATHER_PRESETS = [
    ("ClearNoon", carla.WeatherParameters.ClearNoon),
    ("HardRainNoon", carla.WeatherParameters.HardRainNoon),
    ("CloudyNoon", carla.WeatherParameters.CloudyNoon),
]

# Simulation
FIXED_DELTA = 0.05  # 20 Hz
PRE_ROLL_TICKS = 100  # stabilise physics after map load
TM_PORT = 8000

# Density management
DENSITY_MIN = 40
DENSITY_MAX = 50
DENSITY_TARGET = 45
DENSITY_CHECK_EVERY = 20  # ticks between density audits
DESPAWN_RADIUS = 120.0  # m - cull NPCs farther than this
DENSITY_RADIUS = 70.0  # m - measure density inside this

# Prop spawning
PROP_COUNT_MIN = 15
PROP_COUNT_MAX = 25
PROP_SPAWN_INNER = 10.0  # m from ego (min)
PROP_SPAWN_OUTER = 60.0  # m from ego (max)

# Scene-based collection
SCENE_LENGTH_MIN = 5000  # min scans per scene before ego re-spawn
SCENE_LENGTH_MAX = 10000  # max scans per scene before ego re-spawn

# Density profiles
DENSITY_PROFILES = [
    # few vehicles, minimal pedestrians
    {
        "density_min": 8,
        "density_max": 15,
        "density_target": 12,
        "walkers": 30,
        "prop_min": 3,
        "prop_max": 8,
    },
    # Light traffic
    {
        "density_min": 15,
        "density_max": 25,
        "density_target": 20,
        "walkers": 60,
        "prop_min": 5,
        "prop_max": 12,
    },
    # Medium traffic - urban
    {
        "density_min": 25,
        "density_max": 38,
        "density_target": 32,
        "walkers": 100,
        "prop_min": 10,
        "prop_max": 18,
    },
    # Dense urban - heavy traffic, many pedestrians
    {
        "density_min": 38,
        "density_max": 55,
        "density_target": 48,
        "walkers": 150,
        "prop_min": 15,
        "prop_max": 25,
    },
    # Congested - rush hour
    {
        "density_min": 50,
        "density_max": 70,
        "density_target": 60,
        "walkers": 180,
        "prop_min": 18,
        "prop_max": 30,
    },
]

# Logging / progress
LOG_EVERY = 1000  # scans between progress messages

# nuScenes CLASS MAPPING & RANGE FILTERS
# Maps CARLA blueprint-id *prefixes* -> nuScenes detection class.
# Order matters: first match wins.
_CAR_MAKES = [
    "vehicle.lincoln",
    "vehicle.audi",
    "vehicle.bmw",
    "vehicle.chevrolet",
    "vehicle.citroen",
    "vehicle.dodge",
    "vehicle.ford",
    "vehicle.jeep",
    "vehicle.mercedes",
    "vehicle.micro",
    "vehicle.mini",
    "vehicle.nissan",
    "vehicle.seat",
    "vehicle.tesla",
    "vehicle.toyota",
    "vehicle.volkswagen",
]

BLUEPRINT_TO_NUSCENES = {}
for _m in _CAR_MAKES:
    BLUEPRINT_TO_NUSCENES[_m] = "vehicle.car"

BLUEPRINT_TO_NUSCENES.update(
    {
        # Trucks
        "vehicle.carlamotors.firetruck": "vehicle.truck",
        "vehicle.carlamotors.european_hgv": "vehicle.truck",
        # Buses
        "vehicle.mitsubishi.fusorosa": "vehicle.bus",
        # Bicycles
        "vehicle.bh.crossbike": "vehicle.bicycle",
        "vehicle.diamondback.century": "vehicle.bicycle",
        "vehicle.gazelle.omafiets": "vehicle.bicycle",
        # Motorcycles
        "vehicle.harley-davidson": "vehicle.motorcycle",
        "vehicle.kawasaki": "vehicle.motorcycle",
        "vehicle.vespa": "vehicle.motorcycle",
        "vehicle.yamaha": "vehicle.motorcycle",
        # Pedestrians
        "walker.pedestrian": "human.pedestrian.adult",
        # Static objects
        "static.prop.trafficcone": "movable_object.trafficcone",
        "static.prop.constructioncone": "movable_object.trafficcone",
        "static.prop.streetbarrier": "movable_object.barrier",
    }
)

# Class-specific maximum detection range (metres, 2-D Euclidean).
NUSCENES_RANGE = {
    "vehicle.car": 50.0,
    "vehicle.truck": 50.0,
    "vehicle.bus": 50.0,
    "vehicle.bicycle": 40.0,
    "vehicle.motorcycle": 40.0,
    "human.pedestrian.adult": 40.0,
    "movable_object.trafficcone": 30.0,
    "movable_object.barrier": 30.0,
}


def get_nuscenes_class(blueprint_id: str):
    """Return the nuScenes class string for a CARLA blueprint id, or None."""
    bp = blueprint_id.lower()

    # Try explicit prefix mapping first
    for prefix, nclass in BLUEPRINT_TO_NUSCENES.items():
        if bp.startswith(prefix):
            return nclass

    # Fallback keyword matching for any missed or new CARLA blueprints
    if any(k in bp for k in ("truck", "lorry", "hgv", "firetruck", "carlacola")):
        return "vehicle.truck"
    if any(k in bp for k in ("bus", "fusorosa", "tourus")):
        return "vehicle.bus"
    if any(
        k in bp
        for k in (
            "motorbike",
            "motorcycle",
            "scooter",
            "vespa",
            "yamaha",
            "kawasaki",
            "harley",
        )
    ):
        return "vehicle.motorcycle"
    if any(k in bp for k in ("bike", "bicycle", "cycle")):
        return "vehicle.bicycle"

    # Catch-all for any remaining vehicles
    if bp.startswith("vehicle."):
        return "vehicle.car"

    return None


# GRACEFUL SHUTDOWN
running = True


def _signal_handler(sig, frame):
    global running
    logging.warning("SIGINT received - finishing current scan then stopping ...")
    running = False


signal.signal(signal.SIGINT, _signal_handler)


# DIRECTORY SETUP & RESUME
def setup_directories(output_root: str, town: str):
    """Create per-town lidar/ and labels/ directories."""
    for sub in ("lidar", "labels"):
        os.makedirs(os.path.join(output_root, town, sub), exist_ok=True)


def count_existing_scans(output_root: str, town: str) -> int:
    """Count valid .npy files on disk for crash-resume.

    Validates each scan pair (.npy + .json) and removes corrupted files left behind by mid-crash writes.
    Returns the next scanindex to use (max existing index + 1, or 0 if empty).
    """
    lidar_dir = os.path.join(output_root, town, "lidar")
    label_dir = os.path.join(output_root, town, "labels")
    pattern = os.path.join(lidar_dir, "*.npy")
    files = sorted(glob.glob(pattern))

    if not files:
        return 0

    valid_indices = []
    removed = 0
    for f in files:
        tag = os.path.splitext(os.path.basename(f))[0]
        label_f = os.path.join(label_dir, tag + ".json")
        try:
            arr = np.load(f)
            if arr.ndim != 2 or arr.shape[1] != 4 or arr.shape[0] == 0:
                raise ValueError("bad shape")
            if not os.path.exists(label_f):
                raise ValueError("missing label")
            with open(label_f) as jf:
                json.load(jf)
            valid_indices.append(int(tag))
        except Exception:
            for p in (f, label_f):
                if os.path.exists(p):
                    try:
                        os.remove(p)
                    except OSError:
                        pass
            removed += 1

    if removed > 0:
        logging.warning(
            "Removed %d corrupted scan(s) in %s during resume validation.",
            removed,
            town,
        )

    if not valid_indices:
        return 0
    return max(valid_indices) + 1


def wait_for_carla(host, port, max_wait=300):
    """Block until CARLA server is reachable, up to max_wait seconds."""
    log = logging.getLogger("reconnect")
    log.warning("Lost connection to CARLA - waiting for server at %s:%d ...", host, port)
    t0 = time.time()
    while time.time() - t0 < max_wait:
        try:
            c = carla.Client(host, port)
            c.set_timeout(10.0)
            c.get_server_version()
            log.info("CARLA server reconnected after %.0fs.", time.time() - t0)
            return c
        except Exception:
            time.sleep(5)
    raise TimeoutError(f"CARLA server did not respond within {max_wait}s")


# WORLD & SYNC MODE
def load_and_configure_world(client, town_name, weather=None):
    """Load a CARLA town, enable sync mode, pre-roll physics, return (world, tm).

    Resets the Traffic Manager to clear its internal route cache
    weather is applied after pre-roll so that its attenuation settings affect the LiDAR.
    """
    log = logging.getLogger("world")
    log.info("Loading %s ...", town_name)

    try:
        old_tm = client.get_trafficmanager(TM_PORT)
        old_tm.shut_down()
        log.info("Traffic Manager shut down (clearing stale route cache).")
    except Exception:
        pass

    client.load_world(town_name)

    time.sleep(3.0)
    world = client.get_world()

    # Sync mode
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = FIXED_DELTA
    settings.no_rendering_mode = False  # LiDAR ray-cast doesn't need GPU render
    world.apply_settings(settings)

    # Fresh Traffic Manager for this town
    tm = client.get_trafficmanager(TM_PORT)
    tm.set_synchronous_mode(True)

    # Pre-roll to stabilise physics / nav mesh
    log.info("Pre-rolling %d ticks ...", PRE_ROLL_TICKS)
    for _ in range(PRE_ROLL_TICKS):
        world.tick()

    # Weather - applied from caller's random selection
    if weather is not None:
        world.set_weather(weather)
    log.info("%s ready.", town_name)
    return world, tm


# EGO & LiDAR


def _is_valid_road_spawn(carla_map, transform):
    """Return True if *transform* sits on a drivable lane (not sidewalk/shoulder)."""
    wp = carla_map.get_waypoint(
        transform.location,
        project_to_road=False,
        lane_type=carla.LaneType.Driving | carla.LaneType.Bidirectional,
    )
    if wp is None:
        return False

    # Must be close to the road surface (reject rooftops / stacked geometry)
    dz = abs(transform.location.z - wp.transform.location.z)
    if dz > 2.0:
        return False
    return True


def spawn_ego(world):
    """Spawn a Lincoln MKZ 2020 at a validated road spawn point, autopilot ON.

    Filters spawn points to Driving/Bidirectional lanes and rejects any that are vertically far from the road surface (rooftops, stacked geo).
    """
    bp = world.get_blueprint_library().find("vehicle.lincoln.mkz_2020")
    carla_map = world.get_map()
    spawn_points = list(carla_map.get_spawn_points())
    if not spawn_points:
        raise RuntimeError("No spawn points on this map!")

    # Shuffle and pick the first valid one
    random.shuffle(spawn_points)
    spawn_tf = None
    for sp in spawn_points:
        if _is_valid_road_spawn(carla_map, sp):
            spawn_tf = sp
            break

    if spawn_tf is None:
        # Fallback: use any spawn point
        logging.warning("No validated road spawn point found - using random.")
        spawn_tf = spawn_points[0]

    ego = world.spawn_actor(bp, spawn_tf)
    ego.set_autopilot(True, TM_PORT)
    logging.info(
        "Ego spawned at (%.1f, %.1f, %.1f)",
        spawn_tf.location.x,
        spawn_tf.location.y,
        spawn_tf.location.z,
    )
    return ego


def update_spectator(world, ego):
    """Move the CARLA spectator camera behind and above the ego vehicle.
    Looks down at a 20° angle, matching the typical view in CARLA's built-in LiDAR demos.
    """
    try:
        spectator = world.get_spectator()
        ego_tf = ego.get_transform()
        fwd = ego_tf.get_forward_vector()
        cam_loc = ego_tf.location + carla.Location(
            x=-8.0 * fwd.x, y=-8.0 * fwd.y, z=6.0
        )
        spectator.set_transform(
            carla.Transform(
                cam_loc, carla.Rotation(pitch=-20.0, yaw=ego_tf.rotation.yaw)
            )
        )
    except RuntimeError:
        pass  # ego may be destroyed


def attach_lidar(world, ego):
    """Attach a ray-cast LiDAR to the ego vehicle, return (sensor, queue)."""
    bp = world.get_blueprint_library().find("sensor.lidar.ray_cast")
    for attr, val in LIDAR_CFG.items():
        if bp.has_attribute(attr):
            bp.set_attribute(attr, val)
    tf = carla.Transform(LIDAR_POS)
    lidar = world.spawn_actor(bp, tf, attach_to=ego)
    sq = queue.Queue(maxsize=10)

    def _lidar_callback(data):
        """Non-blocking put - drops stale frames instead of blocking the callback thread."""
        try:
            sq.put_nowait(data)
        except queue.Full:
            pass  # drop frame; collection loop will consume the latest available

    lidar.listen(_lidar_callback)
    logging.info("LiDAR attached (32ch, 1.38M pts/s, 20Hz, 100m).")
    return lidar, sq


# CRASH / TOPPLE DETECTION

CRASH_IMPULSE_THRESHOLD = 5000.0  # N·s - discard scene above this
TOPPLE_ROLL_THRESHOLD = 45.0  # degrees - vehicle on its side
TOPPLE_PITCH_THRESHOLD = 45.0  # degrees - vehicle nose-diving / flipped
EGO_STUCK_SPEED = 0.3  # m/s - below this the ego is "stopped"
EGO_STUCK_TICKS = 200  # consecutive near-zero-speed frames -> stuck (~10s at 20Hz)

# Frame-skip when ego is slow - reduces near-duplicate scans at red lights /
# intersections without discarding any data.
SLOW_SPEED_THRESHOLD = 1.0  # m/s - below this, skip most frames
SLOW_FRAME_SKIP = 10  # save every Nth frame when slow


def attach_collision_sensor(world, ego):
    """Attach a collision sensor to ego. Returns (sensor, event_list)."""
    bp = world.get_blueprint_library().find("sensor.other.collision")
    sensor = world.spawn_actor(bp, carla.Transform(), attach_to=ego)
    events = []  # mutable list shared with callback thread

    def _on_collision(event):
        impulse = event.normal_impulse
        intensity = math.sqrt(impulse.x**2 + impulse.y**2 + impulse.z**2)
        events.append(
            {
                "frame": event.frame,
                "intensity": intensity,
                "other_actor": (
                    event.other_actor.type_id if event.other_actor else "unknown"
                ),
            }
        )

    sensor.listen(_on_collision)
    logging.info("Collision sensor attached.")
    return sensor, events


def check_ego_health(ego, collision_events, stuck_counter, clear=True):
    """Return (healthy: bool, reason: str, updated_stuck_counter: int, reason_type: str).

    Checks roll/pitch, collision impulse, and stuck detection.
    If *clear* is True the collision event buffer is drained after
    inspection so that old minor bumps don't accumulate.
    *stuck_counter* tracks consecutive near-zero-speed frames.

    *reason_type* is one of:
      - 'topple'  - vehicle rolled/pitched beyond threshold (data corrupted)
      - 'crash'   - major collision impulse detected (data corrupted)
      - 'stuck'   - ego stationary too long (data valid, just repetitive)
      - ''        - healthy
    """
    # Roll / Pitch
    rot = ego.get_transform().rotation
    if abs(rot.roll) > TOPPLE_ROLL_THRESHOLD:
        return False, f"roll={rot.roll:.1f}deg", 0, "topple"
    if abs(rot.pitch) > TOPPLE_PITCH_THRESHOLD:
        return False, f"pitch={rot.pitch:.1f}deg", 0, "topple"

    # Collision impulse
    for evt in collision_events:
        if evt["intensity"] > CRASH_IMPULSE_THRESHOLD:
            reason = (
                f"collision intensity={evt['intensity']:.0f} N*s "
                f"with {evt['other_actor']}"
            )
            if clear:
                collision_events.clear()
            return False, reason, 0, "crash"

    if clear:
        collision_events.clear()

    # Stuck detection (ego not moving for too long)
    vel = ego.get_velocity()
    speed = math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)
    if speed < EGO_STUCK_SPEED:
        stuck_counter += 1
    else:
        stuck_counter = 0
    if stuck_counter >= EGO_STUCK_TICKS:
        return (
            False,
            f"stuck for {stuck_counter} ticks (speed={speed:.2f} m/s)",
            0,
            "stuck",
        )

    return True, "", stuck_counter, ""


def discard_scene_scans(output_root, town, start_idx, end_idx):
    """Delete scan files [start_idx, end_idx) after a crash / topple."""
    log = logging.getLogger("discard")
    removed = 0
    for idx in range(start_idx, end_idx):
        tag = f"{idx:06d}"
        for sub, ext in [("lidar", ".npy"), ("labels", ".json")]:
            path = os.path.join(output_root, town, sub, tag + ext)
            if os.path.exists(path):
                try:
                    os.remove(path)
                    removed += 1
                except OSError:
                    pass
    if removed > 0:
        log.warning(
            "Discarded %d files for scans %d-%d in %s (crash/topple).",
            removed,
            start_idx,
            end_idx - 1,
            town,
        )
    return removed


# TRAFFIC SPAWNING
def spawn_vehicles(client, world, tm):
    """Spawn as many NPC vehicles as the map allows, return list of actor ids."""
    log = logging.getLogger("traffic")
    blueprints = world.get_blueprint_library().filter("vehicle.*")
    # Exclude bicycles/motorbikes from TM vehicle pool (they're unstable under TM)
    blueprints = [
        b for b in blueprints if int(b.get_attribute("number_of_wheels")) == 4
    ]
    # Deduplicate by base model so multi-variant vehicles (e.g. firetruck)
    # don't get overrepresented in random draws
    seen_bases = {}
    for b in blueprints:
        base = b.id.rsplit(".", 1)[0]
        if base not in seen_bases:
            seen_bases[base] = b
    blueprints = list(seen_bases.values())
    spawn_points = world.get_map().get_spawn_points()
    random.shuffle(spawn_points)

    batch = []
    for i, sp in enumerate(spawn_points):
        bp = random.choice(blueprints)
        if bp.has_attribute("color"):
            bp.set_attribute(
                "color", random.choice(bp.get_attribute("color").recommended_values)
            )
        bp.set_attribute("role_name", "autopilot")
        batch.append(
            carla.command.SpawnActor(bp, sp).then(
                carla.command.SetAutopilot(carla.command.FutureActor, True, TM_PORT)
            )
        )

    results = client.apply_batch_sync(batch, True)
    vehicle_ids = [r.actor_id for r in results if not r.error]

    # Post-spawn grounding check - destroy vehicles that ended up
    # floating above or buried below the road surface.
    carla_map = world.get_map()
    bad_ids = set()
    for vid in vehicle_ids:
        v = world.get_actor(vid)
        if v is None:
            bad_ids.add(vid)
            continue
        if not _actor_is_grounded(v, carla_map):
            log.debug("Destroying floating/underground NPC vehicle %d.", vid)
            try:
                v.set_autopilot(False, TM_PORT)
            except RuntimeError:
                pass
            try:
                v.destroy()
            except RuntimeError:
                pass
            bad_ids.add(vid)
    vehicle_ids = [vid for vid in vehicle_ids if vid not in bad_ids]

    # TM global settings - balance density with intersection flow
    tm.set_global_distance_to_leading_vehicle(2.5)
    tm.global_percentage_speed_difference(-20)  # 20% above limit
    tm.set_hybrid_physics_mode(True)
    tm.set_hybrid_physics_radius(100.0)
    tm.set_random_device_seed(random.randint(0, 2**31))  # path diversity
    for vid in vehicle_ids:
        v = world.get_actor(vid)
        if v is not None:
            _apply_tm_settings(tm, v)

    log.info(
        "Spawned %d / %d vehicles (after grounding filter).",
        len(vehicle_ids),
        len(spawn_points),
    )
    return vehicle_ids


def _apply_tm_settings(tm, vehicle):
    """Apply per-vehicle TM settings for natural traffic flow."""
    tm.ignore_lights_percentage(vehicle, 0)
    tm.ignore_signs_percentage(vehicle, 0)
    tm.auto_lane_change(vehicle, True)
    tm.distance_to_leading_vehicle(vehicle, 2.5)


def spawn_walkers(client, world, count=150):
    """Spawn pedestrian walkers with AI controllers, return (walker_ids, ctrl_ids)."""
    log = logging.getLogger("traffic")
    walker_bps = world.get_blueprint_library().filter("walker.pedestrian.*")
    ctrl_bp = world.get_blueprint_library().find("controller.ai.walker")

    # Random spawn locations on the navigation mesh
    spawn_locs = []
    for _ in range(count * 3):  # over-sample to handle failures
        loc = world.get_random_location_from_navigation()
        if loc is not None:
            spawn_locs.append(carla.Transform(loc))
    spawn_locs = spawn_locs[:count]

    # Batch spawn walkers
    batch = []
    for sp in spawn_locs:
        bp = random.choice(walker_bps)
        if bp.has_attribute("is_invincible"):
            bp.set_attribute("is_invincible", "false")
        batch.append(carla.command.SpawnActor(bp, sp))

    results = client.apply_batch_sync(batch, True)
    walker_ids = []
    for r in results:
        if not r.error:
            walker_ids.append(r.actor_id)

    # Batch spawn AI controllers
    ctrl_batch = []
    for wid in walker_ids:
        ctrl_batch.append(
            carla.command.SpawnActor(ctrl_bp, carla.Transform(), world.get_actor(wid))
        )
    ctrl_results = client.apply_batch_sync(ctrl_batch, True)
    ctrl_ids = []
    paired_walker_ids = []
    for wid, r in zip(walker_ids, ctrl_results):
        if not r.error:
            ctrl_ids.append(r.actor_id)
            paired_walker_ids.append(wid)

    # Tick so controllers materialise
    world.tick()

    # Start each controller
    for cid in ctrl_ids:
        ctrl = world.get_actor(cid)
        if ctrl is not None:
            ctrl.start()
            dest = world.get_random_location_from_navigation()
            if dest is not None:
                ctrl.go_to_location(dest)
            ctrl.set_max_speed(1.0 + random.random() * 1.5)

    # FIX: In CARLA 0.9.15+, attaching an AIWalkerController can silently
    # disable the walker's collision mesh for performance.  Without collisions,
    # LiDAR rays pass straight through the pedestrian.  Re-enable explicitly.
    for wid in paired_walker_ids:
        w = world.get_actor(wid)
        if w is not None:
            try:
                w.set_collisions(True)
                w.set_simulate_physics(True)
            except (AttributeError, RuntimeError):
                pass  # method unavailable in this CARLA build - collisions likely already on

    log.info(
        "Spawned %d walkers with %d controllers.", len(paired_walker_ids), len(ctrl_ids)
    )

    # Post-spawn grounding check - destroy walkers that ended up floating/underground
    carla_map = world.get_map()
    bad_wids = set()
    for wid in paired_walker_ids:
        w = world.get_actor(wid)
        if w is None:
            bad_wids.add(wid)
            continue
        if not _actor_is_grounded(w, carla_map):
            bad_wids.add(wid)
    # Destroy bad walkers and their paired controllers
    if bad_wids:
        bad_cids = set()
        for wid, cid in zip(paired_walker_ids, ctrl_ids):
            if wid in bad_wids:
                bad_cids.add(cid)
                for aid in (cid, wid):
                    a = world.get_actor(aid)
                    if a is not None:
                        try:
                            if hasattr(a, "stop"):
                                a.stop()
                            a.destroy()
                        except RuntimeError:
                            pass
        paired_walker_ids = [w for w in paired_walker_ids if w not in bad_wids]
        ctrl_ids = [c for c in ctrl_ids if c not in bad_cids]
        log.debug("Removed %d floating/underground walkers.", len(bad_wids))

    return paired_walker_ids, ctrl_ids


def spawn_static_props(world, ego_location, rng=None):
    """Bypassed - random procedural prop spawning is disabled to prevent
    unstructured 'salt-and-pepper' traffic-cone / barrier noise in the
    LiDAR dataset.  Returns an empty list so downstream callers
    (manage_density, cleanup) remain unaffected."""
    return []
    if rng is None:
        rng = random.Random()

    prop_filters = [
        "static.prop.trafficcone01",
        "static.prop.trafficcone02",
        "static.prop.constructioncone",
        "static.prop.streetbarrier",
    ]
    # Collect available blueprints
    bplib = world.get_blueprint_library()
    prop_bps = []
    for pf in prop_filters:
        found = bplib.filter(pf)
        prop_bps.extend(found)
    if not prop_bps:
        # Fallback: try broader filter
        prop_bps = list(bplib.filter("static.prop.trafficcone*"))
        prop_bps += list(bplib.filter("static.prop.constructioncone*"))
        prop_bps += list(bplib.filter("static.prop.streetbarrier*"))
    if not prop_bps:
        log.warning("No prop blueprints found - skipping prop spawn.")
        return []

    carla_map = world.get_map()
    target_count = rng.randint(PROP_COUNT_MIN, PROP_COUNT_MAX)
    prop_ids = []

    for _ in range(target_count * 3):  # over-sample
        if len(prop_ids) >= target_count:
            break
        # Random offset from ego in PROP_SPAWN_INNER..PROP_SPAWN_OUTER
        angle = rng.uniform(0, 2 * math.pi)
        dist = rng.uniform(PROP_SPAWN_INNER, PROP_SPAWN_OUTER)
        ox = ego_location.x + dist * math.cos(angle)
        oy = ego_location.y + dist * math.sin(angle)
        wp = carla_map.get_waypoint(
            carla.Location(x=ox, y=oy, z=ego_location.z),
            project_to_road=True,
            lane_type=(
                carla.LaneType.Driving
                | carla.LaneType.Bidirectional
                | carla.LaneType.Shoulder
                | carla.LaneType.Parking
            ),
        )
        if wp is None:
            continue
        # Place on road shoulder (offset laterally)
        right = wp.transform.get_forward_vector()
        lateral_offset = rng.choice([-1, 1]) * (
            wp.lane_width * 0.5 + rng.uniform(0.2, 1.0)
        )
        spawn_loc = wp.transform.location + carla.Location(
            x=-right.y * lateral_offset, y=right.x * lateral_offset, z=0.3
        )
        # Sanity-check: spawn_loc Z should be near the waypoint Z, not ego Z
        # (prevents props floating on overpasses or sinking on underpasses)
        spawn_tf = carla.Transform(spawn_loc, carla.Rotation(yaw=rng.uniform(0, 360)))
        bp = rng.choice(prop_bps)
        try:
            actor = world.try_spawn_actor(bp, spawn_tf)
            if actor is not None:
                # Verify prop is on the ground, not floating
                if _actor_is_grounded(actor, carla_map):
                    # Enable physics so the prop has a collision mesh
                    # for LiDAR ray-casts.  Without this, rays pass
                    # straight through the prop on some CARLA builds.
                    try:
                        actor.set_simulate_physics(True)
                    except (AttributeError, RuntimeError):
                        pass
                    prop_ids.append(actor.id)
                else:
                    try:
                        actor.destroy()
                    except RuntimeError:
                        pass
        except RuntimeError:
            pass

    log.info("Spawned %d static props.", len(prop_ids))
    return prop_ids


# DYNAMIC DENSITY MANAGEMENT


def _respawn_walkers_near(world, client, ego_loc, count=20):
    """Spawn new walkers near ego_loc to replenish culled pedestrians.
    Returns (new_walker_ids, new_ctrl_ids)."""
    walker_bps = world.get_blueprint_library().filter("walker.pedestrian.*")
    ctrl_bp = world.get_blueprint_library().find("controller.ai.walker")

    # Generate spawn locations near the ego (on the nav mesh)
    spawn_locs = []
    for _ in range(count * 3):
        loc = world.get_random_location_from_navigation()
        if loc is None:
            continue
        d = loc.distance(ego_loc)
        if 15.0 < d < DENSITY_RADIUS:
            spawn_locs.append(carla.Transform(loc))
        if len(spawn_locs) >= count:
            break

    if not spawn_locs:
        return [], []

    # Batch spawn walkers
    batch = []
    for sp in spawn_locs:
        bp = random.choice(walker_bps)
        if bp.has_attribute("is_invincible"):
            bp.set_attribute("is_invincible", "false")
        batch.append(carla.command.SpawnActor(bp, sp))

    results = client.apply_batch_sync(batch, True)
    new_wids = [r.actor_id for r in results if not r.error]

    # Batch spawn AI controllers
    ctrl_batch = [
        carla.command.SpawnActor(ctrl_bp, carla.Transform(), world.get_actor(wid))
        for wid in new_wids
    ]
    ctrl_results = client.apply_batch_sync(ctrl_batch, True)
    new_cids = []
    paired_wids = []
    for wid, r in zip(new_wids, ctrl_results):
        if not r.error:
            new_cids.append(r.actor_id)
            paired_wids.append(wid)

    # Start controllers
    for cid in new_cids:
        ctrl = world.get_actor(cid)
        if ctrl is not None:
            ctrl.start()
            dest = world.get_random_location_from_navigation()
            if dest is not None:
                ctrl.go_to_location(dest)
            ctrl.set_max_speed(1.0 + random.random() * 1.5)

    # FIX: Re-enable collisions so LiDAR rays hit walkers (see spawn_walkers).
    for wid in paired_wids:
        w = world.get_actor(wid)
        if w is not None:
            try:
                w.set_collisions(True)
                w.set_simulate_physics(True)
            except (AttributeError, RuntimeError):
                pass

    # Post-spawn grounding check
    carla_map = world.get_map()
    bad_wids = set()
    for wid in paired_wids:
        w = world.get_actor(wid)
        if w is None:
            bad_wids.add(wid)
            continue
        if not _actor_is_grounded(w, carla_map):
            bad_wids.add(wid)
    if bad_wids:
        bad_cids = set()
        for wid, cid in zip(paired_wids, new_cids):
            if wid in bad_wids:
                bad_cids.add(cid)
                for aid in (cid, wid):
                    a = world.get_actor(aid)
                    if a is not None:
                        try:
                            if hasattr(a, "stop"):
                                a.stop()
                            a.destroy()
                        except RuntimeError:
                            pass
        paired_wids = [w for w in paired_wids if w not in bad_wids]
        new_cids = [c for c in new_cids if c not in bad_cids]

    return paired_wids, new_cids


def _actors_within(world, center, radius, actor_ids):
    """Return (ids_inside, ids_outside) partitioned by distance to center."""
    inside, outside = [], []
    cx, cy, cz = center.x, center.y, center.z
    r2 = radius * radius
    for aid in actor_ids:
        a = world.get_actor(aid)
        if a is None:
            continue
        loc = a.get_location()
        dx, dy = loc.x - cx, loc.y - cy
        if dx * dx + dy * dy <= r2:
            inside.append(aid)
        else:
            outside.append(aid)
    return inside, outside


def manage_density(world, ego, tm, vehicle_ids, walker_ids, ctrl_ids, prop_ids, client):
    """
    Cull far-away NPCs (TM-safe), spawn new ones if density < target.
    Returns updated (vehicle_ids, walker_ids, ctrl_ids, prop_ids, density).
    """
    ego_loc = ego.get_location()

    #  Count inside density radius
    v_in, v_out_density = _actors_within(world, ego_loc, DENSITY_RADIUS, vehicle_ids)
    w_in, _ = _actors_within(world, ego_loc, DENSITY_RADIUS, walker_ids)
    p_in, _ = _actors_within(world, ego_loc, DENSITY_RADIUS, prop_ids)
    density = len(v_in) + len(w_in) + len(p_in)

    #  DESPAWN actors beyond DESPAWN_RADIUS (TM-safe)
    _, v_far = _actors_within(world, ego_loc, DESPAWN_RADIUS, vehicle_ids)
    _, w_far = _actors_within(world, ego_loc, DESPAWN_RADIUS, walker_ids)
    _, p_far = _actors_within(world, ego_loc, DESPAWN_RADIUS, prop_ids)

    # Vehicles: unregister from TM first, then destroy
    for vid in v_far:
        v = world.get_actor(vid)
        if v is not None:
            try:
                v.set_autopilot(False, TM_PORT)
            except RuntimeError:
                pass
            try:
                v.destroy()
            except RuntimeError:
                pass
    vehicle_ids = [vid for vid in vehicle_ids if vid not in set(v_far)]

    # Walkers: stop controller, destroy controller + walker
    w_far_set = set(w_far)
    ctrls_to_destroy = []
    walkers_to_destroy = list(w_far)
    # Find paired controllers
    for wid, cid in zip(walker_ids, ctrl_ids):
        if wid in w_far_set:
            ctrls_to_destroy.append(cid)
    for cid in ctrls_to_destroy:
        c = world.get_actor(cid)
        if c is not None:
            try:
                c.stop()
                c.destroy()
            except RuntimeError:
                pass
    for wid in walkers_to_destroy:
        w = world.get_actor(wid)
        if w is not None:
            try:
                w.destroy()
            except RuntimeError:
                pass
    walker_ids = [wid for wid in walker_ids if wid not in w_far_set]
    ctrl_ids = [cid for cid in ctrl_ids if cid not in set(ctrls_to_destroy)]

    # Props: destroy directly
    for pid in p_far:
        p = world.get_actor(pid)
        if p is not None:
            try:
                p.destroy()
            except RuntimeError:
                pass
    prop_ids = [pid for pid in prop_ids if pid not in set(p_far)]

    #  SPAWN more if density < DENSITY_MIN
    if density < DENSITY_MIN:
        deficit = DENSITY_TARGET - density
        # Try spawning extra vehicles near ego
        spawn_points = world.get_map().get_spawn_points()
        random.shuffle(spawn_points)
        near_sps = [
            sp
            for sp in spawn_points
            if 20.0 < sp.location.distance(ego_loc) < DENSITY_RADIUS
        ]

        blueprints = [
            b
            for b in world.get_blueprint_library().filter("vehicle.*")
            if int(b.get_attribute("number_of_wheels")) == 4
        ]
        # Deduplicate by base model so multi-variant vehicles (e.g. firetruck)
        # don't get overrepresented in random draws
        seen_bases = {}
        for b in blueprints:
            base = b.id.rsplit(".", 1)[0]
            if base not in seen_bases:
                seen_bases[base] = b
        blueprints = list(seen_bases.values())
        batch = []
        for sp in near_sps[:deficit]:
            bp = random.choice(blueprints)
            bp.set_attribute("role_name", "autopilot")
            batch.append(
                carla.command.SpawnActor(bp, sp).then(
                    carla.command.SetAutopilot(carla.command.FutureActor, True, TM_PORT)
                )
            )
        if batch:
            results = client.apply_batch_sync(batch, True)
            carla_map = world.get_map()
            for r in results:
                if not r.error:
                    v = world.get_actor(r.actor_id)
                    if v is not None and _actor_is_grounded(v, carla_map):
                        vehicle_ids.append(r.actor_id)
                        _apply_tm_settings(tm, v)
                    elif v is not None:
                        try:
                            v.set_autopilot(False, TM_PORT)
                        except RuntimeError:
                            pass
                        try:
                            v.destroy()
                        except RuntimeError:
                            pass

        # Respawn walkers near the ego to maintain pedestrian representation
        walker_deficit = max(
            0, deficit - len(batch)
        )  # remaining deficit after vehicles
        if walker_deficit > 0:
            new_w_ids, new_c_ids = _respawn_walkers_near(
                world, client, ego_loc, count=min(walker_deficit, 30)
            )
            walker_ids.extend(new_w_ids)
            ctrl_ids.extend(new_c_ids)

        # Also try spawning a few more props
        new_props = spawn_static_props(world, ego_loc)
        prop_ids.extend(new_props)

    return vehicle_ids, walker_ids, ctrl_ids, prop_ids, density


# STUCK-NPC CULLING - breaks intersection gridlock
NPC_STUCK_SPEED = 0.15  # m/s - below this the NPC is "stopped"
NPC_STUCK_TICKS_LIMIT = 600  # frames at 20 Hz -> 30 seconds stopped -> cull

# Module-level dict tracking per-NPC stuck ticks: {actor_id: consecutive_stuck_frames}
_npc_stuck_counters = {}


def cull_stuck_npcs(world, tm, vehicle_ids, ego_loc, radius=60.0):
    """Detect and destroy NPC vehicles that have been near-stationary.

    Vehicles within *radius* of ego that stay below NPC_STUCK_SPEED
    for NPC_STUCK_TICKS_LIMIT consecutive frames are destroyed.
    This breaks intersection deadlocks where TM's internal planner
    cannot resolve yield-cycle conflicts.

    Returns the updated vehicle_ids list.
    """
    global _npc_stuck_counters
    log = logging.getLogger("stuck_npc")
    r2 = radius * radius
    to_cull = []
    active_ids = set()

    for vid in vehicle_ids:
        v = world.get_actor(vid)
        if v is None:
            continue
        active_ids.add(vid)
        loc = v.get_location()
        dx, dy = loc.x - ego_loc.x, loc.y - ego_loc.y
        if dx * dx + dy * dy > r2:
            # Too far - reset counter but don't cull
            _npc_stuck_counters.pop(vid, None)
            continue

        vel = v.get_velocity()
        speed = math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)
        if speed < NPC_STUCK_SPEED:
            # Don't penalise vehicles legally waiting at a red / yellow light
            try:
                if v.is_at_traffic_light():
                    tl = v.get_traffic_light()
                    if tl is not None and tl.get_state() in (
                        carla.TrafficLightState.Red,
                        carla.TrafficLightState.Yellow,
                    ):
                        _npc_stuck_counters[vid] = 0
                        continue
            except RuntimeError:
                pass  # actor may have been destroyed mid-check
            _npc_stuck_counters[vid] = _npc_stuck_counters.get(vid, 0) + 1
        else:
            _npc_stuck_counters[vid] = 0

        if _npc_stuck_counters.get(vid, 0) >= NPC_STUCK_TICKS_LIMIT:
            to_cull.append(vid)

    # Destroy stuck vehicles (TM-safe)
    cull_set = set(to_cull)
    for vid in to_cull:
        v = world.get_actor(vid)
        if v is not None:
            try:
                v.set_autopilot(False, TM_PORT)
            except RuntimeError:
                pass
            try:
                v.destroy()
            except RuntimeError:
                pass
        _npc_stuck_counters.pop(vid, None)

    if to_cull:
        log.info("Culled %d stuck NPC vehicle(s) near ego.", len(to_cull))

    # Prune counters for actors that no longer exist
    _npc_stuck_counters = {
        k: v for k, v in _npc_stuck_counters.items() if k in active_ids
    }

    return [vid for vid in vehicle_ids if vid not in cull_set]


# LiDAR POINT-CLOUD PROCESSING  (CARLA -> nuScenes coords)


def process_lidar_data(measurement):
    """
    Parse raw LiDAR buffer -> (N, 4) float32 in nuScenes right-handed frame.
    nusc_x = carla_y,  nusc_y = carla_x,  nusc_z = carla_z
    """
    raw = np.frombuffer(measurement.raw_data, dtype=np.float32)
    points = raw.reshape(-1, 4)  # [carla_x, carla_y, carla_z, intensity]
    nusc = np.empty_like(points)
    nusc[:, 0] = points[:, 1]  # nusc X  =  carla Y
    nusc[:, 1] = points[:, 0]  # nusc Y  =  carla X
    nusc[:, 2] = points[:, 2]  # nusc Z  =  carla Z
    nusc[:, 3] = points[:, 3]  # intensity
    return nusc


# BOUNDING-BOX EXTRACTION  (vectorised point-in-box)


def _carla_inv_matrix(transform):
    """Return the 4×4 inverse matrix of a carla.Transform."""
    return np.array(transform.get_inverse_matrix(), dtype=np.float64)


def _count_points_in_box(points_xyz, center, half_ext, yaw_rad):
    """
    Vectorised count of points inside an oriented 3-D box.
    points_xyz : (N, 3) in nuScenes coords
    center     : (3,)   box centre  [x, y, z] nuScenes
    half_ext   : (3,)   half-extents [half_l, half_w, half_h]
                        (length along rot_x, width along rot_y, height along z)
    yaw_rad    : float  box yaw in nuScenes frame (radians)
    Returns int count.
    """
    shifted = points_xyz - center  # (N, 3)
    cos_y, sin_y = np.cos(-yaw_rad), np.sin(-yaw_rad)  # rotate into box frame
    rot_x = shifted[:, 0] * cos_y - shifted[:, 1] * sin_y
    rot_y = shifted[:, 0] * sin_y + shifted[:, 1] * cos_y
    mask = (
        (np.abs(rot_x) < half_ext[0])
        & (np.abs(rot_y) < half_ext[1])
        & (np.abs(shifted[:, 2]) < half_ext[2])
    )
    return int(mask.sum())


def cache_static_map_objects(world):
    """Query static environment objects and level BBs once per town.

    These results are map-dependent but frame-independent - they never
    change between ticks.  Caching eliminates ~5 CARLA RPC round-trips
    per frame (4× get_environment_objects + 1× get_level_bbs).

    Returns
    -------
    env_by_id : dict[int, (EnvironmentObject, str)]
        Deduplicated map of env object ID -> (object, nuScenes class).
    level_bbs : list[carla.BoundingBox]
        Raw level bounding boxes for vehicles.
    """
    log = logging.getLogger("bbox")

    # Strategy A (get_environment_objects) REMOVED - it returns invisible
    # routing triggers and non-physical geometry (e.g. SM_Tesla_SM_0) that
    # produce ghost bounding boxes on empty roads.  Strategy B (get_level_bbs)
    # is the CARLA-dev-recommended source for parked vehicle bounding boxes.
    env_by_id = {}

    # --- Strategy B (fallback): get_level_bbs ---
    level_bbs = []
    veh_label = getattr(
        carla.CityObjectLabel, "Vehicles", getattr(carla.CityObjectLabel, "Car", None)
    )
    if veh_label is not None:
        try:
            level_bbs = world.get_level_bbs(veh_label)
        except Exception:
            pass

    log.info(
        "Cached static map objects: env_objects=%d, level_bbs=%d",
        len(env_by_id),
        len(level_bbs),
    )
    return env_by_id, level_bbs


def _actor_is_grounded(actor, carla_map, max_above=2.0, max_below=1.0):
    """Return True if the actor is close to the road surface.

    Rejects actors that are floating in the air (spawned on rooftops,
    glitched geometry) or buried under the map.
    """
    actor_loc = actor.get_location()
    wp = carla_map.get_waypoint(
        actor_loc,
        project_to_road=True,
        lane_type=carla.LaneType.Any,
    )
    if wp is None:
        return True  # can't check - give benefit of the doubt
    dz = actor_loc.z - wp.transform.location.z
    if dz > max_above or dz < -max_below:
        return False
    return True


def _env_obj_nuscenes_class(obj_name):
    """Map an EnvironmentObject name to a nuScenes class, or None.

    Environment object names are mesh asset paths like
    'SM_Peugeot_ParkedCar_01', 'SM_Bus_Stop_Shelter', etc.
    We classify conservatively - only clear vehicle meshes.
    """
    name = obj_name.lower()
    # Explicit exclusions - street furniture, shelters, signs, etc.
    skip = (
        "shelter",
        "sign",
        "light",
        "pole",
        "bench",
        "trash",
        "bin",
        "hydrant",
        "mailbox",
        "fence",
        "wall",
        "barrier",
        "booth",
        "kiosk",
        "phone",
        "antenna",
        "lamp",
        "stop",
    )
    if any(s in name for s in skip):
        return None
    # Try to classify by keywords in the mesh name
    if any(k in name for k in ("truck", "lorry", "hgv", "firetruck", "carlacola")):
        return "vehicle.truck"
    if any(k in name for k in ("bus", "fusorosa", "tourus")):
        return "vehicle.bus"
    if any(
        k in name
        for k in (
            "motorbike",
            "motorcycle",
            "scooter",
            "vespa",
            "yamaha",
            "kawasaki",
            "harley",
        )
    ):
        return "vehicle.motorcycle"
    if any(k in name for k in ("bike", "bicycle", "cycle")):
        return "vehicle.bicycle"
    # Generic vehicle keywords
    if any(
        k in name
        for k in (
            "car",
            "vehicle",
            "van",
            "taxi",
            "sedan",
            "suv",
            "hatchback",
            "coupe",
            "pickup",
            "ambulance",
            "police",
            "jeep",
            "tesla",
            "audi",
            "bmw",
            "chevrolet",
            "ford",
            "nissan",
            "toyota",
            "mercedes",
            "lincoln",
            "dodge",
        )
    ):
        return "vehicle.car"

    # If no vehicle keywords matched, ignore this environment object.
    # (Since we are now querying ALL objects, we must NOT default to car)
    return None


def _transform_bb_to_world(obj_transform, bb_location):
    """Compute world-space bounding-box center from object transform + local bb offset.

    Uses CARLA's own get_matrix() to guarantee correctness under UE4's
    left-handed coordinate system.  A manual trigonometric matrix risks
    sign errors on the pitch/roll terms because UE4's rotation conventions
    differ from the standard right-handed Tait-Bryan formulas.
    """
    # get_matrix() returns a 4x4 list-of-lists; convert to numpy
    mat = np.array(obj_transform.get_matrix())  # (4, 4)
    local_pt = np.array([bb_location.x, bb_location.y, bb_location.z, 1.0])
    world_pt = mat @ local_pt
    return carla.Location(
        x=float(world_pt[0]), y=float(world_pt[1]), z=float(world_pt[2])
    )


def get_bounding_boxes(
    world, ego, lidar_tf, nusc_points, cached_env_by_id=None, cached_level_bbs=None
):
    """
    Build a list of annotation dicts for all nuScenes-relevant actors.
    Coordinates in nuScenes right-handed LiDAR frame.

    Includes both dynamic actors (from world.get_actors()) and static
    environment objects (parked vehicles baked into the map, from
    world.get_environment_objects()).

    Filters out actors that are floating above or buried below the road
    surface to prevent phantom / impossible labels.

    Performance: accepts pre-cached static data from cache_static_map_objects()
    to avoid re-querying every frame.  Falls back to live queries if caches
    are not provided.
    """
    lidar_inv = _carla_inv_matrix(lidar_tf)
    lidar_yaw = lidar_tf.rotation.yaw
    lidar_z_world = lidar_tf.location.z
    ego_id = ego.id
    carla_map = world.get_map()
    pts_xyz = nusc_points[:, :3].copy()  # (N, 3) for point-in-box tests

    # --- Opt 4: Pre-compute 2-D distances for all points ONCE ---
    # This lets us cheaply mask points before each _count_points_in_box call
    # instead of running the rotation math on all ~34k points every time.
    pts_dist_2d = np.sqrt(pts_xyz[:, 0] ** 2 + pts_xyz[:, 1] ** 2)

    annotations = []

    # ---- Pass 1: Dynamic actors (spawned vehicles, walkers, props) ----
    for actor in world.get_actors():
        if actor.id == ego_id:
            continue
        bp_id = actor.type_id
        nclass = get_nuscenes_class(bp_id)
        if nclass is None:
            continue

        # --- Opt 3: Single get_transform() call replaces separate ---
        # get_location() (in old _actor_is_grounded) + get_transform() (bbox).
        # This eliminates 1 RPC round-trip per actor.
        actor_tf = actor.get_transform()
        actor_loc = actor_tf.location

        # --- 7-DOF integrity guard: reject toppled/flipped vehicles ---
        # Our annotation format is [X,Y,Z,w,l,h,Yaw] - no roll or pitch.
        # A car on its side/roof has inverted LiDAR geometry but the GT box
        # says it's upright.  Skip it so the network treats it as background.
        if nclass.startswith("vehicle."):
            if (
                abs(actor_tf.rotation.roll) > 15.0
                or abs(actor_tf.rotation.pitch) > 15.0
            ):
                continue

        # --- Inline grounded check (was _actor_is_grounded) ---
        # Reuse actor_loc instead of calling actor.get_location() again.
        road_wp = carla_map.get_waypoint(
            actor_loc,
            project_to_road=True,
            lane_type=carla.LaneType.Any,
        )
        if road_wp is not None:
            dz = actor_loc.z - road_wp.transform.location.z
            if dz > 2.0 or dz < -1.0:
                continue  # floating / underground - reject

        # Bounding-box geometric center in world frame
        bb = actor.bounding_box
        bb_world = actor_tf.transform(bb.location)
        world_pt = np.array([bb_world.x, bb_world.y, bb_world.z, 1.0])
        local_carla = lidar_inv @ world_pt  # (4,)

        #  convert to nuScenes
        cx = local_carla[1]
        cy = local_carla[0]
        cz = local_carla[2]

        # distance filter (2-D)
        dist_2d = math.sqrt(cx * cx + cy * cy)
        max_range = NUSCENES_RANGE.get(nclass, 50.0)
        if dist_2d > max_range:
            continue

        # --- bounding-box size ---
        nusc_length = 2.0 * bb.extent.x
        nusc_width = 2.0 * bb.extent.y
        nusc_height = 2.0 * bb.extent.z

        # FIX: CARLA Walker Capsule Bug
        if nclass == "human.pedestrian.adult":
            nusc_length = max(nusc_length, 0.73)
            nusc_width = max(nusc_width, 0.67)
            nusc_height = max(nusc_height, 1.77)

        # --- Ground-clamp bbox bottom ---
        # Reuse road_wp from the grounded check above (was a second RPC call).
        cz_bottom = cz - bb.extent.z
        cz_top = cz + bb.extent.z
        if road_wp is not None:
            road_z_world = road_wp.transform.location.z
            road_z_lidar = road_z_world - lidar_z_world

            if cz_bottom > road_z_lidar and (
                nclass.startswith("vehicle.") or nclass.startswith("human.")
            ):
                cz_bottom = road_z_lidar
            if cz_bottom < road_z_lidar - 0.3:
                cz_bottom = road_z_lidar - 0.3

        if nclass.startswith("vehicle."):
            nusc_height = cz_top - cz_bottom
        final_cz = cz_bottom + nusc_height / 2.0

        # --- yaw ---
        actor_yaw = actor_tf.rotation.yaw  # reuse cached transform
        rel_deg = actor_yaw - lidar_yaw
        rel_rad = math.radians(rel_deg)
        nusc_yaw = math.atan2(math.cos(rel_rad), math.sin(rel_rad))

        # --- velocity in nuScenes frame ---
        vel = actor.get_velocity()
        v_world = np.array([vel.x, vel.y, vel.z, 0.0])
        v_local = lidar_inv @ v_world
        vx = v_local[1]
        vy = v_local[0]
        vz = v_local[2]

        # --- Opt 4: Pre-filtered point-in-box count ---
        center = np.array([cx, cy, final_cz])
        half_ext = np.array([nusc_length / 2.0, nusc_width / 2.0, nusc_height / 2.0])
        # Coarse radial pre-filter: only test points within dist_2d ± diagonal
        box_diag = math.sqrt(half_ext[0] ** 2 + half_ext[1] ** 2) + 0.5
        near_mask = np.abs(pts_dist_2d - dist_2d) < box_diag
        if near_mask.any():
            n_pts = _count_points_in_box(pts_xyz[near_mask], center, half_ext, nusc_yaw)
        else:
            n_pts = 0

        annotations.append(
            {
                "class": nclass,
                "center": [round(cx, 4), round(cy, 4), round(final_cz, 4)],
                "size": [
                    round(nusc_width, 4),
                    round(nusc_length, 4),
                    round(nusc_height, 4),
                ],
                "yaw_rad": round(nusc_yaw, 6),
                "velocity": [round(vx, 4), round(vy, 4), round(vz, 4)],
                "num_lidar_pts": n_pts,
                "actor_id": actor.id,
                "distance_2d": round(dist_2d, 3),
                "blueprint": bp_id,
            }
        )

    # ---- Pass 2: Static environment objects (parked vehicles in map) ----
    log = logging.getLogger("bbox")

    occupied_centers = [(ann["center"][0], ann["center"][1]) for ann in annotations]
    occupied_centers.append((0.0, 0.0))  # ego at LiDAR origin

    # Strategy A (get_environment_objects) removed - always empty.
    if cached_env_by_id is not None:
        env_by_id = cached_env_by_id
    else:
        env_by_id = {}

    if cached_level_bbs is not None:
        level_bbs = cached_level_bbs
    else:
        level_bbs = []
        veh_label = getattr(
            carla.CityObjectLabel,
            "Vehicles",
            getattr(carla.CityObjectLabel, "Car", None),
        )
        if veh_label is not None:
            try:
                level_bbs = world.get_level_bbs(veh_label)
            except Exception:
                pass

    n_env = len(env_by_id)

    log.debug(
        "Pass 2 sources: env_objects=%d, level_bbs=%d",
        n_env,
        len(level_bbs),
    )

    # Strategy A processing loop removed - env_by_id is always empty.

    # --- Strategy B: get_level_bbs (primary source for parked vehicles) ---
    # Process raw bounding boxes for any parked cars that get_environment_objects
    # missed.  Deduplicate against occupied_centers which now contains
    # both Pass 1 (dynamic actors) and Strategy A (env objects) positions.

    n_level_added = 0
    for lbb in level_bbs:
        nusc_length = 2.0 * lbb.extent.x
        nusc_width = 2.0 * lbb.extent.y
        nusc_height = 2.0 * lbb.extent.z

        if nusc_length < 0.5 or nusc_width < 0.5 or nusc_height < 0.3:
            continue
        if nusc_length > 15.0 or nusc_width > 5.0 or nusc_height > 5.0:
            continue

        world_pt = np.array([lbb.location.x, lbb.location.y, lbb.location.z, 1.0])
        local_carla = lidar_inv @ world_pt

        cx = local_carla[1]
        cy = local_carla[0]
        cz = local_carla[2]

        dist_2d = math.sqrt(cx * cx + cy * cy)
        if dist_2d > 50.0:
            continue

        is_duplicate = False
        for ox, oy in occupied_centers:
            if math.hypot(cx - ox, cy - oy) < 2.0:
                is_duplicate = True
                break
        if is_duplicate:
            continue
        occupied_centers.append((cx, cy))

        if abs(cz) > 4.0:
            continue

        road_wp = carla_map.get_waypoint(
            lbb.location,
            project_to_road=True,
            lane_type=carla.LaneType.Any,
        )
        if road_wp is None:
            continue

        cz_bottom = cz - lbb.extent.z
        cz_top = cz + lbb.extent.z
        road_z_world = road_wp.transform.location.z
        road_z_lidar = road_z_world - lidar_z_world

        if cz_bottom > road_z_lidar + 2.0 or cz_bottom < road_z_lidar - 2.0:
            continue

        if cz_bottom > road_z_lidar:
            cz_bottom = road_z_lidar
        if cz_bottom < road_z_lidar - 0.3:
            cz_bottom = road_z_lidar - 0.3

        nusc_height = cz_top - cz_bottom
        final_cz = cz_bottom + nusc_height / 2.0

        nusc_yaw = 0.0
        if hasattr(lbb, "rotation"):
            obj_yaw = lbb.rotation.yaw
            rel_deg = obj_yaw - lidar_yaw
            rel_rad = math.radians(rel_deg)
            nusc_yaw = math.atan2(math.cos(rel_rad), math.sin(rel_rad))

        # --- Opt 4: Pre-filtered point-in-box count ---
        center = np.array([cx, cy, final_cz])
        half_ext = np.array([nusc_length / 2.0, nusc_width / 2.0, nusc_height / 2.0])
        box_diag = math.sqrt(half_ext[0] ** 2 + half_ext[1] ** 2) + 0.5
        near_mask = np.abs(pts_dist_2d - dist_2d) < box_diag
        if near_mask.any():
            n_pts = _count_points_in_box(pts_xyz[near_mask], center, half_ext, nusc_yaw)
        else:
            n_pts = 0

        annotations.append(
            {
                "class": "vehicle.car",
                "center": [round(cx, 4), round(cy, 4), round(final_cz, 4)],
                "size": [
                    round(nusc_width, 4),
                    round(nusc_length, 4),
                    round(nusc_height, 4),
                ],
                "yaw_rad": round(nusc_yaw, 6),
                "velocity": [0.0, 0.0, 0.0],
                "num_lidar_pts": n_pts,
                "actor_id": -1,
                "distance_2d": round(dist_2d, 3),
                "blueprint": "level_bb_vehicle",
                "is_static": True,
            }
        )
        n_level_added += 1

    if n_level_added > 0:
        log.debug("Pass 2 level_bbs added %d extra parked vehicles.", n_level_added)

    return annotations


# PER-FRAME I/O
def save_frame(
    output_root,
    town,
    town_idx,
    global_idx,
    nusc_pc,
    annotations,
    ego_data,
    timestamp,
):
    """Save one scan to disk: .npy point cloud + .json labels.

    ego_data should be a plain dict with 'location', 'rotation', and
    optionally 'size' keys - NOT a CARLA Transform object.
    This allows safe execution from a background thread.
    """
    tag = f"{town_idx:06d}"

    # Point cloud
    pc_path = os.path.join(output_root, town, "lidar", tag + ".npy")
    np.save(pc_path, nusc_pc)

    # Labels
    label = {
        "frame_id": global_idx,
        "town_frame_id": town_idx,
        "town": town,
        "timestamp": timestamp,
        "num_points": int(nusc_pc.shape[0]),
        "ego_transform": ego_data,
        "objects": annotations,
    }
    lbl_path = os.path.join(output_root, town, "labels", tag + ".json")
    with open(lbl_path, "w") as f:
        json.dump(label, f)


class _SaveWorker:
    """Background thread that drains a queue of save_frame() calls.

    Opt 2: Offloads np.save + json.dump to a daemon thread so the main
    thread can immediately proceed to world.tick() without waiting for
    disk I/O.  The GIL is released during file writes, so this gives
    genuine parallelism.
    """

    def __init__(self):
        self._queue = queue.Queue(maxsize=64)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        while True:
            job = self._queue.get()
            if job is None:
                break  # shutdown sentinel
            try:
                save_frame(*job)
            except Exception as e:
                logging.getLogger("save").error("Async save failed: %s", e)
            finally:
                self._queue.task_done()

    def submit(self, *args):
        """Enqueue a save_frame() call (non-blocking unless queue is full)."""
        self._queue.put(args)

    def flush(self):
        """Block until all pending saves complete."""
        self._queue.join()

    def shutdown(self):
        """Signal the worker to exit and wait."""
        self._queue.put(None)
        self._thread.join(timeout=30)


# CLEANUP  (TM-safe)


def cleanup(
    client,
    world,
    tm,
    ego,
    lidar,
    vehicle_ids,
    walker_ids,
    ctrl_ids,
    prop_ids,
    sensor_queue,
    collision_sensor=None,
):
    """Destroy every actor we own, drain sensor queue, restore async mode."""
    log = logging.getLogger("cleanup")
    log.info("Cleaning up ...")

    # Stop collision sensor
    if collision_sensor is not None:
        try:
            collision_sensor.stop()
            collision_sensor.destroy()
        except Exception:
            pass

    # Stop sensor
    try:
        lidar.stop()
    except Exception:
        pass
    # Drain queue
    while not sensor_queue.empty():
        try:
            sensor_queue.get_nowait()
        except queue.Empty:
            break

    # Walker controllers - stop + destroy
    for cid in ctrl_ids:
        c = world.get_actor(cid)
        if c is not None:
            try:
                c.stop()
            except Exception:
                pass
    if ctrl_ids:
        client.apply_batch([carla.command.DestroyActor(x) for x in ctrl_ids])
    # Walkers
    if walker_ids:
        client.apply_batch([carla.command.DestroyActor(x) for x in walker_ids])

    # Vehicles - unregister from TM first
    for vid in vehicle_ids:
        v = world.get_actor(vid)
        if v is not None:
            try:
                v.set_autopilot(False, TM_PORT)
            except Exception:
                pass
    if vehicle_ids:
        client.apply_batch([carla.command.DestroyActor(x) for x in vehicle_ids])

    # Props
    if prop_ids:
        client.apply_batch([carla.command.DestroyActor(x) for x in prop_ids])

    # Ego + LiDAR
    try:
        lidar.destroy()
    except Exception:
        pass
    try:
        ego.destroy()
    except Exception:
        pass

    # Flush destruction
    for _ in range(10):
        try:
            world.tick()
        except Exception:
            break

    # Restore async mode so next load_world works cleanly
    try:
        settings = world.get_settings()
        settings.synchronous_mode = False
        settings.fixed_delta_seconds = None
        world.apply_settings(settings)
    except Exception:
        pass

    log.info("Cleanup done.")


def cleanup_scene(
    client,
    world,
    ego,
    lidar,
    vehicle_ids,
    walker_ids,
    ctrl_ids,
    prop_ids,
    sensor_queue,
    collision_sensor=None,
):
    """Destroy scene actors between re-spawns. Does NOT touch world settings."""
    log = logging.getLogger("cleanup")

    # Stop collision sensor
    if collision_sensor is not None:
        try:
            collision_sensor.stop()
            collision_sensor.destroy()
        except Exception:
            pass

    # Stop sensor
    try:
        lidar.stop()
    except Exception:
        pass
    # Drain queue
    while not sensor_queue.empty():
        try:
            sensor_queue.get_nowait()
        except queue.Empty:
            break

    # Walker controllers - stop + destroy
    for cid in ctrl_ids:
        c = world.get_actor(cid)
        if c is not None:
            try:
                c.stop()
            except Exception:
                pass
    if ctrl_ids:
        client.apply_batch([carla.command.DestroyActor(x) for x in ctrl_ids])
    if walker_ids:
        client.apply_batch([carla.command.DestroyActor(x) for x in walker_ids])

    # Vehicles - unregister from TM first
    for vid in vehicle_ids:
        v = world.get_actor(vid)
        if v is not None:
            try:
                v.set_autopilot(False, TM_PORT)
            except Exception:
                pass
    if vehicle_ids:
        client.apply_batch([carla.command.DestroyActor(x) for x in vehicle_ids])

    # Props
    if prop_ids:
        client.apply_batch([carla.command.DestroyActor(x) for x in prop_ids])

    # Ego + LiDAR
    try:
        lidar.destroy()
    except Exception:
        pass
    try:
        ego.destroy()
    except Exception:
        pass

    # Flush destruction
    for _ in range(10):
        try:
            world.tick()
        except Exception:
            break

    log.info("Scene cleanup done.")


# SUMMARY FILE


def generate_summary(stats, output_root, total_time):
    """Write dataset_summary.json with collection metadata."""
    # Aggregate class counts across all towns
    class_counts_total = {}
    for town_data in stats.values():
        for cls, cnt in town_data.get("class_counts", {}).items():
            class_counts_total[cls] = class_counts_total.get(cls, 0) + cnt

    total_scans = sum(td.get("scans", 0) for td in stats.values())

    summary = {
        "total_scans": total_scans,
        "target_scans": TOTAL_SCANS,
        "towns": {},
        "class_distribution": class_counts_total,
        "sensor_config": {
            "type": "sensor.lidar.ray_cast",
            "vehicle": "vehicle.lincoln.mkz_2020",
            "mount_xyz": [0.0, 0.0, 1.84],
            "channels": 32,
            "points_per_second": 691_200,
            "rotation_frequency_hz": 20,
            "upper_fov_deg": 10.67,
            "lower_fov_deg": -30.67,
            "range_m": 100.0,
        },
        "coordinate_system": "nuScenes LiDAR right-handed (X-right, Y-forward, Z-up)",
        "coordinate_transform": {
            "nusc_x": "carla_y",
            "nusc_y": "carla_x",
            "nusc_z": "carla_z",
        },
        "sim_config": {
            "fixed_delta_seconds": FIXED_DELTA,
            "synchronous_mode": True,
            "pre_roll_ticks": PRE_ROLL_TICKS,
        },
        "density_profiles": [str(p) for p in DENSITY_PROFILES],
        "weather": "randomized_per_scene (ClearNoon, HardRainNoon, CloudyNoon)",
        "carla_version": "0.9.16",
        "collection_date": datetime.datetime.now().isoformat(),
        "total_duration_sec": round(total_time, 2),
        "total_duration_hms": str(datetime.timedelta(seconds=int(total_time))),
    }

    # Per-town stats
    for town, td in stats.items():
        scans = td.get("scans", 0)
        summary["towns"][town] = {
            "scans": scans,
            "duration_sec": round(td.get("duration", 0), 2),
            "avg_points_per_frame": round(td.get("total_points", 0) / max(scans, 1), 1),
            "avg_objects_per_frame": round(
                td.get("total_objects", 0) / max(scans, 1), 2
            ),
            "avg_density_70m": round(
                td.get("total_density", 0) / max(td.get("density_samples", 1), 1), 1
            ),
            "class_counts": td.get("class_counts", {}),
        }

    path = os.path.join(output_root, "dataset_summary.json")
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    logging.info("Summary written -> %s", path)
    return summary


# MAIN LOOP


def main():
    parser = argparse.ArgumentParser(description="S2R CARLA -> nuScenes LiDAR collector")
    parser.add_argument(
        "--output",
        default="nuscenes_custom_dataset",
        help="Root output directory (default: nuscenes_custom_dataset)",
    )
    parser.add_argument("--host", default="localhost", help="CARLA host")
    parser.add_argument("--port", type=int, default=2000, help="CARLA port")
    parser.add_argument(
        "--mini",
        action="store_true",
        help=(
            "Laptop / smoke-test mode: 1 small town (Town02), 500 scans, "
            "reduced NPC density and walker count, shorter pre-roll. "
        ),
    )
    args = parser.parse_args()

    #  Mini Run
    # These shadow the module-level constants for this run only.
    global TOWN_PLAN, DENSITY_MIN, DENSITY_MAX, DENSITY_TARGET
    global PRE_ROLL_TICKS, PROP_COUNT_MIN, PROP_COUNT_MAX
    global DENSITY_PROFILES, SCENE_LENGTH_MIN, SCENE_LENGTH_MAX
    global running
    if args.mini:
        TOWN_PLAN = [("Town02", 500)]  # small town, 500 scans ≈ 25 s real-time
        SCENE_LENGTH_MIN = 100
        SCENE_LENGTH_MAX = 250
        PRE_ROLL_TICKS = 30  # reduced from 100
        DENSITY_PROFILES = [
            {
                "density_min": 5,
                "density_max": 10,
                "density_target": 8,
                "walkers": 15,
                "prop_min": 2,
                "prop_max": 5,
            },
            {
                "density_min": 10,
                "density_max": 18,
                "density_target": 14,
                "walkers": 30,
                "prop_min": 3,
                "prop_max": 8,
            },
        ]
        if args.output == "nuscenes_custom_dataset":
            args.output = "nuscenes_mini_test"  # separate output folder
        print("[MINI MODE] Laptop config active - Town02, 500 scans, low density.")

    output_root = args.output
    os.makedirs(output_root, exist_ok=True)

    # Recompute TOTAL_SCANS after any mini-mode overrides
    global TOTAL_SCANS
    TOTAL_SCANS = sum(q for _, q in TOWN_PLAN)

    # Logging
    log_fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=log_fmt,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(
                os.path.join(output_root, "collection.log"), mode="a", encoding="utf-8"
            ),
        ],
    )
    log = logging.getLogger("main")

    # Dedicated crash log - ERROR and above only + faulthandler segfault traces
    _crash_log_path = os.path.join(output_root, "crash_log.txt")
    _crash_log_fh = open(_crash_log_path, "a", encoding="utf-8")
    faulthandler.enable(file=_crash_log_fh)
    crash_handler = logging.FileHandler(_crash_log_path, mode="a", encoding="utf-8")
    crash_handler.setLevel(logging.ERROR)
    crash_handler.setFormatter(logging.Formatter(log_fmt))
    logging.getLogger().addHandler(crash_handler)

    log.info("=" * 70)
    log.info("S2R CARLA Dataset Collection - START")
    log.info("Output: %s | Target: %d scans", output_root, TOTAL_SCANS)
    log.info("Crash diagnostics -> %s", _crash_log_path)
    log.info("=" * 70)

    #  Connect (with retry - survives server restarts)
    client = None
    for _conn_attempt in range(6):
        try:
            client = carla.Client(args.host, args.port)
            client.set_timeout(30.0)
            server_ver = client.get_server_version()
            log.info(
                "Connected to CARLA at %s:%d | Python %s | Server %s",
                args.host,
                args.port,
                sys.version.split()[0],
                server_ver,
            )
            break
        except RuntimeError as e:
            if _conn_attempt < 5:
                log.warning(
                    "Connection attempt %d/6 failed: %s - retrying in 10s ...",
                    _conn_attempt + 1,
                    e,
                )
                time.sleep(10)
            else:
                log.error("All 6 connection attempts failed. Exiting.")
                raise

    all_stats = {}
    global_idx = 0
    t0_total = time.time()

    # Town loop
    town_idx_in_plan = 0
    while town_idx_in_plan < len(TOWN_PLAN):
        town_name, quota = TOWN_PLAN[town_idx_in_plan]
        if not running:
            break

        setup_directories(output_root, town_name)
        existing = count_existing_scans(output_root, town_name)
        if existing >= quota:
            log.info(
                "[%s] Already complete (%d / %d). Skipping.", town_name, existing, quota
            )
            global_idx += quota
            # Still need stats for summary if town was already done
            if town_name not in all_stats:
                all_stats[town_name] = {
                    "scans": existing,
                    "duration": 0,
                    "total_points": 0,
                    "total_objects": 0,
                    "total_density": 0,
                    "density_samples": 0,
                    "class_counts": {},
                }
            town_idx_in_plan += 1
            continue

        log.info("=" * 50)
        log.info(
            "[%s] Collecting scans %d -> %d (resume from %d)",
            town_name,
            existing,
            quota,
            existing,
        )
        log.info("=" * 50)

        # Load world once per town (weather set per scene for diversity)
        try:
            world, tm = load_and_configure_world(client, town_name)
        except RuntimeError as e:
            if "timeout" in str(e).lower() or "time-out" in str(e).lower():
                log.error("CARLA timeout loading %s: %s", town_name, e)
                log.info("Attempting reconnection ...")
                try:
                    client = wait_for_carla(args.host, args.port)
                    world, tm = load_and_configure_world(client, town_name)
                except Exception:
                    log.exception(
                        "Reconnect failed - skipping %s (will resume next run)",
                        town_name,
                    )
                    town_idx_in_plan += 1
                    continue
            else:
                raise

        # Per-town stats
        town_stats = {
            "scans": 0,
            "duration": 0.0,
            "total_points": 0,
            "total_objects": 0,
            "total_density": 0,
            "density_samples": 0,
            "class_counts": {},
        }
        t0_town = time.time()
        last_density = 0
        town_scan_idx = existing

        # --- Opt 1: Cache static map objects once per town ---
        cached_env_by_id, cached_level_bbs = cache_static_map_objects(world)

        # --- Opt 2: Start async save worker for this town ---
        save_worker = _SaveWorker()

        # SCENE LOOP (periodic re-spawn for spatial diversity)
        scene_num = 0
        _carla_lost = False
        while town_scan_idx < quota and running and not _carla_lost:
            scene_num += 1

            # Sample scene configuration
            profile = random.choice(DENSITY_PROFILES)
            DENSITY_MIN = profile["density_min"]
            DENSITY_MAX = profile["density_max"]
            DENSITY_TARGET = profile["density_target"]
            PROP_COUNT_MIN = profile["prop_min"]
            PROP_COUNT_MAX = profile["prop_max"]
            _scene_walkers = profile["walkers"]

            weather_name, weather_params = random.choice(WEATHER_PRESETS)
            world.set_weather(weather_params)

            scene_length = random.randint(SCENE_LENGTH_MIN, SCENE_LENGTH_MAX)
            scene_length = min(scene_length, quota - town_scan_idx)

            # Spawn ego at a fresh random road location
            ego = spawn_ego(world)
            update_spectator(world, ego)  # snap camera to new ego
            # Let ego settle on the road
            for _ in range(20):
                world.tick()

            # --- Post-settle road validation ---
            # After physics settle the ego might have slid off-road or
            # clipped into geometry.  Verify it is still on a drivable lane.
            ego_loc = ego.get_location()
            _post_wp = world.get_map().get_waypoint(
                ego_loc,
                project_to_road=False,
                lane_type=carla.LaneType.Driving | carla.LaneType.Bidirectional,
            )
            if _post_wp is None or abs(ego_loc.z - _post_wp.transform.location.z) > 2.0:
                log.warning(
                    "Ego landed off-road after settle (z=%.1f, wp_z=%s) - "
                    "re-spawning scene.",
                    ego_loc.z,
                    f"{_post_wp.transform.location.z:.1f}" if _post_wp else "None",
                )
                try:
                    ego.destroy()
                except Exception:
                    pass
                for _ in range(5):
                    world.tick()
                continue  # retry the while loop with a new scene

            lidar, sensor_queue = attach_lidar(world, ego)
            collision_sensor, collision_events = attach_collision_sensor(world, ego)
            # Let sensor initialise
            for _ in range(10):
                world.tick()
            # Drain any stale sensor data
            while not sensor_queue.empty():
                try:
                    sensor_queue.get_nowait()
                except queue.Empty:
                    break
            collision_events.clear()  # ignore settle-phase bumps

            # Spawn traffic matching this scene's density profile
            vehicle_ids = spawn_vehicles(client, world, tm)
            walker_ids, ctrl_ids = spawn_walkers(client, world, count=_scene_walkers)
            for _ in range(20):
                world.tick()
            # Drain sensor queue again after traffic spawn
            while not sensor_queue.empty():
                try:
                    sensor_queue.get_nowait()
                except queue.Empty:
                    break

            prop_ids = spawn_static_props(world, ego.get_location())
            # One more tick
            world.tick()
            while not sensor_queue.empty():
                try:
                    sensor_queue.get_nowait()
                except queue.Empty:
                    break
            collision_events.clear()  # ignore traffic-spawn bumps

            # Scene-level bookkeeping (stats merged only on success)
            scene_start_idx = town_scan_idx
            scene_discarded = False
            scene_scans = 0
            scene_points = 0
            scene_objects = 0
            scene_class_counts = {}
            _stuck_counter = 0  # consecutive low-speed frames
            _npc_stuck_counters.clear()  # reset NPC gridlock tracker for new scene

            log.info(
                "[%s] Scene %d started - %d scans | density: %d–%d | "
                "walkers: %d | weather: %s",
                town_name,
                scene_num,
                scene_length,
                DENSITY_MIN,
                DENSITY_MAX,
                _scene_walkers,
                weather_name,
            )

            # SCENE COLLECTION LOOP
            for scene_scan in range(scene_length):
                if not running:
                    log.warning(
                        "Interrupted at scan %d / %d in %s (scene %d).",
                        town_scan_idx,
                        quota,
                        town_name,
                        scene_num,
                    )
                    break

                scan_idx = town_scan_idx

                try:
                    # Tick simulation
                    world.tick()

                    # Follow cam (every 5 ticks to keep smooth but cheap)
                    if scene_scan % 5 == 0:
                        update_spectator(world, ego)

                    # Grab LiDAR data (blocks until sensor delivers)
                    try:
                        lidar_data = sensor_queue.get(timeout=10.0)
                    except queue.Empty:
                        log.error(
                            "Sensor timeout at scan %d - skipping frame.", scan_idx
                        )
                        continue

                    # Process point cloud
                    nusc_pc = process_lidar_data(lidar_data)

                    # --- Health check (topple / major crash / stuck) ---
                    try:
                        healthy, reason, _stuck_counter, reason_type = check_ego_health(
                            ego,
                            collision_events,
                            _stuck_counter,
                        )
                    except RuntimeError:
                        healthy, reason, _stuck_counter, reason_type = (
                            False,
                            "ego destroyed during health check",
                            0,
                            "crash",
                        )
                    if not healthy:
                        if reason_type == "stuck":
                            # Stuck at intersection / traffic - data is valid,
                            # just end the scene gracefully and keep all scans.
                            log.info(
                                "Ego stuck at scan %d (scene %d): %s - "
                                "ending scene (keeping %d valid scans).",
                                scan_idx,
                                scene_num,
                                reason,
                                scene_scans,
                            )
                            break  # scene_discarded stays False -> scans kept
                        else:
                            # Crash or topple - data is physically corrupted.
                            log.warning(
                                "Ego compromised at scan %d (scene %d): %s - "
                                "discarding scene and re-spawning.",
                                scan_idx,
                                scene_num,
                                reason,
                            )
                            scene_discarded = True
                            break

                    # --- Frame-skip when ego is slow (anti-duplicate) ---
                    ego_vel = ego.get_velocity()
                    ego_speed = math.sqrt(ego_vel.x**2 + ego_vel.y**2 + ego_vel.z**2)
                    if ego_speed < SLOW_SPEED_THRESHOLD:
                        if scene_scan % SLOW_FRAME_SKIP != 0:
                            continue  # skip this frame, don't save

                    # Bounding boxes (Opt 1: pass cached static data)
                    lidar_transform = lidar.get_transform()
                    annotations = get_bounding_boxes(
                        world,
                        ego,
                        lidar_transform,
                        nusc_pc,
                        cached_env_by_id=cached_env_by_id,
                        cached_level_bbs=cached_level_bbs,
                    )

                    # --- Opt 2: Extract ego data to plain dict, submit async ---
                    sim_ts = lidar_data.timestamp
                    ego_tf = ego.get_transform()
                    ego_bb = ego.bounding_box
                    ego_loc = ego_tf.location
                    ego_rot = ego_tf.rotation
                    ego_data = {
                        "location": [
                            round(ego_loc.x, 4),
                            round(ego_loc.y, 4),
                            round(ego_loc.z, 4),
                        ],
                        "rotation": [
                            round(ego_rot.roll, 4),
                            round(ego_rot.pitch, 4),
                            round(ego_rot.yaw, 4),
                        ],
                        "size": [
                            round(2.0 * ego_bb.extent.y, 4),
                            round(2.0 * ego_bb.extent.x, 4),
                            round(2.0 * ego_bb.extent.z, 4),
                        ],
                    }
                    save_worker.submit(
                        output_root,
                        town_name,
                        scan_idx,
                        global_idx,
                        nusc_pc,
                        annotations,
                        ego_data,
                        sim_ts,
                    )

                    # Update scene-level stats (merged into town only on success)
                    scene_scans += 1
                    scene_points += nusc_pc.shape[0]
                    scene_objects += len(annotations)
                    for ann in annotations:
                        cls = ann["class"]
                        scene_class_counts[cls] = scene_class_counts.get(cls, 0) + 1

                    # Density management + stuck-NPC culling
                    if scan_idx % DENSITY_CHECK_EVERY == 0:
                        vehicle_ids, walker_ids, ctrl_ids, prop_ids, density = (
                            manage_density(
                                world,
                                ego,
                                tm,
                                vehicle_ids,
                                walker_ids,
                                ctrl_ids,
                                prop_ids,
                                client,
                            )
                        )
                        last_density = density
                        town_stats["total_density"] += density
                        town_stats["density_samples"] += 1

                        # Cull NPCs stuck at intersections
                        vehicle_ids = cull_stuck_npcs(
                            world, tm, vehicle_ids, ego.get_location()
                        )

                    # Progress logging
                    total_town_scans = town_stats["scans"] + scene_scans
                    if scan_idx % LOG_EVERY == 0 and scan_idx > existing:
                        elapsed = time.time() - t0_town
                        fps = total_town_scans / max(elapsed, 1e-6)
                        avg_pts = (town_stats["total_points"] + scene_points) / max(
                            total_town_scans, 1
                        )
                        avg_obj = (town_stats["total_objects"] + scene_objects) / max(
                            total_town_scans, 1
                        )
                        eta_sec = (quota - scan_idx) / max(fps, 1e-6)
                        log.info(
                            "[%s] %d / %d scans (scene %d) | density: %d | "
                            "%.0f pts/frame | %.1f obj/frame | "
                            "%.2f fps | ETA: %s",
                            town_name,
                            scan_idx,
                            quota,
                            scene_num,
                            last_density,
                            avg_pts,
                            avg_obj,
                            fps,
                            str(datetime.timedelta(seconds=int(eta_sec))),
                        )

                    town_scan_idx += 1
                    global_idx += 1

                except RuntimeError as e:
                    err_msg = str(e).lower()
                    if "destroyed actor" in err_msg:
                        log.error(
                            "Ego/actor destroyed at scan %d (scene %d) - "
                            "discarding scene: %s",
                            scan_idx,
                            scene_num,
                            e,
                        )
                        scene_discarded = True
                        break
                    elif "timeout" in err_msg or "time-out" in err_msg:
                        log.error(
                            "CARLA timeout at scan %d (scene %d): %s",
                            scan_idx,
                            scene_num,
                            e,
                        )
                        _carla_lost = True
                        break
                    else:
                        raise

            # --- Merge or discard scene results ---
            if scene_discarded:
                discard_scene_scans(
                    output_root, town_name, scene_start_idx, town_scan_idx
                )
                scans_lost = town_scan_idx - scene_start_idx
                global_idx -= scans_lost
                town_scan_idx = scene_start_idx
                log.warning(
                    "[%s] Scene %d DISCARDED - rewound %d scans.",
                    town_name,
                    scene_num,
                    scans_lost,
                )
            else:
                # Merge scene stats into town totals
                town_stats["scans"] += scene_scans
                town_stats["total_points"] += scene_points
                town_stats["total_objects"] += scene_objects
                for cls, cnt in scene_class_counts.items():
                    town_stats["class_counts"][cls] = (
                        town_stats["class_counts"].get(cls, 0) + cnt
                    )

            # End of scene - cleanup actors
            try:
                cleanup_scene(
                    client,
                    world,
                    ego,
                    lidar,
                    vehicle_ids,
                    walker_ids,
                    ctrl_ids,
                    prop_ids,
                    sensor_queue,
                    collision_sensor,
                )
            except Exception:
                log.exception("Error during scene cleanup (non-fatal).")
            log.info(
                "[%s] Scene %d ended - %d / %d town scans collected.",
                town_name,
                scene_num,
                town_scan_idx - existing,
                quota - existing,
            )

        # Handle CARLA connection loss after scene loop exits
        if _carla_lost:
            log.warning(
                "[%s] CARLA connection lost - attempting reconnect for retry ...",
                town_name,
            )
            try:
                client = wait_for_carla(args.host, args.port)
                # Don't continue to next town - re-enter the town loop
                # to retry this town from its checkpoint.
                log.info("Reconnected. Will resume %s from checkpoint.", town_name)
                continue
            except TimeoutError:
                log.error(
                    "CARLA server did not come back - saving progress and exiting."
                )
                running = False
                break

        # END SCENE LOOP

        town_stats["duration"] = time.time() - t0_town
        all_stats[town_name] = town_stats
        log.info(
            "[%s] Finished - %d scans in %d scenes, %s.",
            town_name,
            town_stats["scans"],
            scene_num,
            str(datetime.timedelta(seconds=int(town_stats["duration"]))),
        )

        # --- Opt 2: Flush all pending async saves before switching town ---
        save_worker.flush()
        save_worker.shutdown()

        # Restore async mode before next town's load_world
        try:
            settings = world.get_settings()
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            world.apply_settings(settings)
        except Exception:
            pass

        town_idx_in_plan += 1

    # Final summary
    total_time = time.time() - t0_total
    summary = generate_summary(all_stats, output_root, total_time)
    total_collected = sum(td.get("scans", 0) for td in all_stats.values())
    log.info("=" * 70)
    log.info(
        "COLLECTION COMPLETE - %d / %d scans in %s",
        total_collected,
        TOTAL_SCANS,
        str(datetime.timedelta(seconds=int(total_time))),
    )
    log.info("=" * 70)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logging.warning("KeyboardInterrupt - exiting.")
    except Exception as e:
        logging.exception("Fatal error: %s", e)
        sys.exit(1)
