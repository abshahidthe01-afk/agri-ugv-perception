"""
Command-line tool: record sets of pictures with depth from the dome cameras during a mission.

For make_labels. Start the simulation with the cameras measuring depth, e.g.
    ros2 launch agri_ugv_perception_sim mission.launch.py plots:=198 cameras:=top
        camera_depth:=true scanners:=false gui:=false
then this tool, then /mission/start:
    ros2 run agri_ugv_perception_sim record_pictures --output ~/agri_data/sim/recordings/p198
Every --every seconds of simulated time, once the robot has moved at least --min-move metres
since the last set, it keeps the next set of pictures that all cameras took at the same
moment: the colour pictures, the depth images, the calibrations, the cameras' places on the
robot (their optical links in base_footprint, from /robot_description) and the robot's true
pose at that moment (/ground_truth/odom, between the two nearest samples). It stops when
the mission is complete (/mission/status), after --sets sets or --seconds of wall-clock
time, or on Ctrl+C, and saves <output>.npz and a contact sheet <output>.png.
"""

import argparse
import json
import math
from pathlib import Path
import time

import numpy as np

TOP = (16, 17, 18, 19)          # the top ring of dome cameras, closest to looking straight down


def pose_at(samples, t):
    """
    Return the pose (x, y, z, roll, pitch, yaw) at time t.

    samples: (t, x, y, z, roll, pitch, yaw) in time order. The pose changes evenly between
    the two samples around t (angles the short way round); before the first sample or
    after the last one, it is that sample's pose.
    """
    samples = np.asarray(samples, dtype=float)
    after = int(np.searchsorted(samples[:, 0], t))
    if after == 0:
        return tuple(samples[0, 1:])
    if after == len(samples):
        return tuple(samples[-1, 1:])
    a, b = samples[after - 1], samples[after]
    w = (t - a[0]) / (b[0] - a[0]) if b[0] > a[0] else 0.0
    step = b[1:] - a[1:]
    step[3:] = (step[3:] + math.pi) % (2 * math.pi) - math.pi
    pose = a[1:] + w * step
    pose[3:] = (pose[3:] + math.pi) % (2 * math.pi) - math.pi
    return tuple(pose)


def sheet(pictures, columns, scale=0.5):
    """Return the pictures side by side as one RGB array, 'columns' per row, at 'scale'."""
    from PIL import Image   # Pillow (python3-pil)

    h, w = (int(round(n * scale)) for n in pictures[0].shape[:2])
    rows = -(-len(pictures) // columns)
    board = np.zeros((rows * h, columns * w, 3), np.uint8)
    for n, picture in enumerate(pictures):
        small = np.asarray(Image.fromarray(picture).resize((w, h)))
        r, c = divmod(n, columns)
        board[r * h:(r + 1) * h, c * w:(c + 1) * w] = small
    return board


def record(cameras, sets, every, min_move, seconds):
    """Follow the cameras, the robot and the mission; return the kept sets and the context."""
    from agri_ugv_phenotyping.capture import depth_image, picture, pose_of, stamp
    from nav_msgs.msg import Odometry
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import DurabilityPolicy, QoSProfile
    from rosgraph_msgs.msg import Clock
    from sensor_msgs.msg import CameraInfo, Image
    from std_msgs.msg import String

    pictures = {i: {} for i in cameras}
    depths = {i: {} for i in cameras}
    calibration = {}
    state = {'description': None, 'truth': [], 'clock': [], 'mission': None}

    def keep(store, i, t, value):
        store[i][t] = value
        for old in sorted(store[i])[:-3]:
            del store[i][old]

    def on_truth(msg):
        state['truth'] = (state['truth'] + [(stamp(msg), *pose_of(msg))])[-1000:]

    def on_status(msg):
        state['mission'] = json.loads(msg.data).get('state')

    rclpy.init()
    node = Node('record_pictures')
    for i in cameras:
        node.create_subscription(Image, f'/cameras/dome_{i}/image',
                                 lambda m, i=i: keep(pictures, i, stamp(m), picture(m)), 2)
        node.create_subscription(Image, f'/cameras/dome_{i}/depth_image',
                                 lambda m, i=i: keep(depths, i, stamp(m), depth_image(m)), 2)
        node.create_subscription(CameraInfo, f'/cameras/dome_{i}/camera_info',
                                 lambda m, i=i: calibration.__setitem__(i, list(m.k)), 2)
    latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
    node.create_subscription(String, '/robot_description',
                             lambda m: state.__setitem__('description', m.data), latched)
    node.create_subscription(String, '/mission/status', on_status, latched)
    node.create_subscription(Odometry, '/ground_truth/odom', on_truth, 50)
    node.create_subscription(
        Clock, '/clock', lambda m: state['clock'].append(
            (time.monotonic(), m.clock.sec + 1e-9 * m.clock.nanosec)), 10)

    kept, last_t, last_xy = [], None, None
    begin, warned = time.monotonic(), 0.0
    print(f'Waiting for pictures from cameras {list(cameras)} (start the mission) ...')
    try:
        while len(kept) < sets and time.monotonic() - begin < seconds:
            rclpy.spin_once(node, timeout_sec=0.05)
            if state['mission'] == 'complete':
                print('Mission complete')
                break
            missing = [i for i in cameras if not depths[i] or i not in calibration]
            if missing or not state['truth']:
                if time.monotonic() - warned > 15:
                    warned = time.monotonic()
                    print(f'  still waiting: cameras without depth/calibration {missing}, '
                          f'true pose {"yes" if state["truth"] else "no"}')
                continue
            common = set.intersection(*[set(pictures[i]) & set(depths[i]) for i in cameras])
            if not common:
                continue
            t = max(common)
            if (last_t is not None and t < last_t + every) or t > state['truth'][-1][0]:
                continue                       # too soon, or the pose at t not known yet
            pose = pose_at(state['truth'], t)
            if last_xy is not None and math.hypot(pose[0] - last_xy[0],
                                                  pose[1] - last_xy[1]) < min_move:
                continue
            kept.append({'t': t, 'truth': (t, *pose),
                         'pictures': [pictures[i][t] for i in cameras],
                         'depths': [depths[i][t] for i in cameras]})
            last_t, last_xy = t, pose[:2]
            print(f'Set {len(kept)} at {t:.1f} s: robot at x {pose[0]:.2f} y {pose[1]:.2f} '
                  f'heading {math.degrees(pose[5]):.1f} deg')
    except KeyboardInterrupt:
        print('Stopped (Ctrl+C)')
    clock = state['clock']
    rtf = None
    if len(clock) > 20 and clock[-1][0] - clock[0][0] > 1.0:
        rtf = (clock[-1][1] - clock[0][1]) / (clock[-1][0] - clock[0][0])
    context = {'k': [calibration.get(i) for i in cameras], 'description': state['description'],
               'rtf': rtf}
    node.destroy_node()
    rclpy.try_shutdown()
    return kept, context


def main(argv=None):
    """Record, then save the sets."""
    from agri_ugv_phenotyping.capture import mounts
    from PIL import Image   # Pillow (python3-pil)

    parser = argparse.ArgumentParser(description='Record sets of pictures with depth from the '
                                     'dome cameras during a mission.')
    parser.add_argument('--output', required=True, help='file name without .npz')
    parser.add_argument('--cameras', type=int, nargs='+', default=list(TOP),
                        help='dome cameras (default: the top ring, 16 17 18 19)')
    parser.add_argument('--sets', type=int, default=1000, help='sets to keep at most (1000)')
    parser.add_argument('--every', type=float, default=12.0,
                        help='simulated seconds between sets at least (12)')
    parser.add_argument('--min-move', type=float, default=0.5,
                        help='metres the robot moves between sets at least (0.5)')
    parser.add_argument('--seconds', type=float, default=7200.0,
                        help='wall-clock seconds to record at most (7200)')
    args = parser.parse_args(argv)
    kept, context = record(args.cameras, args.sets, args.every, args.min_move, args.seconds)
    if not kept:
        print('No pictures: is the simulation on with camera_depth:=true, and did the mission '
              'start?')
        return
    if context['description'] is None:
        raise SystemExit('No /robot_description: the places of the cameras are unknown')
    output = Path(args.output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    depth = np.array([s['depths'] for s in kept])
    depth_mm = np.where(np.isfinite(depth) & (depth > 0), np.round(1000 * depth), 0)
    np.savez_compressed(
        f'{output}.npz', t=np.array([s['t'] for s in kept]), cameras=np.array(args.cameras),
        pictures=np.array([s['pictures'] for s in kept]),
        depth_mm=np.clip(depth_mm, 0, 65535).astype(np.uint16),
        k=np.array(context['k'], dtype=float), mounts=mounts(context['description'],
                                                             args.cameras),
        truth=np.array([s['truth'] for s in kept]), description=context['description'],
        rtf=np.nan if context['rtf'] is None else context['rtf'],
        fields=json.dumps({'pictures': 'set, camera, row, column, RGB',
                           'depth_mm': 'millimetres along the view, 0 = nothing seen',
                           'truth': 't x y z roll pitch yaw (true pose, base_footprint)',
                           'mounts': 'optical link in base_footprint (x along the view)'}))
    Image.fromarray(sheet([p for s in kept for p in s['pictures']], len(args.cameras))).save(
        f'{output}.png')
    rtf = f'real-time factor {context["rtf"]:.2f}' if context['rtf'] else 'no /clock'
    print(f'Saved {len(kept)} sets ({len(kept) * len(args.cameras)} pictures) to {output}.npz '
          f'({Path(f"{output}.npz").stat().st_size / 2**20:.1f} MB) and {output}.png; {rtf}')
