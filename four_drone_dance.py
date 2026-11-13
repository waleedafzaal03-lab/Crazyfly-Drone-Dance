#!/usr/bin/env python3
"""Four Crazyflies with Flow decks: a small dance in separate flight areas.

Addresses are preconfigured for Drone 1 (...01) through Drone 4 (...04).
Use one Crazyradio, the SAME configured channel/bitrate on all four drones,
and the 2 m square arrangement shown in floor_layout.png. All noses face FRONT.

Examples (80 is an EXAMPLE channel; use the channel from your own cfclient):
  python four_drone_dance.py --plan
  python four_drone_dance.py --channel 80 --identify
  python four_drone_dance.py --channel 80 --check
  python four_drone_dance.py --channel 80 --hover --only 1
  python four_drone_dance.py --channel 80 --hover
  python four_drone_dance.py --channel 80 --dance

IMPORTANT: Flow provides relative odometry. Neither the planned separation nor
the checks using estimated positions guarantee physical collision avoidance.
This program cannot detect unobserved Flow drift or verify the real starting
positions. It does not measure the other drones. Use the supplied spacing and
floor conditions; inspect a short hover before trying the dance.

Ctrl+C requests a straight landing for all started drones while links work.
No LEDs, external positioning, ROS, Crazyswarm, or gamepad are required.
Tested with simulated hardware only; not flight-tested on physical Crazyflies.
"""

import argparse
import itertools
import math
import signal
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from threading import Event, Lock

import cflib.crtp
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.swarm import CachedCfFactory, Swarm


# Physical layout only: X points to FRONT, Y points LEFT. These offsets are
# added to local Flow estimates for a conservative ESTIMATED separation check.
# They are not shared/global coordinates supplied by the Flow decks.
DRONES = {
    1: ('E7E7E7E701', (1.0, 1.0)),    # front left
    2: ('E7E7E7E702', (1.0, -1.0)),   # front right
    3: ('E7E7E7E703', (-1.0, 1.0)),   # back left
    4: ('E7E7E7E704', (-1.0, -1.0)),  # back right
}
HEIGHT = 0.50
EXCURSION = 0.20
MOVE_SECONDS = 3.0
SETTLE_SECONDS = 0.40
LAND_SECONDS = 3.0
MIN_PREFLIGHT_VBAT = 3.90  # resting voltage; start with charged batteries
MIN_FLIGHT_VBAT = 3.30    # sustained under load for 0.5 s
MAX_LOCAL_XY = 0.45      # estimated boundary, not a physical fence
MAX_HEIGHT = 0.90
MIN_ESTIMATED_SEPARATION = 1.20
TELEMETRY_MAX_AGE = 1.0
FAULT_BITS = (1 << 5) | (1 << 6) | (1 << 7) | (1 << 11)
POSITION_LOGS = {
    'stateEstimate.x': 'float', 'stateEstimate.y': 'float',
    'stateEstimate.z': 'float', 'pm.vbat': 'float',
    'range.zrange': 'uint16_t', 'supervisor.info': 'uint16_t',
}
VARIANCE_LOGS = {name: 'float' for name in ('kalman.varPX', 'kalman.varPY', 'kalman.varPZ')}


class AbortFlight(RuntimeError):
    pass


@dataclass
class DroneState:
    number: int
    uri: str
    home: tuple
    lock: Lock = field(default_factory=Lock)
    data: dict = field(default_factory=dict)
    last_position: float = 0.0
    last_variance: float = 0.0
    variances: dict = field(default_factory=lambda: {
        name: deque(maxlen=10) for name in VARIANCE_LOGS})
    logs: list = field(default_factory=list)
    callbacks: list = field(default_factory=list)
    armed_attempt: bool = False
    started: bool = False
    low_battery_since: object = None
    range_mismatch_since: object = None
    tracking_error_since: object = None
    planned: tuple = (0.0, 0.0)
    cleanup_errors: list = field(default_factory=list)

    def snapshot(self):
        with self.lock:
            return dict(self.data), self.last_position, self.last_variance


@dataclass
class Session:
    states: list
    abort: Event = field(default_factory=Event)
    reason_lock: Lock = field(default_factory=Lock)
    reason: str = ''
    inflight: bool = False

    def request_abort(self, reason):
        with self.reason_lock:
            if not self.reason:
                self.reason = reason
                print(f'\nLANDING REQUESTED: {reason}', file=sys.stderr, flush=True)
        self.abort.set()


def deck_present(cf, name):
    if cf.param.toc.get_element_by_complete_name(name) is None:
        return False
    return int(cf.param.get_value(name)) != 0


def choreography(number):
    """Relative XY moves. Each path stays within +/- 0.20 m of its own pad."""
    home_x, home_y = DRONES[number][1]
    sx, sy = math.copysign(1, home_x), math.copysign(1, home_y)
    a = EXCURSION
    return [
        ('Fan outward', sx * a, sy * a),
        ('Return to pad centre', -sx * a, -sy * a),
        ('Sway left', 0.0, a),
        ('Sway right', 0.0, -2 * a),
        ('Return to pad centre', 0.0, a),
        ('Sway forward', a, 0.0),
        ('Sway backward', -2 * a, 0.0),
        ('Return to pad centre', a, 0.0),
    ]


def validate_plan():
    for number in DRONES:
        x = y = 0.0
        for _name, dx, dy in choreography(number):
            x, y = x + dx, y + dy
            if max(abs(x), abs(y)) > EXCURSION + 1e-9:
                raise ValueError('Choreography leaves its planned area.')
        if math.hypot(x, y) > 1e-9:
            raise ValueError('Choreography does not return to its pad.')
    # All nominal horizontal path boxes are disjoint, even if stages are not
    # perfectly synchronized. Actual odometry/tracking errors are not bounded
    # by this mathematical plan check.
    gaps = []
    for (i, (_, a)), (j, (_, b)) in itertools.combinations(DRONES.items(), 2):
        gap = math.hypot(max(0.0, abs(a[0] - b[0]) - 2 * EXCURSION),
                         max(0.0, abs(a[1] - b[1]) - 2 * EXCURSION))
        if gap < 1.5:
            raise ValueError(f'Planned path boxes for drones {i} and {j} are too close.')
        gaps.append(gap)
    return min(gaps)


def inspect_telemetry(session):
    now = time.monotonic()
    estimates = []
    for state in session.states:
        data, position_time, variance_time = state.snapshot()
        if now - position_time > TELEMETRY_MAX_AGE:
            raise AbortFlight(f'Drone {state.number}: position telemetry is stale.')
        if now - variance_time > TELEMETRY_MAX_AGE:
            raise AbortFlight(f'Drone {state.number}: estimator telemetry is stale.')
        if not all(math.isfinite(float(value)) for value in data.values()):
            raise AbortFlight(f'Drone {state.number}: invalid telemetry.')
        if int(data['supervisor.info']) & FAULT_BITS:
            raise AbortFlight(f'Drone {state.number}: supervisor reports a fault/crash.')
        if not session.inflight:
            continue
        x, y, z = (data[f'stateEstimate.{axis}'] for axis in ('x', 'y', 'z'))
        if max(abs(x), abs(y)) > MAX_LOCAL_XY:
            raise AbortFlight(f'Drone {state.number}: estimated XY boundary exceeded.')
        if not -0.15 <= z <= MAX_HEIGHT:
            raise AbortFlight(f'Drone {state.number}: estimated height outside limits.')
        if any(data[name] < 0 or data[name] > 0.10 for name in VARIANCE_LOGS):
            raise AbortFlight(f'Drone {state.number}: high estimator uncertainty.')
        if data['pm.vbat'] < MIN_FLIGHT_VBAT:
            if state.low_battery_since is None:
                state.low_battery_since = now
            elif now - state.low_battery_since >= 0.5:
                raise AbortFlight(f'Drone {state.number}: low flight battery.')
        else:
            state.low_battery_since = None
        # Once above the near-ground region, compare the range sensor with
        # estimated altitude. This is only a sanity check, not another-drone
        # detector. A floor discontinuity can also cause a mismatch.
        mismatch = z > 0.25 and abs(data['range.zrange'] / 1000.0 - z) > 0.20
        if mismatch:
            if state.range_mismatch_since is None:
                state.range_mismatch_since = now
            elif now - state.range_mismatch_since > 0.5:
                raise AbortFlight(f'Drone {state.number}: range/altitude disagreement.')
        else:
            state.range_mismatch_since = None
        estimates.append((state.number, state.home[0] + x, state.home[1] + y))
    for (i, x1, y1), (j, x2, y2) in itertools.combinations(estimates, 2):
        if math.hypot(x1 - x2, y1 - y2) < MIN_ESTIMATED_SEPARATION:
            raise AbortFlight(f'Drones {i}/{j}: estimated horizontal separation too small.')


def wait_seconds(seconds, session, monitor=False, ignore_abort=False):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if not ignore_abort and session.abort.is_set():
            raise AbortFlight(session.reason or 'Stop requested.')
        if monitor:
            inspect_telemetry(session)
        left = deadline - time.monotonic()
        # During landing a requested abort must not skip the descent interval.
        if ignore_abort:
            time.sleep(min(0.05, max(0.0, left)))
        else:
            session.abort.wait(min(0.05, max(0.0, left)))
    if not ignore_abort and session.abort.is_set():
        raise AbortFlight(session.reason or 'Stop requested.')
    if monitor:
        inspect_telemetry(session)


def wait_for_parameters(scf, session, state):
    deadline = time.monotonic() + 30
    while not scf.is_params_updated():
        if time.monotonic() > deadline:
            raise RuntimeError(f'Drone {state.number}: parameter download timed out.')
        wait_seconds(0.05, session)


def prepare(scf, session, state):
    try:
        wait_for_parameters(scf, session, state)
        cf = scf.cf
        if not (deck_present(cf, 'deck.bcFlow2') or deck_present(cf, 'deck.bcFlow')):
            raise RuntimeError(f'Drone {state.number}: Flow deck not detected.')
        cf.param.set_value('stabilizer.estimator', '2')
        wait_seconds(0.5, session)
        # Give each drone its own local origin. Place all real drones facing
        # FRONT; no attempt is made to pretend these are absolute room poses.
        for axis in ('X', 'Y', 'Z', 'Yaw'):
            cf.param.set_value(f'kalman.initial{axis}', '0')
        cf.param.set_value('kalman.resetEstimation', '1')
        time.sleep(0.1)
        cf.param.set_value('kalman.resetEstimation', '0')
        wait_seconds(0.5, session)

        def data_callback(_timestamp, data, config):
            with state.lock:
                state.data.update(data)
                if config.name == 'Dance position':
                    state.last_position = time.monotonic()
                else:
                    state.last_variance = time.monotonic()
                    for name in VARIANCE_LOGS:
                        state.variances[name].append(float(data[name]))

        def log_error(_config, message):
            session.request_abort(f'Drone {state.number}: logging error: {message}')

        def link_lost(_uri, message):
            session.request_abort(f'Drone {state.number}: radio connection lost: {message}')

        state.callbacks = [data_callback, log_error, link_lost]
        cf.connection_lost.add_callback(link_lost)
        for name, period, variables in (
                ('Dance position', 100, POSITION_LOGS),
                ('Dance variance', 200, VARIANCE_LOGS)):
            config = LogConfig(name=name, period_in_ms=period)
            for variable, kind in variables.items():
                config.add_variable(variable, kind)
            cf.log.add_config(config)
            config.data_received_cb.add_callback(data_callback)
            config.error_cb.add_callback(log_error)
            state.logs.append(config)
            config.start()

        deadline = time.monotonic() + 20
        while True:
            if session.abort.is_set():
                raise AbortFlight(session.reason)
            with state.lock:
                history = [list(h) for h in state.variances.values()]
            data, position_time, variance_time = state.snapshot()
            complete = all(name in data for name in (*POSITION_LOGS, *VARIANCE_LOGS))
            stable = all(len(h) == 10 and all(math.isfinite(v) for v in h)
                         and min(h) >= 0 and max(h) < 0.05
                         and max(h) - min(h) < 0.001 for h in history)
            fresh = (time.monotonic() - position_time < 0.5
                     and time.monotonic() - variance_time < 0.5)
            if complete and stable and fresh:
                break
            if time.monotonic() > deadline:
                raise RuntimeError(f'Drone {state.number}: estimator/telemetry did not settle.')
            wait_seconds(0.05, session)

        if data['pm.vbat'] < MIN_PREFLIGHT_VBAT:
            raise RuntimeError(f'Drone {state.number}: battery {data["pm.vbat"]:.2f} V; charge to >= 3.90 V.')
        if int(data['supervisor.info']) & FAULT_BITS:
            raise RuntimeError(f'Drone {state.number}: supervisor reports a fault.')
        if max(abs(data['stateEstimate.x']), abs(data['stateEstimate.y'])) > 0.10:
            raise RuntimeError(f'Drone {state.number}: local origin unstable; keep it still on the floor.')
        if abs(data['stateEstimate.z']) > 0.15:
            raise RuntimeError(f'Drone {state.number}: start on the floor, not on a raised platform.')
        print(f'Drone {state.number} READY: {state.uri}, battery {data["pm.vbat"]:.2f} V', flush=True)
    except Exception as exc:
        session.request_abort(str(exc))
        raise


def identify(scf, session, state):
    wait_for_parameters(scf, session, state)
    cf = scf.cf
    previous = cf.param.get_value('led.bitmask')
    print(f'Drone {state.number}: flashing onboard lights ({DRONES[state.number][0]})', flush=True)
    try:
        for _ in range(3):
            cf.param.set_value('led.bitmask', '255')
            wait_seconds(0.25, session)
            cf.param.set_value('led.bitmask', '0')
            wait_seconds(0.25, session)
    finally:
        cf.param.set_value('led.bitmask', previous)
        time.sleep(0.1)


def arm(scf, session, state):
    try:
        if session.abort.is_set():
            raise AbortFlight(session.reason)
        state.armed_attempt = True
        scf.cf.supervisor.send_arming_request(True)
        deadline = time.monotonic() + 3
        while True:
            data, _p, _v = state.snapshot()
            if int(data.get('supervisor.info', 0)) & (1 << 1):
                return
            if time.monotonic() >= deadline:
                raise RuntimeError(f'Drone {state.number}: arming not confirmed.')
            wait_seconds(0.05, session, monitor=True)
    except Exception as exc:
        session.request_abort(str(exc))
        raise


def takeoff(scf, session, state):
    if session.abort.is_set():
        return
    state.started = True
    scf.cf.high_level_commander.takeoff(HEIGHT, 3.0, yaw=0.0)


def move(scf, session, state, stage):
    if session.abort.is_set():
        return
    _name, dx, dy = choreography(state.number)[stage]
    scf.cf.high_level_commander.go_to(dx, dy, 0.0, 0.0,
                                      MOVE_SECONDS, relative=True)
    state.planned = (state.planned[0] + dx, state.planned[1] + dy)


def verify_hover(session):
    for state in session.states:
        data, _p, _v = state.snapshot()
        z = data['stateEstimate.z']
        if not 0.35 <= z <= 0.65:
            raise AbortFlight(f'Drone {state.number}: did not reach the expected hover height.')


def verify_target(session):
    for state in session.states:
        data, _p, _v = state.snapshot()
        error = math.hypot(data['stateEstimate.x'] - state.planned[0],
                           data['stateEstimate.y'] - state.planned[1])
        if error > 0.20:
            raise AbortFlight(f'Drone {state.number}: reported position is too far from the planned target.')
    verify_hover(session)


def land(scf, session, state):
    if not state.started:
        return
    data, _p, _v = state.snapshot()
    # Do not issue a flight manoeuvre to a drone already reported crashed or
    # tumbled; stop it while the other drones land in their own flight areas.
    info = int(data.get('supervisor.info', 0))
    if info & FAULT_BITS:
        try:
            scf.cf.high_level_commander.stop()
        except Exception as exc:
            state.cleanup_errors.append(str(exc))
        return
    try:
        scf.cf.high_level_commander.land(0.0, LAND_SECONDS, yaw=0.0)
    except Exception as exc:
        state.cleanup_errors.append(f'Landing command: {exc}')
        print(f'Drone {state.number}: landing command failed: {exc}', file=sys.stderr)


def stop_and_close_logs(scf, session, state):
    def attempt(action, label):
        try:
            action()
        except Exception as exc:
            state.cleanup_errors.append(f'{label}: {exc}')
            print(f'Drone {state.number}: {label} failed: {exc}', file=sys.stderr)

    if state.started:
        attempt(scf.cf.high_level_commander.stop, 'motor stop')
        state.started = False
    if state.armed_attempt:
        attempt(lambda: scf.cf.supervisor.send_arming_request(False), 'disarming')
        state.armed_attempt = False
        time.sleep(0.15)
    for config in state.logs:
        attempt(config.delete, 'log cleanup')
    state.logs.clear()


def plan_text():
    gap = validate_plan()
    print('PRECONFIGURED ADDRESSES AND REQUIRED PHYSICAL STARTING POSITIONS')
    for number, (address, home) in DRONES.items():
        print(f'Drone {number}: 0x{address}; home X={home[0]:+.1f} m, Y={home[1]:+.1f} m')
    print('All noses face FRONT (+X). Adjacent pads are 2 m apart, centre to centre.')
    print(f'Height {HEIGHT:.2f} m; local XY excursion +/-{EXCURSION:.2f} m.')
    print(f'Nominal path-box gap, allowing any phase mismatch: {gap:.2f} m.')
    for stage, (name, _dx, _dy) in enumerate(choreography(1), 1):
        print(f'{stage}. {name}: {MOVE_SECONDS:.1f} s + {SETTLE_SECONDS:.1f} s settling')
    print('About 38 s airborne. --hover is about 11 s airborne.')
    print('No planned path crossings, position swaps, yaw turns, or stacked altitudes.')
    print('Actual collision avoidance is NOT guaranteed with Flow-only estimates.')


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    modes = parser.add_mutually_exclusive_group(required=True)
    for mode in ('plan', 'identify', 'check', 'hover', 'dance'):
        modes.add_argument(f'--{mode}', action='store_const', dest='mode', const=mode)
    parser.add_argument('--channel', type=int, help='Common radio channel from YOUR cfclient URI')
    parser.add_argument('--rate', choices=('250K', '1M', '2M'), default='2M')
    parser.add_argument('--radio', type=int, default=0)
    parser.add_argument('--only', type=int, nargs='+', choices=tuple(DRONES),
                        help='Optional drone numbers; otherwise all four are selected')
    args = parser.parse_args()
    validate_plan()
    if args.mode == 'plan':
        plan_text()
        return 0
    if args.channel is None or not 0 <= args.channel <= 125:
        parser.error('Provide --channel using YOUR common cfclient radio channel (80 is only an example).')
    if args.radio < 0:
        parser.error('--radio must be a nonnegative Crazyradio index.')
    numbers = args.only or list(DRONES)
    if len(set(numbers)) != len(numbers):
        parser.error('Do not repeat a drone number in --only.')
    states = [DroneState(n, f'radio://{args.radio}/{args.channel}/{args.rate}/{DRONES[n][0]}', DRONES[n][1])
              for n in sorted(numbers)]
    session = Session(states)
    arguments = {s.uri: [session, s] for s in states}
    old_handler = signal.signal(signal.SIGINT,
                                lambda _sig, _frame: session.request_abort('Ctrl+C pressed'))
    cflib.crtp.init_drivers()
    failed = False
    try:
        print('Close cfclient. Keep every selected drone still at its marked starting pad.')
        print('Connecting to ' + ', '.join(f'Drone {s.number}' for s in states), flush=True)
        with Swarm([s.uri for s in states], factory=CachedCfFactory(rw_cache='./cache')) as swarm:
            try:
                if args.mode == 'identify':
                    swarm.sequential(identify, args_dict=arguments)
                    return 0
                swarm.parallel_safe(prepare, args_dict=arguments)
                inspect_telemetry(session)
                if args.mode == 'check':
                    print('All selected drones passed preflight. No arming or motor commands sent.')
                    return 0
                print('All preflights passed. Takeoff countdown:', flush=True)
                for remaining in (5, 4, 3, 2, 1):
                    print(remaining, flush=True)
                    wait_seconds(1.0, session, monitor=True)
                swarm.parallel_safe(arm, args_dict=arguments)
                # Verify all are armed and telemetry is current before ANY takeoff.
                inspect_telemetry(session)
                session.inflight = True
                swarm.parallel_safe(takeoff, args_dict=arguments)
                wait_seconds(3.0 + SETTLE_SECONDS, session, monitor=True)
                verify_hover(session)
                wait_seconds(2.0 if args.mode == 'dance' else 4.0, session, monitor=True)
                if args.mode == 'dance':
                    for stage in range(len(choreography(1))):
                        print(f'Dance {stage + 1}/8: {choreography(1)[stage][0]}', flush=True)
                        move_args = {uri: [*values, stage] for uri, values in arguments.items()}
                        swarm.parallel_safe(move, args_dict=move_args)
                        # Full duration is measured AFTER every command has been
                        # dispatched, so no following go_to overlaps this one.
                        wait_seconds(MOVE_SECONDS + SETTLE_SECONDS, session, monitor=True)
                        verify_target(session)
                    wait_seconds(1.0, session, monitor=True)
            except Exception as exc:
                failed = True
                session.request_abort(str(exc))
                cause = exc
                while cause.__cause__ is not None:
                    cause = cause.__cause__
                print(f'Flight/check stopped: {cause}', file=sys.stderr)
            finally:
                if any(s.started for s in states):
                    print('Landing all started drones in their separate flight areas...', flush=True)
                    swarm.parallel_safe(land, args_dict=arguments)
                    wait_seconds(LAND_SECONDS + 0.8, session, ignore_abort=True)
                swarm.parallel_safe(stop_and_close_logs, args_dict=arguments)
        if failed or session.abort.is_set() or any(s.cleanup_errors for s in states):
            return 1
        print('Complete; links closed.')
        return 0
    finally:
        signal.signal(signal.SIGINT, old_handler)


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception as exc:
        cause = exc
        while cause.__cause__ is not None:
            cause = cause.__cause__
        print(f'Error: {cause}', file=sys.stderr)
        sys.exit(1)
