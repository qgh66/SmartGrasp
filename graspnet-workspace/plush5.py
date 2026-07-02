import argparse
import json
import sys
import termios
import tty
import select
import time
import threading
from pathlib import Path

try:
    import h5py
except ModuleNotFoundError:
    h5py = None

try:
    import numpy as np
except ModuleNotFoundError:
    np = None

try:
    from scipy.spatial.transform import Rotation as R
except ModuleNotFoundError:
    R = None

# 全局变量
record_log = []
gripper_status = 0  # 0 = open, 1 = closed
recording = True
WORKSPACE_ROOT = Path(__file__).resolve().parent
JKRC_DIR = WORKSPACE_ROOT / "jkrc"
VENDOR_DIR = WORKSPACE_ROOT / "vendor"
DEFAULT_JAKA_PYTHON = "/home/admin128/anaconda3/envs/smartgrasp310/bin/python"


def import_jkrc_backend():
    jkrc_path = str(JKRC_DIR)
    if JKRC_DIR.exists() and jkrc_path not in sys.path:
        sys.path.insert(0, jkrc_path)

    local_jaka_api = JKRC_DIR / "libjakaAPI.so"
    if local_jaka_api.exists():
        import ctypes

        ctypes.CDLL(str(local_jaka_api), mode=ctypes.RTLD_GLOBAL)

    try:
        import jkrc
    except Exception as exc:
        raise RuntimeError(
            "进入键盘控制模式需要 jkrc。已优先尝试加载本项目下的 "
            f"{JKRC_DIR / 'jkrc.so'}，但导入失败: {exc!r}。如果错误包含 "
            "Py_TPFLAGS_HAVE_GC，说明这个 jkrc.so 与当前 Python ABI 不兼容，"
            "需要换成当前 smartgrasp Python 版本匹配的 JAKA jkrc.so/wheel，"
            "或切到该 jkrc 编译时对应的 Python 环境。"
        ) from exc

    print(f"[jkrc] loaded from {getattr(jkrc, '__file__', 'unknown')}")
    return jkrc


def import_robotiq_backend():
    try:
        from robotiq_gripper_python import RobotiqGripper

        return "robotiq_gripper_python", RobotiqGripper
    except Exception as first_error:
        vendor_path = str(VENDOR_DIR)
        if vendor_path not in sys.path:
            sys.path.insert(0, vendor_path)
        try:
            from pyrobotiqgripper import RobotiqGripper

            return "pyrobotiqgripper", RobotiqGripper
        except Exception as second_error:
            raise RuntimeError(
                "Failed to import a Robotiq gripper backend. Tried robotiq_gripper_python "
                f"and local vendor/pyrobotiqgripper. First error: {first_error}; "
                f"second error: {second_error}"
            ) from second_error


def connect_gripper(comport="/dev/ttyUSB0", slave_address=9, activate=True):
    backend, robotiq_cls = import_robotiq_backend()
    if backend == "robotiq_gripper_python":
        gripper = robotiq_cls(comport=comport)
        gripper.start()
    else:
        gripper = robotiq_cls(portname=comport, slaveAddress=int(slave_address))
        if activate and hasattr(gripper, "activate"):
            gripper.activate()
    print(f"[gripper] connected via {backend} on {comport}")
    return backend, gripper


def move_gripper(gripper, backend, position, vel=30, force=30, block=True):
    position = int(max(0, min(255, position)))
    if backend == "robotiq_gripper_python":
        gripper.move(pos=position, vel=vel, force=force, block=block)
    elif backend == "pyrobotiqgripper":
        gripper.goTo(position=position, speed=vel, force=force)
    else:
        raise RuntimeError(f"Unknown gripper backend: {backend}")


def shutdown_gripper(gripper, backend):
    try:
        if backend == "robotiq_gripper_python" and hasattr(gripper, "shutdown"):
            gripper.shutdown()
        elif hasattr(gripper, "disconnect"):
            gripper.disconnect()
    except Exception:
        pass


def command_gripper(action, comport="/dev/ttyUSB0", vel=30, force=30):
    backend, gripper = connect_gripper(comport=comport)
    try:
        if action == "open":
            print("Gripper Opening")
            move_gripper(gripper, backend, 0, vel=vel, force=force)
        elif action == "close":
            print("Gripper Closing")
            move_gripper(gripper, backend, 255, vel=vel, force=force)
        else:
            raise ValueError(f"Unsupported gripper action: {action}")
    finally:
        shutdown_gripper(gripper, backend)


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Manual JAKA/Robotiq control.")
    parser.add_argument("--gripper", choices=["open", "close"], help="Only command the Robotiq gripper and exit.")
    parser.add_argument("--comport", default="/dev/ttyUSB0", help="Robotiq serial port.")
    parser.add_argument("--jaka-ip", default="192.168.1.199", help="JAKA controller IP.")
    return parser


def is_integer_value(value):
    integer_types = (int,)
    if np is not None:
        integer_types = (int, np.integer)
    return isinstance(value, integer_types)


def is_sequence_value(value):
    if isinstance(value, (list, tuple)):
        return True
    return np is not None and isinstance(value, np.ndarray)


def jaka_return_code(raw):
    if raw is None:
        return None
    if is_integer_value(raw):
        return int(raw)
    if is_sequence_value(raw) and len(raw) > 0:
        first = raw[0]
        if is_integer_value(first):
            return int(first)
    return None


def check_jaka_call(name, raw, allow_none=True):
    print(f"[jaka] {name} returned: {raw!r}")
    code = jaka_return_code(raw)
    if code is None:
        if allow_none:
            return
        raise RuntimeError(f"{name} 返回格式异常: {raw!r}")
    if code != 0:
        if name == "login" and code == -1:
            raise RuntimeError(
                "login 失败: ret=-1。当前加载的是 JAKA SDK V2.2.7，这个 SDK 的发布说明写明"
                "需要控制器版本 1.7.2_28 及以上；如果机器人控制器是 1.7.0_x 或 1.5.x，"
                "需要换回 SDK v2.1.11 或更早版本，或升级机器人控制器固件。"
            )
        raise RuntimeError(f"{name} 失败: ret={code}, raw={raw!r}")


def print_jaka_sdk_info(robot):
    for method_name in ("get_jaka_pymoudle_version", "get_sdk_version", "get_SDK_filepath"):
        method = getattr(robot, method_name, None)
        if method is None:
            continue
        try:
            print(f"[jaka] {method_name}: {method()!r}")
        except Exception as exc:
            print(f"[jaka] {method_name} failed: {exc!r}")


def get_current_tcp_position(robot):
    raw = robot.get_tcp_position()

    ret = None
    pos = None
    # Common SDK patterns:
    # 1) [ret, [x,y,z,rx,ry,rz]]
    # 2) [ret, x,y,z,rx,ry,rz]
    # 3) [x,y,z,rx,ry,rz]
    # 4) [ret] when the controller call failed
    if is_sequence_value(raw):
        raw_list = list(raw)
        if len(raw_list) == 2:
            ret, pos = raw_list[0], raw_list[1]
        elif len(raw_list) == 7:
            ret, pos = raw_list[0], raw_list[1:]
        elif len(raw_list) == 6:
            ret, pos = 0, raw_list
        elif len(raw_list) >= 2:
            ret, pos = raw_list[0], raw_list[1]
        elif len(raw_list) == 1:
            ret, pos = raw_list[0], None
    elif is_integer_value(raw):
        ret, pos = int(raw), None

    if ret is None or pos is None:
        raise RuntimeError(f"get_tcp_position 返回格式异常: raw={raw!r}")
    if int(ret) != 0:
        raise RuntimeError(f"get_tcp_position 失败: ret={ret}, raw={raw!r}")
    if len(pos) != 6:
        raise RuntimeError(f"get_tcp_position 位姿长度异常: pos={pos!r}, raw={raw!r}")
    return [float(value) for value in pos]

# 非阻塞输入
def kbhit():
    dr, dw, de = select.select([sys.stdin], [], [], 0)
    return dr != []

def getch():
    return sys.stdin.read(1)

def set_input_mode():
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    tty.setcbreak(fd)
    return old_settings

def restore_input_mode(old_settings):
    fd = sys.stdin.fileno()
    termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

# 保存为 HDF5 文件
def save_to_hdf5(filename, data_log):
    N = len(data_log)
    if N == 0:
        print("No data to save.")
        return

    if h5py is None or np is None or any("robot0_eef_quat" not in entry for entry in data_log):
        fallback_path = Path(filename).with_suffix(".jsonl")
        with fallback_path.open("w", encoding="utf-8") as f:
            for entry in data_log:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        print(f"\nOptional h5py/numpy/scipy dependency missing; data saved to {fallback_path}")
        return

    timestamps = np.zeros((N,), dtype=np.float64)
    positions = np.zeros((N, 3), dtype=np.float64)
    quaternions = np.zeros((N, 4), dtype=np.float64)
    gripper_states = np.zeros((N,), dtype=np.uint8)

    for i, entry in enumerate(data_log):
        timestamps[i] = entry['timestamp']
        positions[i] = entry['robot0_eef_pos']
        quaternions[i] = entry['robot0_eef_quat']
        gripper_states[i] = int(entry['robot0_gripper_qpos'])

    with h5py.File(filename, "w") as f:
        f.create_dataset("timestamp", data=timestamps)
        f.create_dataset("robot0_eef_pos", data=positions)
        f.create_dataset("robot0_eef_quat", data=quaternions)
        f.create_dataset("robot0_gripper_qpos", data=gripper_states)

    print("\nData saved to {}".format(filename))

# 记录线程
def record_data(robot):
    global record_log, gripper_status, recording
    if R is None:
        print("[record] scipy is not available; recording TCP Euler angles and saving JSONL on exit.")
    while recording:
        try:
            pos = get_current_tcp_position(robot)
        except RuntimeError as exc:
            print(f"[record] {exc}")
            time.sleep(0.5)
            continue

        timestamp = time.time()
        x, y, z, rx, ry, rz = pos
        entry = {
            'timestamp': timestamp,
            'robot0_eef_pos': [x, y, z],
            'robot0_eef_rpy': [rx, ry, rz],
            'robot0_gripper_qpos': bool(gripper_status)
        }
        if R is not None:
            quat = R.from_euler('xyz', [rx, ry, rz], degrees=False).as_quat()
            entry['robot0_eef_quat'] = quat.tolist()
        record_log.append(entry)
        time.sleep(0.1)

# 主函数
def main():
    global gripper_status, recording

    args = build_arg_parser().parse_args()
    if args.gripper:
        command_gripper(args.gripper, comport=args.comport)
        return

    if sys.version_info[:2] > (3, 10):
        print(
            "[jaka] current Python is "
            f"{sys.version_info.major}.{sys.version_info.minor}; "
            "if this controller only supports the Python 3.10-compatible SDK, "
            f"run keyboard control with: {DEFAULT_JAKA_PYTHON} plush5.py"
        )

    jkrc = import_jkrc_backend()

    robot = jkrc.RC(args.jaka_ip)
    try:
        print_jaka_sdk_info(robot)
        check_jaka_call("login", robot.login())
        check_jaka_call("power_on", robot.power_on())
        check_jaka_call("enable_robot", robot.enable_robot())

        current_pos = get_current_tcp_position(robot)
        print(f"[jaka] current TCP: {current_pos}")
    except Exception:
        try:
            robot.logout()
        except Exception:
            pass
        raise

    gripper_backend, gripper = connect_gripper(comport=args.comport)
    x, y, z, rx, ry, rz = current_pos
    step = 20.0
    ABS = 0

    record_thread = threading.Thread(target=record_data, args=(robot,))
    record_thread.start()

    print("Use W/A/S/D/Q/E to move. Press ESC to quit.")
    old_settings = set_input_mode()

    try:
        while True:
            if not kbhit():
                time.sleep(0.05)
                continue

            key = getch()
            moved = False

            if key == '\x1b':  # ESC
                break
            elif key == 'a':
                y += step
                moved = True
            elif key == 'd':
                y -= step
                moved = True
            elif key == 's':
                x -= step
                moved = True
            elif key == 'w':
                x += step
                moved = True
            elif key == 'q':
                z += step
                moved = True
            elif key == 'e':
                z -= step
                moved = True
            elif key == 'z':
                print("Gripper Closing")
                move_gripper(gripper, gripper_backend, 255, vel=30, force=30)
                gripper_status = 1
            elif key == 'c':
                print("Gripper Opening")
                move_gripper(gripper, gripper_backend, 0, vel=30, force=30)
                gripper_status = 0

            if moved:
                new_pose = [x, y, z, rx, ry, rz]
                print("\nMoving to: x={}, y={}, z={}".format(x, y, z))
                ret = robot.linear_move_extend(new_pose, ABS, True, 60, 60, 1)
                code = jaka_return_code(ret)
                if code not in (None, 0):
                    print("Move failed: ret={}, raw={}".format(code, ret))
                else:
                    print("Move done.")
                termios.tcflush(sys.stdin, termios.TCIFLUSH)

    finally:
        restore_input_mode(old_settings)
        recording = False
        record_thread.join()
        robot.logout()
        shutdown_gripper(gripper, gripper_backend)
        print("Control ended. Bye!")

        save_to_hdf5("record_log.h5", record_log)

if __name__ == "__main__":
    main()
