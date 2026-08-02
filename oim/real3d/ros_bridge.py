"""Two-process fallback bridge: run this in the ROS env; run the planner
(`pusht_real.py --socket`) in the JAX env.

Use this only when ROS 2 (`rclpy`) and CUDA JAX will not share one
environment (the pixi/RoboStack single-env is the preferred path -- see
`oim/real3d/pixi.toml`). This process holds the real `Ros2Interface` (all the
ROS I/O: `/joint_states`, the FoundationPose TF, the velocity publisher, the
watchdog, the static world->base transform) and serves its `read_state` /
`send_velocity` over a localhost socket to a `SocketInterface` in the planner
process. It re-introduces OI-MPPI's planner/interface split, but over a plain
socket instead of zerorpc.

    ROS env:   python -m oim.real3d.ros_bridge --port 5599
    JAX env:   python examples/pusht_real.py --socket --bridge-port 5599 ...
"""

import argparse
import socket

from oim.real3d.interface import Ros2Interface, recv_msg, send_msg


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=5599)
    p.add_argument("--object-frame", default="fp_object_pose",
                   help="TF frame of the tracked object (e.g. sam6d_object)")
    p.add_argument("--velocity-topic", default="velocity_controller/commands")
    args = p.parse_args()

    interface = Ros2Interface(
        object_frame=args.object_frame,
        velocity_command_topic=args.velocity_topic,
    )

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((args.host, args.port))
    srv.listen(1)
    print(f"[bridge] waiting for the planner on {args.host}:{args.port} ...")
    conn, addr = srv.accept()
    print(f"[bridge] planner connected from {addr}")

    try:
        while True:
            op, payload = recv_msg(conn)
            if op == "read":
                # Surface a not-ready state as an error the client can raise,
                # rather than crashing the bridge.
                try:
                    send_msg(conn, interface.read_state())
                except Exception as e:  # noqa: BLE001
                    send_msg(conn, ("err", str(e)))
            elif op == "cmd":
                interface.send_velocity(payload)
                send_msg(conn, "ok")
            elif op == "time":
                send_msg(conn, interface.time())
            elif op == "close":
                break
    except (ConnectionError, EOFError):
        print("[bridge] planner disconnected")
    finally:
        conn.close()
        interface.close()  # publishes a zero-velocity command on shutdown
        print("[bridge] stopped")


if __name__ == "__main__":
    main()
