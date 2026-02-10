#!/usr/bin/env python3
import sys
import argparse
from g_trac import consts
from g_trac.roles import anchor, worker, client


def main():
    if len(sys.argv) < 2:
        print("Usage: python run.py [anchor|worker|client] ...")
        sys.exit(1)

    role = sys.argv[1].lower()

    if role == "anchor":
        port = int(sys.argv[2]) if len(sys.argv) > 2 else 5000
        # Auto-detect layers logic can go here or inside anchor.start_anchor
        anchor.start_anchor(port, consts.MODEL_LAYERS)

    elif role == "worker":
        if len(sys.argv) < 12:
            print(
                "Usage: worker <MY_IP> <MY_PORT> <CPU> <FAIL> <TRUST> <ID> <ANCHOR_IP> <ANCHOR_PORT> <L_START> <L_END>")
            sys.exit(1)

        class WorkerArgs:
            pass

        a = WorkerArgs()
        a.ip = sys.argv[2]
        a.port = int(sys.argv[3])
        a.cpu_load = int(sys.argv[4])
        a.fail_rate = float(sys.argv[5])
        a.trust0 = float(sys.argv[6])
        a.id = sys.argv[7]
        a.anchor_ip = sys.argv[8]
        a.anchor_port = int(sys.argv[9])
        a.layer_start = int(sys.argv[10])
        a.layer_end = int(sys.argv[11])

        # Pass the parsed arguments to the logic function
        worker.start_worker(a)

    elif role == "client":
        if len(sys.argv) < 4:
            print("Usage: client <ANCHOR_IP> <ANCHOR_PORT> [MODE]")
            sys.exit(1)

        a_ip = sys.argv[2]
        a_port = int(sys.argv[3])
        mode = sys.argv[4] if len(sys.argv) > 4 else "g-trac"

        client.start_client(a_ip, a_port, mode)
    else:
        print(f"Unknown role: {role}")
        sys.exit(1)


if __name__ == "__main__":
    main()