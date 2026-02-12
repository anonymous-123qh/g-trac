#!/usr/bin/env python3
import sys
import argparse
import traceback
print("DEBUG: run.py script started...", flush=True)
print(f"DEBUG: Raw sys.argv: {sys.argv}", flush=True)
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="G-TRAC Runner")
    subparsers = parser.add_subparsers(dest="role", required=True)

    #Client Args
    client_parser = subparsers.add_parser("client")
    client_parser.add_argument("anchor_ip", type=str)
    client_parser.add_argument("anchor_port", type=int)
    client_parser.add_argument("mode", type=str, nargs="?", default="g-trac")

    #Worker Args
    worker_parser = subparsers.add_parser("worker")
    worker_parser.add_argument("ip", type=str)
    worker_parser.add_argument("port", type=int)
    worker_parser.add_argument("cpu_load", type=int)
    worker_parser.add_argument("fail_rate", type=float)
    worker_parser.add_argument("trust0", type=float)
    worker_parser.add_argument("id", type=str)
    worker_parser.add_argument("anchor_ip", type=str)
    worker_parser.add_argument("anchor_port", type=int)
    worker_parser.add_argument("layer_start", type=int)
    worker_parser.add_argument("layer_end", type=int)

    #Anchor Args
    anchor_parser = subparsers.add_parser("anchor")
    anchor_parser.add_argument("port", type=int)
    anchor_parser.add_argument("--layers", type=int, default=36, help="Total model layers")

    try:
        print("DEBUG: Parsing arguments...", flush=True)
        args = parser.parse_args()
        print(f"DEBUG: Parse successful. Mode={args.role}", flush=True)
    except SystemExit as e:
        print(f"DEBUG: Argparse failed and tried to exit! Code: {e}", flush=True)
        print("DEBUG: Did you provide all required arguments?", flush=True)
        sys.exit(1)
    except Exception as e:
        print(f"DEBUG: Unexpected error parsing args: {e}", flush=True)
        traceback.print_exc()
        sys.exit(1)

    #LAZY IMPORT
    if args.role == "client":
        print("Starting Client...", flush=True)
        try:
            #flat import first (Common on phones)
            try:
                import client
            except ImportError:
                #nested import (Original structure)
                from g_trac.roles import client

            client.start_client(args.anchor_ip, args.anchor_port, args.mode)

        except ImportError as e:
            print(f"CRITICAL ERROR: Could not import client.py.", flush=True)
            print(f"Detail: {e}", flush=True)
            print("Check: Is client.py in the same folder as run.py?", flush=True)
            traceback.print_exc()
        except Exception as e:
            print(f"CRITICAL ERROR inside client: {e}", flush=True)
            traceback.print_exc()

    elif args.role == "worker":
        print("Starting Worker...")
        try:
            try:
                import worker
            except ImportError:
                from g_trac.roles import worker
            worker.start_worker(args)
        except Exception as e:
            print(f"Error starting worker: {e}")



    elif args.role == "anchor":
        print("Starting Anchor...")
        try:
            try:
                import anchor
            except ImportError:
                from g_trac.roles import anchor
            anchor.start_anchor(args.port, args.layers)
        except Exception as e:
            print(f"Error starting anchor: {e}")