"""Run training and expose its exit status to the Cloud Studio browser watcher."""

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import subprocess
import sys
import threading
import uuid


class StatusHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.partition("?")[0] != "/":
            self.send_error(404)
            return

        exit_code = self.server.exit_code
        if exit_code is None:
            state = "running"
        else:
            state = "succeeded" if exit_code == 0 else "failed"

        body = json.dumps({
            "protocol": "dgekt-cloudstudio-training-v1",
            "run_id": self.server.run_id,
            "state": state,
            "exit_code": exit_code,
        }).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        # Keep periodic browser checks out of the training terminal.
        pass


class StatusServer(ThreadingHTTPServer):
    def __init__(self, port):
        self.run_id = uuid.uuid4().hex
        self.exit_code = None
        super().__init__(("0.0.0.0", port), StatusHandler)


def main():
    parser = argparse.ArgumentParser(
        description="运行训练命令，并把完成状态发布到 Cloud Studio 的私有端口预览。",
        epilog="命令退出后，脚本继续提供状态，直至云端关闭或按 Ctrl+C。"
               "训练期间按 Ctrl+C 会中断任务并关闭状态页。",
    )
    parser.add_argument("--port", type=int, default=8765, help="状态页端口，默认 8765")
    parser.add_argument("command", nargs=argparse.REMAINDER, help="在 -- 后面填写训练命令")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port 必须在 1 至 65535 之间")
    command = args.command
    if command[:1] == ["--"]:
        command = command[1:]
    if not command:
        parser.error("请在 -- 后填写训练命令")

    try:
        server = StatusServer(args.port)
    except OSError as error:
        parser.exit(1, f"无法启动端口 {args.port} 的状态页：{error}\n")

    with server:
        listener = threading.Thread(target=server.serve_forever, daemon=True)
        listener.start()
        try:
            print(f"请在 Cloud Studio 的端口面板将 {args.port} 设为「仅自己可见」，"
                  "再在浏览器中打开预览。", flush=True)
            try:
                exit_code = subprocess.run(command, check=False).returncode
            except OSError as error:
                print(f"无法启动训练命令：{error}", file=sys.stderr, flush=True)
                exit_code = 127
            if exit_code < 0:
                exit_code = 128 - exit_code
            server.exit_code = exit_code

            print(f"训练命令已退出，退出码 {exit_code}。状态页会保留供浏览器检查。", flush=True)
            threading.Event().wait()
        except KeyboardInterrupt:
            return 130 if server.exit_code is None else server.exit_code
        finally:
            server.shutdown()


if __name__ == "__main__":
    sys.exit(main())
