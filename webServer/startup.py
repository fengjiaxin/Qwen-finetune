import multiprocessing as mp
import subprocess
import sys


def run_webui(url: str, host: str, port: int):
    cmd = ["streamlit", "run", "webui.py",
           "--server.address", host,
           "--server.port", str(port),
           "--theme.base", "light",
           "--theme.primaryColor", "#165dff",
           "--theme.secondaryBackgroundColor", "#f5f5f5",
           "--theme.textColor", "#000000",
           ]
    cmd += ["--"]
    cmd += ["--url", url]
    p = subprocess.Popen(cmd)
    p.wait()


if __name__ == '__main__':  # 必须将进程过程放到main函数中去
    DEFAULT_BIND_HOST = "0.0.0.0" if sys.platform != "win32" else "127.0.0.1"

    PORT1 = 8051
    URL1 = "http://10.200.101.150:8080/stream"

    PORT2 = 8052
    URL2 = "http://10.200.101.150:8081/stream"
    process_list = [mp.Process(target=run_webui, args=(URL1, DEFAULT_BIND_HOST, PORT1)),
                    mp.Process(target=run_webui, args=(URL2, DEFAULT_BIND_HOST, PORT2))]
    for p in process_list:
        p.start()
    for p in process_list:
        p.join()
