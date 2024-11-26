import multiprocessing as mp
import subprocess
from config import (DEFAULT_BIND_HOST, PORT)


def run_webui(host: str, port: int):
    cmd = ["streamlit", "run", "webui.py",
           "--server.address", host,
           "--server.port", str(port),
           "--theme.base", "light",
           "--theme.primaryColor", "#165dff",
           "--theme.secondaryBackgroundColor", "#f5f5f5",
           "--theme.textColor", "#000000",
           ]
    p = subprocess.Popen(cmd)
    p.wait()


if __name__ == '__main__':  # 必须将进程过程放到main函数中去
    p1 = mp.Process(target=run_webui, args=(DEFAULT_BIND_HOST, PORT))
    p1.start()
    p1.join()
