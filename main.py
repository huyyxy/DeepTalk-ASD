from dotenv import load_dotenv

try:
    dotenv_path = '.env'  # 请将路径替换为正确的.env文件路径
    load_dotenv(dotenv_path)
except FileNotFoundError:
    pass

from deeptalk_logger import DeepTalkLogger
import queue
from websocket_handlers import server
from worker import proc
import traceback
from concurrent.futures import ThreadPoolExecutor



logger = DeepTalkLogger(__name__)

if __name__ == "__main__":
    logger.info(f"====== Start Main ======")

    worker_thread = ThreadPoolExecutor(1, thread_name_prefix="worker")

    date_queue = queue.Queue()
    score_queue = queue.Queue()

    worker_thread.submit(proc, date_queue)

    try:
        server.start(date_queue, score_queue)
    except KeyboardInterrupt:
        logger.info(f"=== 程序终止: KeyboardInterrupt ===")
    except Exception:
        logger.info(f"=== 程序终止: Exception ===")
        traceback.print_exc()
    finally:
        worker_thread.shutdown(wait=True, cancel_futures=True)
        logger.info(f"=== 资源清理完毕，退出程序。 ===")
