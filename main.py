from dotenv import load_dotenv

try:
    dotenv_path = '.env'  # 请将路径替换为正确的.env文件路径
    load_dotenv(dotenv_path)
except FileNotFoundError:
    pass

from deeptalk_logger import DeepTalkLogger
from websocket_handlers import server
import traceback



logger = DeepTalkLogger(__name__)

if __name__ == "__main__":
    logger.info(f"====== Start Main ======")
    try:
        server.start()
    except KeyboardInterrupt:
        logger.info(f"=== 程序终止: KeyboardInterrupt ===")
    except Exception:
        logger.info(f"=== 程序终止: Exception ===")
        traceback.print_exc()
    finally:
        logger.info(f"=== 资源清理完毕，退出程序。 ===")
