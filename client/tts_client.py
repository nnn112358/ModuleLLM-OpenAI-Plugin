import json
import socket
import time
import uuid
from contextlib import contextmanager
from typing import Generator
import logging
import threading

logger = logging.getLogger("llm_client")
logger.setLevel(logging.DEBUG)

class TTSClient:
    def __init__(self, host: str = "localhost", port: int = 10001):
        self._lock = threading.Lock()
        self.host = host
        self.port = port
        self.sock = None
        self.work_id = None
        self._initialized = False
        self._connect()
        
    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, _exc_type, _exc_val, _exc_tb):
        self.close()

    def _connect(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            self.sock.connect((self.host, self.port))
        except ConnectionRefusedError as e:
            raise RuntimeError(f"Failed to connect to {self.host}:{self.port}") from e

    def close(self):
        if self.sock:
            self.sock.close()
            self.sock = None

    def _send_request(self, action: str, object: str, data: dict) -> str:
        request_id = str(uuid.uuid4())
        object_type = object.split('.')[0] if '.' in object else "llm"
        payload = {
            "request_id": request_id,
            "work_id": self.work_id or object_type,
            "action": action,
            "object": object,
            "data": data
        }
        
        logger.debug(
            f"Sending request: [ID:{request_id}] "
            f"Action:{action} WorkID:{payload['work_id']}\n"
            f"Data: {str(data)[:100]}..."
        )
        self.sock.sendall(json.dumps(payload, ensure_ascii=False).encode('utf-8'))
        return request_id

    def setup(self, object: str, model_config: dict) -> dict:
        if not self.sock:
            self._connect()
        request_id = self._send_request("setup", object, model_config)
        return self._wait_response(request_id)

    def inference_stream(self, query: str, object_type: str = "llm.utf-8") -> Generator[str, None, None]:
        request_id = self._send_request("inference", object_type, query)
        buffer = b''
        
        while True:
            start_time = time.time()
            while time.time() - start_time < 10:
                chunk = self.sock.recv(4096)
                if not chunk:
                    break
                buffer += chunk
                
                while b'\n' in buffer:
                    line, buffer = buffer.split(b'\n', 1)
                    try:
                        response = json.loads(line.decode('utf-8'))
                        if response["request_id"] == request_id:
                            yield response["data"]["delta"]
                            if response["data"].get("finish", False):
                                self.work_id = response["work_id"]
                                return
                    except (json.JSONDecodeError, UnicodeDecodeError) as e:
                        logger.error(f"Failed to decode response: {e}")
                        continue

    def stop_inference(self) -> dict:
        request_id = self._send_request("pause", "llm.utf-8", {})
        return request_id

    def send_jpeg(self, query: str, object_type: str = "vlm.jpeg.base64") -> str:
        request_id = self._send_request("inference", object_type, query)
        return request_id
    
    def exit(self) -> dict:
        request_id = self._send_request("exit", "None", "None")
        result = self._wait_response(request_id)
        self._initialized = False
        return result

    def _wait_response(self, request_id: str) -> dict:
        start_time = time.time()
        while time.time() - start_time < 10:
            response = json.loads(self.sock.recv(4096).decode())
            if response["request_id"] == request_id:
                if response["error"]["code"] != 0:
                    raise RuntimeError(f"Server error: {response['error']['message']}")
                self.work_id = response["work_id"]
                return response
        raise TimeoutError("No response from server")

    def connect(self):
        with self._lock:
            if not self.sock:
                self._connect()

if __name__ == "__main__":
    with TTSClient(host='192.168.20.183') as client:
        setup_response = client.setup("melotts.setup", {
            "model": "melotts_zh-cn",
            "response_format": "pcm.stream.base64",
            "input": "tts.utf-8",
            "enoutput": True,
        })
        print("Setup response:", setup_response)
        time.sleep(1)
        for chunk in client.inference_stream("你叫什么名字？", object_type="tts.utf-8"):
            print("Received data chunk:", chunk)
            pass
        exit_response = client.exit()
        print("Exit response:", exit_response)