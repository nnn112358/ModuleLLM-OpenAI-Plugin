import json
import socket
import time
import uuid
from typing import Generator
import logging
import threading

logger = logging.getLogger("asr_client")
logger.setLevel(logging.DEBUG)

class ASRClient:
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
            f"Data: {str(data)[:20]}..."
        )
        
        self.sock.sendall(json.dumps(payload, ensure_ascii=False).encode('utf-8'))
        return request_id
    
    def _send_request_stream(self, action: str, object: str, delta: str, index: int, finish: bool) -> str:
        request_id = str(uuid.uuid4())
        object_type = object.split('.')[0] if '.' in object else "llm"
        payload = {
            "request_id": request_id,
            "work_id": self.work_id or object_type,
            "action": action,
            "object": object,
            "data": {
                "delta": delta,
                "index": index,
                "finish": finish
            }
        }

        logger.debug(
            f"Sending streaming request: [ID:{request_id}] "
            f"Action:{action} WorkID:{payload['work_id']}\n"
            f"Delta: {delta[:20]}... Index: {index} Finish: {finish}"
        )
        
        self.sock.sendall(json.dumps(payload, ensure_ascii=False).encode('utf-8'))
        return request_id

    def setup(self, object: str, model_config: dict) -> dict:
        if not self.sock:
            self._connect()
        request_id = self._send_request("setup", object, model_config)
        return self._wait_response(request_id)

    def inference(self, query: str, object_type: str = "asr.base64") -> Generator[str, None, None]:
        request_id = self._send_request("inference", object_type, query)
        
        while True:
            response = json.loads(self.sock.recv(4096).decode())
            if response["request_id"] != request_id:
                continue
                
            yield response["data"]
            break

    def inference_stream(self, delta: str, index: int, finish: bool, object_type: str = "asr.base64") -> Generator[str, None, None]:
        request_id = self._send_request_stream("inference", object_type, delta, index, finish)

        while finish:
            response = json.loads(self.sock.recv(4096).decode())
            if response["request_id"] != request_id:
                continue
                
            yield response["data"]
            break

    def stop_inference(self) -> dict:
        request_id = self._send_request("pause", "llm.utf-8", {})
        return request_id

    def exit(self) -> dict:
        request_id = self._send_request("exit", "llm.utf-8", {})
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