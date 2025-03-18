import logging
import asyncio
import yaml
from typing import Optional
from client.sys_client import SYSClient

class GetModelList:
    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self.logger = logging.getLogger("get_model_list")
        self._sys_client: Optional[SYSClient] = None
        
    async def get_model_list(self, required_mem: int) -> None:
        try:
            if not self._sys_client:
                self._sys_client = SYSClient(host=self.host, port=self.port)

            with open('config/config.yaml', 'r') as f:
                config = yaml.safe_load(f)
            models_config = config.get('models', {})
            model_list = await self._get_model_list()

            for model_data in model_list["data"]:
                mode = model_data.get("mode")
                model_type = model_data.get("type")

                if not mode or not model_type:
                    continue 

                if model_type not in ['llm', 'vlm', 'tts', 'asr']:
                    continue
                
                if mode not in models_config:
                    new_entry = {
                        "host": self.host,
                        "port": self.port,
                        "type": model_type,
                        "input": f"{model_type}.utf-8",
                        "model_name": mode,
                    }

                    if model_type in ['llm', 'vlm']:
                        new_entry.update({
                            "response_format": f"{model_type}.utf-8.stream",
                            "object": f"{model_type}.setup",
                            "system_prompt": "You are a helpful assistant."
                        })
                        if '-1.5B-' in mode:
                            new_entry['memory_required'] = 1782579
                            new_entry['pool_size'] = 1
                        elif '-1B-' in mode:
                            new_entry['memory_required'] = 1363148
                            new_entry['pool_size'] = 2
                        elif '-0.5B-' in mode:
                            new_entry['memory_required'] = 560460
                            new_entry['pool_size'] = 2

                    elif model_type == 'tts':
                        if 'melotts' in mode.lower():
                            obj = 'melotts.setup'
                            new_entry['memory_required'] = 59764
                        else:
                            obj = 'tts.setup'
        
                        new_entry.update({
                            "response_format": "wav.base64",
                            "object": obj
                        })
                    elif model_type == 'asr':
                        if 'whisper' in mode.lower():
                            obj = 'whisper.setup'
                            if 'tiny' in mode:
                                new_entry['memory_required'] = 289132
                        else:
                            obj = 'asr.setup'
                        new_entry.update({
                            "input": "pcm.base64",
                            "response_format": "asr.utf-8",
                            "object": obj
                        })
                    else:
                        continue

                    models_config[mode] = new_entry
                    config['models'] = models_config
                    with open('config/config.yaml', 'w') as f:
                        yaml.safe_dump(config, f, default_flow_style=False, sort_keys=False)

        except Exception as e:
            self.logger.error(f"Get model failed: {str(e)}")
            raise

    async def _get_model_list(self):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, 
            self._sys_client.model_list
        ) 
