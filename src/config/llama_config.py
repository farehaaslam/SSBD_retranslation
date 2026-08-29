import sys
from pathlib import Path    
from platformdirs import user_config_dir

class LlamaServerManager:
    def __init__(self):
        self.config_dir = Path(user_config_dir("ssbd"))
        
        self.is_windows = sys.platform == "win32"
        
        self.bin_dir = self.config_dir / "bin"
        self.models_dir = self.config_dir / "models"
        
        self.server_bin = self.bin_dir / ("llama-server.exe" if self.is_windows else "llama-server")
        self.hf_repo = "unsloth/gemma-3-4b-it-GGUF"
        self.model_filename = "gemma-3-4b-it-Q4_K_M.gguf"
        
        self.llm_path = self.models_dir / self.model_filename
        
        self.llama_release_url = "https://api.github.com/repos/ggml-org/llama.cpp/releases/latest"