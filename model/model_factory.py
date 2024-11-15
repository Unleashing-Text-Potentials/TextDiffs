from config.base_config import Config
from model.clip_model import CLIPTextDiffs

class ModelFactory:
    @staticmethod
    def get_model(config: Config):
         return CLIPTextDiffs(config)
