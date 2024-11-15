from config.base_config import Config
from model.clip_model import CLIPTextDiffs

class ModelFactory:
    @staticmethod
    def get_model(config: Config):
        if config.arch == 'clip_stochastic':
            return CLIPTextDiffs(config)
        else:
            raise NotImplementedError
