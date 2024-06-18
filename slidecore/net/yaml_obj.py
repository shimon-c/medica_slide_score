import yaml

class YamlObj:
    def __init__(self, yaml_path:str=None):
        with open(yaml_path, "r") as f:
            self.dct = yaml.safe_load(f)

    def get_params(self):
        return self.dct