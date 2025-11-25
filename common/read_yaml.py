import yaml
# 读取配置文件

class ReadYaml:
    def __init__(self):
        self.yaml_root = '../yaml_file/param.yaml'

    def read(self):
        try:
            # 打开文件
            with open(self.yaml_root, "r", encoding="utf-8") as f:
                data = yaml.load(f, Loader=yaml.FullLoader)
                return data
        except:
            print(f"Error reading YAML file: {self.yaml_root}")
            return None