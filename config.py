import os
import yaml


class ConfigObj:
    def __init__(self, **entries):
        for key in entries:
            if type(entries[key]) == dict:
                self.__dict__[key] = ConfigObj(**entries[key])
            else:
                self.__dict__[key] = entries[key]


def get_config(ymlfn, tag="default"):
    with open(os.path.join(ymlfn)) as f:
        config = yaml.load(f, Loader=yaml.CLoader)
    return ConfigObj(**config)
