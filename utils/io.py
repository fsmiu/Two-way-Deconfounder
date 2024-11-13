import yaml
import random
import numpy as np
import pandas as pd

def load_yaml(path, key='parameters'):
    with open(path, 'r') as stream:
        try:
            return yaml.safe_load(stream)[key]
        except yaml.YAMLError as exc:
            print(exc)
