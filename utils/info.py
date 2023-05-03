import os
from pathlib import Path
import sys
# insert root path
sys.path.insert(1, os.path.join(Path(__file__).resolve().parent.parent))

import json


with open(os.path.join(Path(__file__).resolve().parent.parent, 'res', 'settings.json'), 'r') as f:
    SETTINGS = json.load(f)

vs = "1.0.0"

class PATH:
    ROOT = Path(__file__).resolve().parent.parent
    RES = os.path.join(ROOT, 'res')
    APP = os.path.join(ROOT, 'app')
    SRC = os.path.join(ROOT, 'src')
    MODELS = os.path.join(SRC, 'models')
    CACHE = os.path.join(SRC, 'cache')

class INSTRUCTIONS:
    rp = """
    Below is an 
    """