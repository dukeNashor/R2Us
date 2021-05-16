import os

import quickdraw as qd
import requests

from PIL import Image

print(requests.__version__)

# set proxy settings
os.environ["HTTP_PROXY"] = "http://127.0.0.1:7890"
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7890"

print(os.environ["HTTP_PROXY"]) 
print(os.environ["HTTPS_PROXY"])

anvil = qd.QuickDrawData().get_drawing("anvil")

print(anvil)

anvils = qd.QuickDrawDataGroup("anvil")
print(anvils.drawing_count)

for i in range(anvils.drawing_count):
    drawing = anvils.get_drawing(index = i)
    drawing.image.show()
    
    




pass