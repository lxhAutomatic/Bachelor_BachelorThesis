#-------------------------------------#
#       Make predictions on a single image
#-------------------------------------#
from yolo import YOLO
from PIL import Image
import time

yolo = YOLO()

while True:
    img = input('Input image filename:')
    try:
        image = Image.open(img)
        image = image.convert('RGB')
    except:
        print('Open Error! Try again!')
        continue
    else:
        since = time.time()
        r_image = yolo.detect_image(image)
        print(time.time() - since)
        r_image.show()
