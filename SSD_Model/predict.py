'''
There are several points to note about predict.py
1. This code cannot directly perform batch prediction. If you want to perform batch prediction, you can use os.listdir() to traverse the folder and use Image.open to open the image file for prediction.
For the specific process, please refer to get_dr_txt.py. In get_dr_txt.py, traversal is realized and target information is saved.
2. If you want to save the detected image, use r_image.save("img.jpg") to save it, and modify it directly in predict.py.
3. If you want to get the coordinates of the prediction box, you can enter the detect_image function and read the four values ​​​​of top, left, bottom, and right in the drawing part.
4. If you want to use the prediction frame to intercept the target, you can enter the detect_image function and use the obtained four values ​​of top, left, bottom, and right in the drawing part.
Use the matrix method to intercept the original image.
5. If you want to write additional words on the prediction map, such as the number of specific targets detected, you can enter the detect_image function and judge predicted_class in the drawing part.
For example, if predicted_class == 'car': can determine whether the current target is a car, and then record the number. Use draw.text to write.
'''
from PIL import Image

from ssd import SSD

ssd = SSD()

while True:
    img = input('Input image filename:')
    try:
        image = Image.open(img)
    except:
        print('Open Error! Try again!')
        continue
    else:
        r_image = ssd.detect_image(image)
        # r_image.save("img.jpg")
        r_image.show()
