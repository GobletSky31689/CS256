from PIL import Image, ImageDraw
import os
import sys
import shutil
import math
import random
import numpy

NUM_EXAMPLES = int(sys.argv[2])
DEST_FOLDER = sys.argv[1]
FINAL_DIMENSION = 25

# specify maximum rotation in degrees, Value between [0,180)
rotation_range = 180
# specify maximum stroke, Value between [1,5)
STROKE_WIDTH = 3
# specify maximum scale range value between [1,1.5)
SCALE_RANGE = 1.5
# specify whether varying position, Value either True or False,
# Note that scale range should be greater than 1, if this True
VARY_POS = True
# specify MAX num of splotches, value between 1, 10
NUM_SPLOTCHES = 9
# specify MAX size of splotch, value between 1, 5
SIZE_SPLOTCH = 3


def get_square():
    return Image.open('zener_src/Q.png').convert('1'), 'Q.png'


def get_circle():
    return Image.open('zener_src/C.png').convert('1'), 'C.png'


def get_plus():
    return Image.open('zener_src/P.png').convert('1'), 'P.png'


def get_waves():
    return Image.open('zener_src/W.png').convert('1'), 'W.png'


def get_star():
    return Image.open('zener_src/S.png').convert('1'), 'S.png'


def initialize():
    if os.path.exists(DEST_FOLDER):
        for root, dirs, files in os.walk(DEST_FOLDER):
            for f in files:
                os.unlink(os.path.join(root, f))
            for d in dirs:
                shutil.rmtree(os.path.join(root, d))
    else:
        os.makedirs(DEST_FOLDER)


def rotate_symbol(img):
    rotation_factor = math.pow(random.uniform(0.0, 1.0), 4)
    rotation_direction = (1, -1)[random.random() > 0.5]
    rotation_angle = int(math.floor(rotation_range * rotation_factor * rotation_direction))
    img = img.convert('RGBA').rotate(rotation_angle, expand=1)
    return Image.alpha_composite(Image.new('RGBA', img.size, (255, 255, 255)), img)


def resize_move_symbol(img):
    # scaled size will be between 12 to 25
    scaled_size = int(FINAL_DIMENSION/random.uniform(1, SCALE_RANGE))
    img.thumbnail((scaled_size, scaled_size))
    white_canvas = Image.new('1', (FINAL_DIMENSION, FINAL_DIMENSION), 1)
    # Center the image
    if not VARY_POS:
        x1 = int(math.floor((FINAL_DIMENSION - scaled_size) / 2))
        y1 = int(math.floor((FINAL_DIMENSION - scaled_size) / 2))
    else:
        x1 = int(random.uniform(0, FINAL_DIMENSION - scaled_size))
        y1 = int(random.uniform(0, FINAL_DIMENSION - scaled_size))
    white_canvas.paste(img, (x1, y1))
    return white_canvas


def change_symbol_stroke(img):
    variation = int(random.uniform(0, STROKE_WIDTH-1))
    for _ in xrange(variation-1):
        img = add_stroke(img)
    return img


def add_splotches(img):
    num_splotches = int(random.uniform(1, NUM_SPLOTCHES))
    size_splotch = int(random.uniform(1, SIZE_SPLOTCH))
    draw = ImageDraw.Draw(img)
    for _ in xrange(num_splotches):
        x1 = int(random.uniform(0, FINAL_DIMENSION - size_splotch))
        y1 = int(random.uniform(0, FINAL_DIMENSION - size_splotch))
        x2 = int(random.uniform(x1, x1 + size_splotch))
        y2 = int(random.uniform(y1, y1 + size_splotch))
        draw.ellipse((x1, y1, x2, y2), fill='black')
    return img


def add_stroke(img):
    numpy_array = numpy.array(img.convert('L'))
    new_arr = numpy.array(numpy_array, copy=True)
    for i in range(len(numpy_array)):
        for j in range(len(numpy_array[0])):
            if numpy_array[i][j] >= 50:
                if i-1 >= 0 and numpy_array[i-1][j] < numpy_array[i][j]:
                    new_arr[i][j] = numpy_array[i-1][j]
                elif i+1 < len(numpy_array) and numpy_array[i+1][j] < numpy_array[i][j]:
                    new_arr[i][j] = numpy_array[i+1][j]
                elif j-1 >= 0 and numpy_array[i][j-1] < numpy_array[i][j]:
                    new_arr[i][j] = numpy_array[i][j-1]
                elif j+1 < len(numpy_array[0]) and numpy_array[i][j+1] < numpy_array[i][j]:
                    new_arr[i][j] = numpy_array[i][j+1]
    new_image = Image.fromarray(new_arr)
    return new_image


generators = [get_circle, get_square, get_plus, get_star, get_waves]


def get_gen(main_gen):
    one = generators[int(random.uniform(0, 5))]
    toss = int(random.uniform(0, 2))
    if toss == 1:
        return one()
    else:
        return main_gen()


initialize()
for ex_num in xrange(1, NUM_EXAMPLES+1):
    symbol, file_name = get_gen(get_circle)
    symbol = rotate_symbol(symbol)
    for _ in xrange(STROKE_WIDTH-1):
        symbol = change_symbol_stroke(symbol)
    symbol = resize_move_symbol(symbol)
    symbol = add_splotches(symbol)
    symbol.save(DEST_FOLDER+'/' + str(ex_num) + '_' + file_name)
