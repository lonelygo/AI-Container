# -*- coding:utf-8 -*-
import linecache

from captcha.image import ImageCaptcha  # pip install captcha
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageOps
import random
import cv2
import os
import uuid
import colorsys

list = []

BG_PATH = './background_img/id/'
SAVE_PATH = './output/'
FONTS_PATH = './ctn_ttf/'

number = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

ALPHABET = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P',
            'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
BAKCGROUND = []

FONTCOLOR = ['0', '255', '250', '254', '25', '253', '252']

ID1 = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
       'W', 'X', 'Y', 'Z']
ID2 = ['0', '2', '5', '6', '8', '9', '1', '3', '4', '7']
ID3 = ['0', '2', '5', '6', '8', '9', '1', '3', '4', '7']

FONTS = [
         '096-CAI978.ttf'
         ]


def rad(x):
    return x * np.pi / 180


def get_dominant_color(image):

    image = image.convert('RGBA')
    max_score = 0
    dominant_color = 0
    for count, (r, g, b, a) in image.getcolors(image.size[0] * image.size[1]):
        if a == 0:
            continue

        saturation = colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)[1]
        y = min(abs(r * 2104 + g * 4130 + b * 802 + 4096 + 131072) >> 13, 235)
        y = (y - 16.0) / (235 - 16)

        if y > 0.9:
            continue
        score = (saturation + 0.1) * count

        if score > max_score:
            max_score = score
            dominant_color = (r, g, b)

    return dominant_color


def perspective_transform_new(img):

    # 扩展图像，保证内容不超出可视范围
    PIL_img = img
    img = np.array(img)

    # img = cv2.copyMakeBorder(img, 65, 65, 65, 65, cv2.BORDER_CONSTANT, value=get_dominant_color(PIL_img))
    img = cv2.copyMakeBorder(img, 65, 65, 65, 65, cv2.BORDER_CONSTANT, value=0)
    w, h = img.shape[0:2]
    # print(w, h)

    anglex = random.randint(-10, 10)
    angley = random.randint(-10, 10)
    anglez = random.randint(-20, 20)
    fov = random.randint(10, 20)


    z = np.sqrt(w ** 2 + h ** 2) / 2 / np.tan(rad(fov / 2))

    rx = np.array([[1, 0, 0, 0],
                   [0, np.cos(rad(anglex)), -np.sin(rad(anglex)), 0],
                   [0, -np.sin(rad(anglex)), np.cos(rad(anglex)), 0, ],
                   [0, 0, 0, 1]], np.float32)

    ry = np.array([[np.cos(rad(angley)), 0, np.sin(rad(angley)), 0],
                   [0, 1, 0, 0],
                   [-np.sin(rad(angley)), 0, np.cos(rad(angley)), 0, ],
                   [0, 0, 0, 1]], np.float32)

    rz = np.array([[np.cos(rad(anglez)), np.sin(rad(anglez)), 0, 0],
                   [-np.sin(rad(anglez)), np.cos(rad(anglez)), 0, 0],
                   [0, 0, 1, 0],
                   [0, 0, 0, 1]], np.float32)

    r = rx.dot(ry).dot(rz)

    # 四对点的生成
    pcenter = np.array([h / 2, w / 2, 0, 0], np.float32)

    p1 = np.array([0, 0, 0, 0], np.float32) - pcenter
    p2 = np.array([w, 0, 0, 0], np.float32) - pcenter
    p3 = np.array([0, h, 0, 0], np.float32) - pcenter
    p4 = np.array([w, h, 0, 0], np.float32) - pcenter

    dst1 = r.dot(p1)
    dst2 = r.dot(p2)
    dst3 = r.dot(p3)
    dst4 = r.dot(p4)

    list_dst = [dst1, dst2, dst3, dst4]

    org = np.array([[0, 0],
                    [w, 0],
                    [0, h],
                    [w, h]], np.float32)

    dst = np.zeros((4, 2), np.float32)

    # 投影至成像平面
    for i in range(4):
        dst[i, 0] = list_dst[i][0] * z / (z - list_dst[i][2]) + pcenter[0]
        dst[i, 1] = list_dst[i][1] * z / (z - list_dst[i][2]) + pcenter[1]

    warpR = cv2.getPerspectiveTransform(org, dst)

    # result = cv2.warpPerspective(img, warpR, (h, w), borderValue=get_dominant_color(PIL_img))
    result = cv2.warpPerspective(img, warpR, (h, w), borderValue=(0, 0, 0))

    # cv2.namedWindow('result', cv2.WINDOW_NORMAL)
    # cv2.imshow("result", result)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return result


def perspective_transform(im):
    PIL_img = im
    im = np.array(im)
    w, h = im.shape[0:2]

    im = cv2.copyMakeBorder(im, 0, 0, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])

    anglex = random.randint(-6, 6)
    angley = random.randint(-6, 6)
    anglez = random.randint(-6, 6)
    fov = 42
    z = np.sqrt(w ** 2 + h ** 2) / 2 / np.tan(rad(fov / 2))
    # 齐次变换矩阵
    rx = np.array([[1, 0, 0, 0],
                   [0, np.cos(rad(anglex)), -np.sin(rad(anglex)), 0],
                   [0, -np.sin(rad(anglex)), np.cos(rad(anglex)), 0, ],
                   [0, 0, 0, 1]], np.float32)

    ry = np.array([[np.cos(rad(angley)), 0, np.sin(rad(angley)), 0],
                   [0, 1, 0, 0],
                   [-np.sin(rad(angley)), 0, np.cos(rad(angley)), 0, ],
                   [0, 0, 0, 1]], np.float32)

    rz = np.array([[np.cos(rad(anglez)), np.sin(rad(anglez)), 0, 0],
                   [-np.sin(rad(anglez)), np.cos(rad(anglez)), 0, 0],
                   [0, 0, 1, 0],
                   [0, 0, 0, 1]], np.float32)

    r = rx.dot(ry).dot(rz)

    # 四对点的生成
    pcenter = np.array([h / 1.5, w / 1.5, 0, 0], np.float32)

    p1 = np.array([0, 0, 0, 0], np.float32) - pcenter
    p2 = np.array([w, 0, 0, 0], np.float32) - pcenter
    p3 = np.array([0, h, 0, 0], np.float32) - pcenter
    p4 = np.array([w, h, 0, 0], np.float32) - pcenter

    dst1 = r.dot(p1)
    dst2 = r.dot(p2)
    dst3 = r.dot(p3)
    dst4 = r.dot(p4)

    list_dst = [dst1, dst2, dst3, dst4]

    org = np.array([[0, 0],
                    [w, 0],
                    [0, h],
                    [w, h]], np.float32)

    dst = np.zeros((4, 2), np.float32)

    # 投影至成像平面
    for _i in range(4):
        dst[_i, 0] = list_dst[_i][0] * z / (z - list_dst[_i][2]) + pcenter[0]
        dst[_i, 1] = list_dst[_i][1] * z / (z - list_dst[_i][2]) + pcenter[1]

    warpR = cv2.getPerspectiveTransform(org, dst)

    im = cv2.warpPerspective(im, warpR, (h, w), flags=cv2.INTER_LINEAR, borderValue=get_dominant_color(PIL_img))

    return im


def gen_captcha_text_and_image(index):

    tmp1 = random.sample(ID1, 1) + random.sample(ID1, 1) + random.sample(ID1, 1) \
          + random.sample(ID1, 1)

    tmp2 = random.sample(ID2, 1) + random.sample(ID2, 1) + random.sample(ID2, 1) \
          + random.sample(ID2, 1) + random.sample(ID2, 1) + random.sample(ID2, 1)

    tmp3 = random.sample(ID3, 1)

    IMAGENAME = random.sample(BAKCGROUND, 1)[0]
    text = ''.join(tmp1)
    # text = ''.join(tmp1) + '  ' + ''.join(tmp2) + ''.join(tmp3)

    filename = text

    if text in list:
        filename = text + '-' + str(index)

    list.append(text)

    # print("gen ", text, '......')
    im = gen_rgba_image(text)
    w, h = np.array(im).shape[0:2]

    im = perspective_transform(im)

    im2 = Image.open(BG_PATH + 'BG1.jpg')
    # im2 = Image.open(BG_PATH + IMAGENAME)

    target = Image.new('RGB', im2.size, (0, 0, 0, 0))

    target.paste(im2)

    w2, h2 = np.array(im2).shape[0:2]

    im = Image.fromarray(im).filter(ImageFilter.SMOOTH_MORE)

    im = im.resize((int(h / (h / h2)), int(w / (w / w2))))

    r, g, b, a = im.split()

    target.paste(im, mask=a)
    target = target.filter(ImageFilter.SMOOTH_MORE)

    # plt.imshow(target)
    # plt.show()

    str2 = filename + '-' + str(uuid.uuid1()) + '.jpg'
    target.save(SAVE_PATH + str2)  # 写到文件

    return text, im


def make_image(text1, text2, text3, background_img_name, check_digit_img, fontsize=35, fontname=''.join(random.sample(FONTS, 1))):
    """Make an image out of a poem"""
    text = text1 + "      " + text2
    #
    # pad = int(20)
    font = ImageFont.truetype(os.path.join(FONTS_PATH, fontname), fontsize)
    num_lines = (1 + text.strip().count('\n'))
    print("num_lines: " + str(num_lines))
    # height = num_lines * font.getsize(text[:10])[1] + 2 * pad
    # font_length = max(font.getsize(line)[0] for line in text.split('\n'))
    # width = font_length + 2 * pad
    image = Image.open(os.path.join(BG_PATH, background_img_name))

    draw = ImageDraw.Draw(image)
    draw.text((10, 0), text, (255, 255, 255), font=font)
    img_resized = image.resize(image.size, Image.ANTIALIAS)

    width1, height1 = img_resized.size
    print("PIL_resized_img: ")
    print("height: " + str(height1) + " width: " + str(width1))

    width2, height2 = check_digit_img.size

    check_digit_img = check_digit_img.resize((int(width2 / 1.4), int(height2 / 1.4)), Image.ANTIALIAS)

    # 第三个参数是mask，设置成透明图
    img_resized.paste(check_digit_img, (width1 - width2 + 10, 0), check_digit_img)
    # img_resized.show()

    # transformed_img = perspective_transform(img_resized)
    #
    # plt.imshow(transformed_img, cmap='gray', interpolation='nearest')
    # plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    # plt.show()

    # cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    # cv2.imshow("image", transformed_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    str_filename = text1 + text2 + text3 + '-' + str(uuid.uuid1()) + '.jpg'
    # img_resized.save(SAVE_PATH + str2)
    return img_resized, str_filename


def draw_rect_number(number, fontsize=53, fontname=''.join(random.sample(FONTS, 1))):

    # source_img = Image.open("D:\\ctn_ttf\\background_img\\size\\22G1-312.jpg")

    transparent_img = Image.new('RGBA', (60, 60), (0, 0, 0, 0))

    rect_text_draw = ImageDraw.Draw(transparent_img)
    # put text on image
    text_font = ImageFont.truetype(os.path.join(FONTS_PATH, fontname), fontsize)
    text_size = text_font.getsize(str(number))
    print(text_size[0], text_size[1])

    text_x = 10
    text_y = -3

    rect_text_draw.text((text_x, text_y), str(number), font=text_font)

    cor = (text_x - 5, text_y + 8, text_size[0] + 14, text_size[1] + 3)  # (x1,y1, x2,y2)

    line = (cor[0], cor[1], cor[0], cor[3])
    rect_text_draw.line(line, fill="white", width=int(fontsize / 15))

    line = (cor[0], cor[1], cor[2], cor[1])
    rect_text_draw.line(line, fill="white", width=int(fontsize / 15))

    line = (cor[0], cor[3], cor[2], cor[3])
    rect_text_draw.line(line, fill="white", width=int(fontsize / 15))

    line = (cor[2], cor[1], cor[2], cor[3])
    rect_text_draw.line(line, fill="white", width=int(fontsize / 15))

    # source_img.show()

    return transparent_img.crop((0, 0, 40, 60))
    # rect_text_draw.rectangle((text_x, text_y, text_size[0] + 20, text_size[1] + 10))

    # put button on source image in position (0, 0)
    # source_img.paste(button_img, (0, 0))


if __name__ == '__main__':

    for i in range(1):
        tmp1 = ''.join(random.sample(ID1, 1) + random.sample(ID1, 1) + random.sample(ID1, 1) + random.sample(ID1, 1))

        tmp2 = ''.join(random.sample(ID2, 1) + random.sample(ID2, 1) + random.sample(ID2, 1) \
            + random.sample(ID2, 1) + random.sample(ID2, 1) + random.sample(ID2, 1))

        tmp3 = ''.join(random.sample(ID3, 1))

        img_resized, str_filename = make_image(tmp1, tmp2, tmp3, 'BG1.jpg', draw_rect_number(tmp3))

        img_resized.save(os.path.join(SAVE_PATH, str_filename))

        # transformed_img = perspective_transform(img_resized)
        # transformed_img = perspective_transform_new(img_resized)

        # cv2.imwrite(os.path.join(SAVE_PATH, str_filename), cv2.cvtColor(transformed_img, cv2.COLOR_RGB2BGR))

        # plt.imshow(transformed_img, cmap='gray', interpolation='nearest')
        # plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
        # plt.show()


    # draw_rect_number(9).show()

    # cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    # cv2.imshow("image", draw_rect_number(9))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # list_image_file()
    # for i in range(1):
    #     print(i)
    #     text, image = gen_captcha_text_and_image(i)
