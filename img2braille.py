import argparse
import sys
import cv2
import numpy as np


class CannotReadImage(Exception):
    pass


def grayscale_avg(img):
    b, g, r = cv2.split(img)
    return ((b + g + r) / 3).astype('uint8')


def grayscale_luma(img):
    # Otherwise take max components
    return img.max(axis=2)


def open_image_in_grayscale(filename, method=grayscale_luma):
    img = cv2.imread(filename, cv2.IMREAD_COLOR)
    if img is None:
        raise CannotReadImage(filename)
    return img, method(img)


def size_max(w, h):
    return 1


def size_from_width_ratio(new_width):
    def get_size(w, h):
        return (new_width * 2) / w
    return get_size


def size_from_height_ratio(new_height):
    def get_size(w, h):
        return (new_height * 3) / h
    return get_size


def resize(img, ratio):
    h, w = img.shape[0], img.shape[1]
    new_size = (
        int(w * ratio),
        int(h * ratio),
    )
    return cv2.resize(img, new_size, interpolation=cv2.INTER_LINEAR)


def smooth(img):
    return cv2.bilateralFilter(img, 9, sigmaSpace=10, sigmaColor=10)


def threshold(img):
    return cv2.adaptiveThreshold(img, 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, 11, 2)


def convert_to_braille(img, colour_img=None, invert=False):
    # Image already resized appropriately
    height, width = img.shape
    for y in range((height // 3) - 1):
        for x in range((width // 2) - 1):
            box = img[y*3:(y+1)*3, x*2:(x+1)*2]
            seq = [box[0, 0], box[1, 0], box[2, 0],
                   box[0, 1], box[1, 1], box[2, 1]]
            num = sum((2**i) * x for i, x in enumerate(seq))
            if invert:
                num = 63 - num

            # If we're in invert mode then it makes no sense
            # to compute the colour
            if not invert and colour_img is not None:
                yield seq_dominant_colour(colour_img[y*3:(y+1)*3, x*2:(x+1)*2])

            yield int_to_braille(num)

            if not invert and colour_img is not None:
                yield "\x1b[0m"
        yield "\n"


def seq_dominant_colour(seq):
    B = seq[:, :, 0]
    G = seq[:, :, 1]
    R = seq[:, :, 2]
    b, g, r = int(np.mean(B)), int(np.mean(G)), int(np.mean(R))
    return f"\x1b[38;2;{r};{g};{b}m"


def int_to_braille(i: int) -> str:
    # https://en.wikipedia.org/wiki/Braille_Patterns
    return chr(0x2800 + i)


def img2braille(
    filename,
    resize_size=size_max,
    grayscale_method=grayscale_luma,
    smoothing=True,
    invert=False,
    colour=False,
):
    colour_img, img = open_image_in_grayscale(
        filename,
        method=grayscale_method,
    )

    # Resizing
    h, w = img.shape
    img = resize(img, resize_size(w, h))

    if colour:
        colour_img = resize(colour_img, resize_size(w, h))

    # Perform smoothing
    if smoothing:
        img = smooth(img)

    img = threshold(img)
    colour_img = None if not colour else colour_img
    return convert_to_braille(img, colour_img=colour_img, invert=invert)


def main():
    parser = argparse.ArgumentParser(description='Converts image to braille.')
    parser.add_argument('file', metavar='filename', type=str)
    parser.add_argument('--disable-smoothing', dest='disable_smoothing',
                        action='store_true',
                        default=False,
                        help='Use bilateral filter for image smoothing')
    parser.add_argument('--grayscale-method', dest='grayscale_method',
                        type=str, default='luma', choices={'luma', 'avg'},
                        help='Grayscale method (luma/avg)')
    parser.add_argument('--no-resize', dest='no_resize', action='store_true',
                        default=False,
                        help='Prevent resizing')
    parser.add_argument('--width', dest='width', type=int, default=None,
                        help='Width of result')
    parser.add_argument('--height', dest='height', type=int, default=None,
                        help='Height of result')
    parser.add_argument('--invert', dest='invert', action='store_true',
                        help='Invert image')
    parser.add_argument('--colour', dest='colour', action='store_true',
                        help='Emit true colour')
    args = parser.parse_args()

    if args.no_resize:
        resize_size = size_max
    elif args.width and not args.height:
        resize_size = size_from_width_ratio(args.width)
    elif args.height and not args.width:
        resize_size = size_from_height_ratio(args.height)
    else:
        resize_size = size_from_width_ratio(30)

    result = img2braille(
        filename=args.file,
        resize_size=resize_size,
        grayscale_method={
            "luma": grayscale_luma,
            "avg": grayscale_avg,
        }[args.grayscale_method],
        smoothing=not args.disable_smoothing,
        invert=args.invert,
        colour=args.colour,
    )
    try:
        for c in result:
            sys.stdout.write(c)
    except BrokenPipeError:
        pass


if __name__ == '__main__':
    main()
