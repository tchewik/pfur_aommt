import numpy as np

from .compression import RLECompressor, LZ77Compressor, HuffmanCompressor
from .convolution import Convolution
from .wavelet import HaarWavelet


class RAWImage:
    def __init__(self, color_mode, height, width, pixels_raw):
        self.color_mode = color_mode
        self.height = height
        self.width = width
        self.pixels_raw = pixels_raw

    def to_matrix(self):
        if self.color_mode == 3:
            return self.pixels_raw.reshape(self.height, self.width, 3)
        elif self.color_mode == 2:
            return self.pixels_raw.reshape(self.height, self.width)


class ImageProcessor:
    def read(self, filename, compressed=False):
        '''
        Args: image name
        Return: 2D or 3D list of pixels in [y][x][c] format (c is optional)

        Read PPM or PGM image.
        FOR PGM:
        Returns 2D matrix, where each value is a greyscale value in 0..255

        For PPM:
        Returns 3D matrix, where each value is a color channel value in 0.255
        0 for red
        1 for green
        2 for blue
        '''
        with open(filename, 'r') as ppm_file:
            lines = ppm_file.readlines()

        color_mode = int(lines[0][1])  # which matrix to read, 2d or 3d

        if lines[1][0] == "#":
            offset = 1
        else:
            offset = 0

        size = lines[1 + offset].split()
        width, height = int(size[0]), int(size[1])
        print("width={}, height={}, colors={}".format(width, height, color_mode))
        content = lines[3 + offset:]

        if not compressed:
            allValues = np.array([[int(v) for v in value.split()] for value in content]).flatten()
            image = RAWImage(color_mode, height, width, allValues)
            return image

        else:
            allValues = np.array([[v for v in value.split()] for value in content]).flatten()
            image = RAWImage(color_mode, height, width, allValues)
            return image

    def write(self, filename, image):
        '''
        Args: image name, image data, where image in PPM or PGM formats
        Return: none
        '''

        mtrx = image.to_matrix()

        if image.color_mode == 3:
            plain_text = '\n'.join(['\n'.join(
                [' '.join(str(mtrx[y][x][c]) for c in range(mtrx.shape[2]))
                 for x in range(mtrx.shape[1])])
                for y in range(mtrx.shape[0])])
        else:
            plain_text = '\n'.join(['\n'.join(
                [str(mtrx[y][x]) for x in range(mtrx.shape[1])])
                for y in range(mtrx.shape[0])])

        with open(filename, 'w') as ppm_file:
            ppm_file.write("P%d\n" % image.color_mode)
            ppm_file.write(str(image.width) + " " + str(image.height) + "\n")
            ppm_file.write("255\n")
            ppm_file.write(plain_text)
            ppm_file.write('\n')

    def write_compressed(self, filename, image):
        plain_text = ' '.join([str(pixel) for pixel in image.pixels_raw])
        plain_text = plain_text.replace(' \n ', '\n')

        with open(filename, 'w') as ppm_file:
            ppm_file.write("P%d\n" % image.color_mode)
            ppm_file.write('%d %d\n' % (image.width, image.height))
            ppm_file.write("255\n")
            ppm_file.write(plain_text)
            ppm_file.write('\n')

    def compress_image_(self, image, compressor, to_bytes=False, with_dict=False):
        def compress_channel(channel):
            channel = list(channel)

            if with_dict:
                # Huffman case
                channel = ''.join(list(map(chr, channel)))
                compressed, dict = compressor.compress(channel)
                dict = ['{}:{}'.format(key, ord(dict[key])) for key in dict.keys()]
                return compressed + ['\n'] + dict

            if to_bytes:
                channel = ''.join(list(map(chr, channel)))
                return compressor.compress(channel)

            return compressor.compress(channel)

        if image.color_mode == 2:
            new_image = RAWImage(
                color_mode=image.color_mode,
                height=image.height,
                width=image.width,
                pixels_raw=compress_channel(image.pixels_raw)
            )
            return new_image

        if image.color_mode == 3:
            mtrx = image.to_matrix()
            red_channel = mtrx[:, :, 0].flatten()
            green_channel = mtrx[:, :, 1].flatten()
            blue_channel = mtrx[:, :, 2].flatten()

            red_compressed = compress_channel(red_channel)
            green_compressed = compress_channel(green_channel)
            blue_compressed = compress_channel(blue_channel)

            encoded = red_compressed + ['\n'] + green_compressed + ['\n'] + blue_compressed
            new_image = RAWImage(
                color_mode=image.color_mode,
                height=image.height,
                width=image.width,
                pixels_raw=encoded
            )
            return new_image

    def decompress_image_(self, image, compressor, with_dict=False):
        def decompress_channel(channel, dictionary=None):
            if dictionary:
                # Huffman case
                compr = HuffmanCompressor()
                compr.reverse_mapping = {item.split(':')[0]: item.split(':')[1] for item in dictionary}
                actual_pixels = compr.decompress(channel)
                return list(map(int, actual_pixels))

            channel = list(channel)
            if type(channel[0]) in (str, np.str_, np.int64):
                return list(map(int, compressor.decompress(channel)))
            else:
                return list(map(ord, compressor.decompress(channel)))

        if image.color_mode == 2:
            if with_dict:
                pixels_raw = np.array(decompress_channel(image.pixels_raw[0], image.pixels_raw[1]))
            else:
                pixels_raw = np.array(decompress_channel(image.pixels_raw))

            new_image = RAWImage(
                color_mode=image.color_mode,
                height=image.height,
                width=image.width,
                pixels_raw=pixels_raw
            )
            return new_image

        if image.color_mode == 3:
            if with_dict:
                red_channel, red_dict, green_channel, green_dict, blue_channel, blue_dict = image.pixels_raw
                red_decoded = decompress_channel(red_channel, red_dict)
                green_decoded = decompress_channel(green_channel, green_dict)
                blue_decoded = decompress_channel(blue_channel, blue_dict)
            else:
                red_channel, green_channel, blue_channel = image.pixels_raw
                red_decoded = decompress_channel(red_channel)
                green_decoded = decompress_channel(green_channel)
                blue_decoded = decompress_channel(blue_channel)

            pixels_raw = np.array(
                [[red_decoded[i], green_decoded[i], blue_decoded[i]] for i in range(len(red_decoded))]).flatten()
            new_image = RAWImage(
                color_mode=image.color_mode,
                height=image.height,
                width=image.width,
                pixels_raw=pixels_raw,
            )
            return new_image

    def compress_rle(self, image):
        compressor = RLECompressor()
        return self.compress_image_(image, compressor)

    def decompress_rle(self, image):
        compressor = RLECompressor()
        return self.decompress_image_(image, compressor)

    def compress_lz77(self, image, window_size):
        compressor = LZ77Compressor(window_size)
        return self.compress_image_(image, compressor, to_bytes=True)

    def decompress_lz77(self, image, window_size):
        compressor = LZ77Compressor(window_size)
        return self.decompress_image_(image, compressor)

    def compress_huffman(self, image):
        compressor = HuffmanCompressor()
        return self.compress_image_(image, compressor, with_dict=True)

    def decompress_huffman(self, image):
        compressor = HuffmanCompressor()
        return self.decompress_image_(image, compressor, with_dict=True)

    def convolve(self, image, kernel):
        processor = Convolution(kernel)

        def process(channel):
            mtrx = np.array(channel).reshape(image.height, image.width)
            new_mtrx = processor(mtrx)
            return np.array(new_mtrx).flatten()

        if image.color_mode == 2:
            pixels_raw = process(image.pixels_raw)

        if image.color_mode == 3:
            green_channel = [image.pixels_raw[i] for i in range(1, image.height * image.width * 3, 3)]
            blue_channel = [image.pixels_raw[i] for i in range(2, image.height * image.width * 3, 3)]
            red_convolved = process(red_channel)
            green_convolved = process(green_channel)
            blue_convolved = process(blue_channel)
            pixels_raw = np.array(
                [[red_convolved[i], green_convolved[i], blue_convolved[i]] for i in
                 range(len(red_convolved))]).flatten()

        new_image = RAWImage(
            color_mode=image.color_mode,
            height=image.height,
            width=image.width,
            pixels_raw=pixels_raw
        )

        return new_image

    def crop_image(self, image, top=0, right=0, bottom=0, left=0):
        mtrx = image.to_matrix()
        if image.color_mode == 2:
            pixels_raw = mtrx[left:mtrx.shape[0] - right, top:mtrx.shape[1] - bottom]

        if image.color_mode == 3:
            red = mtrx[left:mtrx.shape[0] - right, top:mtrx.shape[1] - bottom, 0].flatten()
            green = mtrx[left:mtrx.shape[0] - right, top:mtrx.shape[1] - bottom, 1].flatten()
            blue = mtrx[left:mtrx.shape[0] - right, top:mtrx.shape[1] - bottom, 2].flatten()
            pixels_raw = np.array(
                [[red[i], green[i], blue[i]] for i in range(len(red))]).flatten()

        new_image = RAWImage(
            color_mode=image.color_mode,
            height=image.height - top - bottom,
            width=image.width - left - right,
            pixels_raw=pixels_raw
        )

        return new_image

    def pad_image(self, image, top=0, right=0, bottom=0, left=0):
        mtrx = image.to_matrix()
        image.width += left + right
        image.height += top + bottom
        if image.color_mode == 2:
            image.pixels_raw = np.pad(mtrx, ((top, bottom), (left, right)), 'constant')

        if image.color_mode == 3:
            new_mtrx = np.pad(mtrx, ((top, bottom), (left, right), (0, 0)), 'constant')
            image.pixels_raw = new_mtrx.flatten()

        return image

    def apply_haar(self, function, image, axis):
        right, bottom = 0, 0
        if image.width % 2:
            right = 1
        if image.height % 2:
            bottom = 1

        if right or bottom:
            image = self.pad_image(image, right=right, bottom=bottom)

        if image.color_mode == 2:
            mtrx = image.to_matrix()
            pixels_raw = np.apply_along_axis(function, axis, mtrx)

        if image.color_mode == 3:
            mtrx = image.to_matrix()
            red_channel = mtrx[:, :, 0]
            green_channel = mtrx[:, :, 1]
            blue_channel = mtrx[:, :, 2]

            new_red = np.apply_along_axis(function, axis, red_channel).flatten()
            new_green = np.apply_along_axis(function, axis, green_channel).flatten()
            new_blue = np.apply_along_axis(function, axis, blue_channel).flatten()

            pixels_raw = np.array([[new_red[i], new_green[i], new_blue[i]] for i in range(len(new_red))]).flatten()

        new_image = RAWImage(
            color_mode=image.color_mode,
            height=image.height,
            width=image.width,
            pixels_raw=pixels_raw
        )

        return new_image

    def haar_encode(self, image, times):
        processor = HaarWavelet()
        vertically = image
        while times:
            horizontally = self.apply_haar(processor.encode, vertically, axis=0)
            vertically = self.apply_haar(processor.encode, horizontally, axis=1)
            times -= 1
        return vertically

    def haar_decode(self, image, times):
        processor = HaarWavelet()
        horizontally = image
        while times:
            vertically = self.apply_haar(processor.decode, horizontally, axis=1)
            horizontally = self.apply_haar(processor.decode, vertically, axis=0)
            times -= 1
        horizontally.pixels_raw = np.abs(horizontally.pixels_raw).round(0).astype('uint8')
        return horizontally
