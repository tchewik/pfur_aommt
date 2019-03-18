import optparse
import sys
from utils.image_processor import ImageProcessor

if __name__ == '__main__':
    parser = optparse.OptionParser(usage='Usage: %prog [options] [args]')
    parser.add_option('--input', dest='input', action='store', default='', type='str',
                      help='путь к *.ppm или *.pgm файлу')
    parser.add_option('--output', dest='output', action='store', default='', type='str',
                      help='путь к файлу, который следует записать.')
    parser.add_option('--kernel', dest='kernel', action='store', default=None,
                      help='Путь к текстовому файлу с ядром.')

    options, args = parser.parse_args()

    filename = ''.join(options.input.split('.')[:-1])
    type = options.input.split('.')[-1]
    proc = ImageProcessor()

    try:
        image = proc.read(options.input)

        with open(options.kernel, 'r') as f:
            kernel = [[float(value) for value in line.split()] for line in f.readlines()]

        result = proc.convolve(image, kernel)
        result = proc.crop_image(result, top=len(kernel), left=len(kernel))
        if not options.output:
            filename_output = filename + '_tmp.' + type
        else:
            filename_output = options.output

        proc.write(filename_output, result)

        print('Файл {} успешно записан.'.format(filename_output))

    except FileNotFoundError:
        print('Проверьте путь к файлу.')
        sys.exit(1)
