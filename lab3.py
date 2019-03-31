import optparse
import sys
from utils.image_processor import ImageProcessor

if __name__ == '__main__':
    parser = optparse.OptionParser(usage='Usage: %prog [options] [args]')
    parser.add_option('--input', dest='input', action='store', default='', type='str',
                      help='путь к *.ppm или *.pgm файлу')
    parser.add_option('--output', dest='output', action='store', default='', type='str',
                      help='путь к файлу, который следует записать.')
    parser.add_option('--encode', dest='encode', action='store_true', default=False,
                      help='режим кодирования')
    parser.add_option('--decode', dest='decode', action='store_true', default=False,
                      help='режим декодирования')
    parser.add_option('--times', dest='times', action='store', default=1, type='int',
                      help='сколько раз выполнить преобразование')

    options, args = parser.parse_args()

    filename = ''.join(options.input.split('.')[:-1])
    type = options.input.split('.')[-1]
    proc = ImageProcessor()

    try:
        if options.encode or options.decode:

            if options.encode:
                image = proc.read(options.input)
                result = proc.haar_encode(image, options.times)
            else:
                image = proc.read(options.input, compressed=True)
                result = proc.haar_decode(image, options.times)

            if not options.output:
                filename_output = filename + '_tmp.' + type
            else:
                filename_output = options.output


            print('>>>', result.pixels_raw.shape)

            proc.write(filename_output, result)

            print('Файл {} успешно записан.'.format(filename_output))

    except FileNotFoundError:
        print('Проверьте путь к файлу.')
        sys.exit(1)
