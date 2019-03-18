import optparse
import sys
from utils.image_processor import ImageProcessor


if __name__ == '__main__':
    parser = optparse.OptionParser(usage='Usage: %prog [options] [args]')
    parser.add_option('--input', dest='input', action='store', default='', type='str',
                      help='путь к *.ppm или *.pgm файлу')
    parser.add_option('--output', dest='output', action='store', default='', type='str',
                      help='путь к файлу, который следует записать.')
    parser.add_option('--compressor', dest='compressor', action='store', default='', type='str',
                      help='алгоритм сжатия: rle | lz77 | huffman.')
    parser.add_option('--compress', dest='compress', action='store_true', default=False,
                      help='используйте этот флаг, если нужно сжать файл')
    parser.add_option('--window', dest='window_size', action='store', default=64, type='int',
                      help='размер окна для сжатия LZ77')

    options, args = parser.parse_args()

    filename = ''.join(options.input.split('.')[:-1])
    type = options.input.split('.')[-1]
    proc = ImageProcessor()

    try:
        if options.compressor:
            compressor_name = options.compressor.lower()
            if options.compress:
                image = proc.read(options.input)
                print(image.to_matrix().shape)
                if compressor_name == 'rle':
                    print('Сжатие изображения при помощи алгоритма RLE.')
                    result = proc.compress_rle(image)
                elif compressor_name == 'lz77':
                    print('Сжатие изображения при помощи алгоритма LZ77.')
                    result = proc.compress_lz77(image, options.window_size)
                elif compressor_name == 'huffman':
                    print('Сжатие изображения при помощи алгоритма Хаффмана.')
                    result = proc.compress_huffman(image)
                if not options.output:
                    filename_output = filename + '.' + type + '.' + options.compressor.lower()
                else:
                    filename_output = options.output
                proc.write_compressed(filename_output, result)

                length_old = len(' '.join([str(value) for value in image.pixels_raw]).strip())
                print('Длина оригинала:', length_old)
                length_new = len(' '.join([str(value) for value in result.pixels_raw]).strip())
                print('Длина сжатого изображения:', length_new)
                print('Коэффициент сжатия:', length_new / length_old)

            else:
                image = proc.read(options.input, compressed=True)

                if options.compressor.lower() == 'rle':
                    print('Декомпрессия изображения при помощи алгоритма RLE.')
                    result = proc.decompress_rle(image)
                    if not options.output:
                        filename_output = ''.join(options.input.split('.')[:-2] + ['.'] + options.input.split('.')[-2:-1])
                    else:
                        filename_output = options.output

                    proc.write(filename_output, result)

                if options.compressor.lower() == 'lz77':
                    print('Декомпрессия изображения при помощи алгоритма LZ77.')
                    result = proc.decompress_lz77(image, options.window_size)
                    if not options.output:
                        filename_output = ''.join(options.input.split('.')[:-2] + ['.'] + options.input.split('.')[-2:-1])
                    else:
                        filename_output = options.output

                    proc.write(filename_output, result)

                if options.compressor.lower() == 'huffman':
                    print('Декомпрессия изображения при помощи алгоритма Хаффмана.')
                    result = proc.decompress_huffman(image)
                    if not options.output:
                        filename_output = ''.join(options.input.split('.')[:-2] + ['.'] + options.input.split('.')[-2:-1])
                    else:
                        filename_output = options.output

                    proc.write(filename_output, result)

        else:
            image = proc.read(options.input)
            result = image
            filename_output = filename+'_tmp.'+type
            proc.write(filename_output, result)

        print('Файл {} успешно записан.'.format(filename_output))

    except FileNotFoundError:
        print('Проверьте путь к файлу.')
        sys.exit(1)
