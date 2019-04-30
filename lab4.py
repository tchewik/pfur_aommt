from utils.audio_processor import AudioProcessorWAVE
import sys
import optparse


def main(options):

    filename = ''.join(options.input.split('.')[:-1])
    extension = options.input.split('.')[-1]
    processor = AudioProcessorWAVE()

    try:
        audio = processor.read(options.input)

        if options.apply_window:
            audio.data = processor.apply_hanna_window(audio)

        if options.transform:
            audio.data = processor.calc_dft(audio)

        if options.itransform:
            audio.data = processor.calc_idft(audio)

        if not options.output:
            filename_output = filename + '_tmp.' + extension
        else:
            filename_output = options.output

        processor.write(audio, filename_output)

        print('Файл {} успешно записан.'.format(filename_output))

    except FileNotFoundError:
        print('Проверьте путь к файлу.')
        sys.exit(1)


if __name__ == '__main__':
    parser = optparse.OptionParser(usage='Usage: %prog [options] [args]')
    parser.add_option('--input', dest='input', action='store', default='', type='str',
                      help='путь к *.wav файлу')
    parser.add_option('--output', dest='output', action='store', default='', type='str',
                      help='путь к файлу, который следует записать.')
    parser.add_option('--transform', dest='transform', action='store_true', default=True,
                      help='применить прямое преобразование')
    parser.add_option('--itransform', dest='itransform', action='store_true', default=False,
                      help='применить обратное преобразование')
    parser.add_option('--apply_window', dest='apply_window', action='store_true', default=False,
                      help='применить оконную функцию Ханна')
    parser.add_option('--test', dest='test', action='store_true', default=False,
                      help='тестовый режим')

    options, args = parser.parse_args()
    main(options)
