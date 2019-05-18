from utils.audio_processor import AudioProcessorWAVE
import sys
import optparse


def main(options):

    filename = ''.join(options.input.split('.')[:-1])
    extension = options.input.split('.')[-1]
    processor = AudioProcessorWAVE()

    try:
        audio = processor.read(options.input)

        if options.encode:
            if options.apply_mu:
                new_audio = processor.mu_law(audio, encode=True)
                #new_audio = processor.mu_law(new_audio, encode=False)
            elif options.apply_alpha:
                new_audio = processor.alpha_law(audio, encode=True)
                #new_audio = processor.alpha_law(new_audio, encode=False)

        if options.decode:
            if options.apply_mu:
                new_audio = processor.mu_law(audio, encode=False)
            elif options.apply_alpha:
                new_audio = processor.alpha_law(audio, encode=False)

        if not options.output:
            filename_output = filename + '_tmp.' + extension
        else:
            filename_output = options.output

        processor.write(new_audio, filename_output)

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
    parser.add_option('--encode', dest='encode', action='store_true', default=True,
                      help='применить прямое преобразование')
    parser.add_option('--decode', dest='decode', action='store_true', default=False,
                      help='применить обратное преобразование')
    parser.add_option('--mu', dest='apply_mu', action='store_true', default=False,
                      help='применить мю-закон')
    parser.add_option('--alpha', dest='apply_alpha', action='store_true', default=False,
                      help='применить альфа-закон')

    options, args = parser.parse_args()
    main(options)
