import heapq
import pickle

class RLECompressor:
    def compress(self, array):
        result = []
        counter = 1
        for i in range(1, len(array)):
            if array[i - 1] == array[i]:
                counter += 1
            else:
                if counter == 1:
                    result.append(array[i - 1])
                else:
                    result.append("%d.%s" % (counter, array[i - 1]))
                counter = 1

        if counter == 1:
            result.append(array[i])
        else:
            result.append("%d.%s" % (counter, array[i]))
        return result

    def decompress(self, array):
        result = []
        for i in array:
            if '.' in i:
                counter, unit = i.split('.')
            else:
                counter, unit = 1, i
            result += [unit] * int(counter)
        return result


class LZ77Compressor:
    DEFAULT_WINDOW_SIZE = 64

    def __init__(self, window_size=DEFAULT_WINDOW_SIZE):
        self.window_size_ = window_size
        self.look_ahead_buf_size = self.window_size_ // 3
        self.search_buf_size = self.window_size_ - self.look_ahead_buf_size

    def find_in_buffer_(self, data, current_position):
        end_of_buffer = min(current_position + self.look_ahead_buf_size, len(data) + 1)

        best_match_distance = -1
        best_match_length = -1

        for j in range(current_position, end_of_buffer):

            start_index = max(0, current_position - self.window_size_)
            substring = data[current_position:j]

            for i in range(start_index, current_position):
                matched_string = data[i:min(current_position, i + len(substring))]

                if matched_string == substring and len(substring) > best_match_length:
                    best_match_distance = current_position - i
                    best_match_length = len(substring)

        if best_match_length <= best_match_distance:
            if best_match_distance > 0 and best_match_length > 0:
                # to find the less distance
                substr_to_find = data[
                                 current_position - best_match_distance:current_position - best_match_distance + best_match_length]

                substr_best_position = data[:current_position].rfind(substr_to_find)
                best_match_d = current_position - substr_best_position
                if best_match_d < current_position:
                    best_match_distance = best_match_d

                return best_match_distance, best_match_length

        return None

    def compress(self, array):
        i = 0
        output_buffer = []
        while i < len(array):
            match = self.find_in_buffer_(array, i)
            if match:
                (bestMatchDistance, bestMatchLength) = match
                out_string = '{}.{}'.format(bestMatchDistance, bestMatchLength)
                output_buffer.append(out_string)
                i += bestMatchLength
            else:
                out_string = '{}.{}.{}'.format(0, 0, ord(array[i]))
                output_buffer.append(out_string)
                i += 1
        return output_buffer

    def decompress(self, array):
        result = []
        ar = list(array)
        counter = 0
        while len(ar):
            current_symbol = ar.pop(0).split('.')
            counter += 1
            if len(current_symbol) == 3:
                distance, length, symbol = current_symbol
                if int(distance) == 0 and int(length) == 0:
                    result += [symbol]
            else:
                distance, length = list(map(int, current_symbol))
                max = len(result)
                symbol = result[max - distance: min(max, max - distance + length)]
                result += symbol

        return result


class HeapNode:
    def __init__(self, char, freq):
        self.char = char
        self.freq = freq
        self.left = None
        self.right = None

    def __lt__(self, other):
        return self.freq < other.freq

    def __eq__(self, other):
        if not other or not isInstance(other, HeapNode):
            return False
        return self.freq == other.freq


class HuffmanCompressor:
    def __init__(self, root=None):
        self.heap = []
        self.codes = {}
        self.reverse_mapping = {}

    def make_frequency_dict(self, text):
        frequency = {}
        for character in text:
            if character not in frequency:
                frequency[character] = 0
            frequency[character] += 1
        return frequency

    def make_heap(self, frequency_dict):
        for key in frequency_dict.keys():
            node = HeapNode(key, frequency_dict[key])
            heapq.heappush(self.heap, node)

    def merge_nodes(self):
        while len(self.heap) > 1:
            node1 = heapq.heappop(self.heap)
            node2 = heapq.heappop(self.heap)
            merged = HeapNode(None, node1.freq + node2.freq)
            merged.left = node1
            merged.right = node2
            heapq.heappush(self.heap, merged)

    def make_codes_helper(self, root, current_code):
        if root:
            if root.char:
                self.codes[root.char] = current_code
                self.reverse_mapping[current_code] = root.char
                return

            self.make_codes_helper(root.left, current_code + "0")
            self.make_codes_helper(root.right, current_code + "1")

    def make_codes(self):
        root = heapq.heappop(self.heap)
        current_code = ""
        self.make_codes_helper(root, current_code)

    def get_encoded_text(self, array):
        encoded_text = ""
        for character in array:
            encoded_text += self.codes[character]
        return encoded_text

    def pad_encoded_text(self, encoded_text):
        extra_padding = 8 - len(encoded_text) % 8
        for i in range(extra_padding):
            encoded_text += "0"

        padded_info = "{0:08b}".format(extra_padding)
        encoded_text = padded_info + encoded_text
        return encoded_text

    def get_byte_array(self, padded_encoded_text):
        if len(padded_encoded_text) % 8 != 0:
            exit(0)

        b = []
        for i in range(0, len(padded_encoded_text), 8):
            byte = padded_encoded_text[i:i + 8]
            b.append(int(byte, 2))

        return b

    def compress(self, array):
        frequency = self.make_frequency_dict(array)
        self.make_heap(frequency)
        self.merge_nodes()
        self.make_codes()

        encoded_text = self.get_encoded_text(array)
        padded_encoded_text = self.pad_encoded_text(encoded_text)

        result = self.get_byte_array(padded_encoded_text)
        inv_map = {v: k for k, v in self.codes.items()}

        return (result, inv_map)

    def remove_padding(self, padded_encoded_text):
        padded_info = padded_encoded_text[:8]
        extra_padding = int(padded_info, 2)

        padded_encoded_text = padded_encoded_text[8:]
        encoded_text = padded_encoded_text[:-1 * extra_padding]

        return encoded_text

    def decode_text(self, encoded_text):
        current_code = ""
        decoded_text = []

        for bit in encoded_text:
            current_code += bit
            if (current_code in self.reverse_mapping):
                character = self.reverse_mapping[current_code]
                decoded_text.append(character)
                current_code = ""

        return decoded_text

    def decompress(self, array):
        bit_string = ""
        i = 0
        while i < len(array):
            byte = int(array[i])
            bits = bin(byte)[2:].rjust(8, '0')
            bit_string += bits
            i += 1

        encoded_text = self.remove_padding(bit_string)
        decompressed_text = self.decode_text(encoded_text)

        return decompressed_text
