import array

class StandardPostings:
    """ 
    Class dengan static methods, untuk mengubah representasi postings list
    yang awalnya adalah List of integer, berubah menjadi sequence of bytes.
    Kita menggunakan Library array di Python.

    ASUMSI: postings_list untuk sebuah term MUAT di memori!

    Silakan pelajari:
        https://docs.python.org/3/library/array.html
    """

    @staticmethod
    def encode(postings_list):
        """
        Encode postings_list menjadi stream of bytes

        Parameters
        ----------
        postings_list: List[int]
            List of docIDs (postings)

        Returns
        -------
        bytes
            bytearray yang merepresentasikan urutan integer di postings_list
        """
        # Untuk yang standard, gunakan L untuk unsigned long, karena docID
        # tidak akan negatif. Dan kita asumsikan docID yang paling besar
        # cukup ditampung di representasi 4 byte unsigned.
        return array.array('L', postings_list).tobytes()

    @staticmethod
    def decode(encoded_postings_list):
        """
        Decodes postings_list dari sebuah stream of bytes

        Parameters
        ----------
        encoded_postings_list: bytes
            bytearray merepresentasikan encoded postings list sebagai keluaran
            dari static method encode di atas.

        Returns
        -------
        List[int]
            list of docIDs yang merupakan hasil decoding dari encoded_postings_list
        """
        decoded_postings_list = array.array('L')
        decoded_postings_list.frombytes(encoded_postings_list)
        return decoded_postings_list.tolist()

    @staticmethod
    def encode_tf(tf_list):
        """
        Encode list of term frequencies menjadi stream of bytes

        Parameters
        ----------
        tf_list: List[int]
            List of term frequencies

        Returns
        -------
        bytes
            bytearray yang merepresentasikan nilai raw TF kemunculan term di setiap
            dokumen pada list of postings
        """
        return StandardPostings.encode(tf_list)

    @staticmethod
    def decode_tf(encoded_tf_list):
        """
        Decodes list of term frequencies dari sebuah stream of bytes

        Parameters
        ----------
        encoded_tf_list: bytes
            bytearray merepresentasikan encoded term frequencies list sebagai keluaran
            dari static method encode_tf di atas.

        Returns
        -------
        List[int]
            List of term frequencies yang merupakan hasil decoding dari encoded_tf_list
        """
        return StandardPostings.decode(encoded_tf_list)

class VBEPostings:
    """ 
    Berbeda dengan StandardPostings, dimana untuk suatu postings list,
    yang disimpan di disk adalah sequence of integers asli dari postings
    list tersebut apa adanya.

    Pada VBEPostings, kali ini, yang disimpan adalah gap-nya, kecuali
    posting yang pertama. Barulah setelah itu di-encode dengan Variable-Byte
    Enconding algorithm ke bytestream.

    Contoh:
    postings list [34, 67, 89, 454] akan diubah dulu menjadi gap-based,
    yaitu [34, 33, 22, 365]. Barulah setelah itu di-encode dengan algoritma
    compression Variable-Byte Encoding, dan kemudian diubah ke bytesream.

    ASUMSI: postings_list untuk sebuah term MUAT di memori!

    """

    @staticmethod
    def vb_encode_number(number):
        """
        Encodes a number using Variable-Byte Encoding
        Lihat buku teks kita!
        """
        bytes = []
        while True:
            bytes.insert(0, number % 128) # prepend ke depan
            if number < 128:
                break
            number = number // 128
        bytes[-1] += 128 # bit awal pada byte terakhir diganti 1
        return array.array('B', bytes).tobytes()

    @staticmethod
    def vb_encode(list_of_numbers):
        """ 
        Melakukan encoding (tentunya dengan compression) terhadap
        list of numbers, dengan Variable-Byte Encoding
        """
        bytes = []
        for number in list_of_numbers:
            bytes.append(VBEPostings.vb_encode_number(number))
        return b"".join(bytes)

    @staticmethod
    def encode(postings_list):
        """
        Encode postings_list menjadi stream of bytes (dengan Variable-Byte
        Encoding). JANGAN LUPA diubah dulu ke gap-based list, sebelum
        di-encode dan diubah ke bytearray.

        Parameters
        ----------
        postings_list: List[int]
            List of docIDs (postings)

        Returns
        -------
        bytes
            bytearray yang merepresentasikan urutan integer di postings_list
        """
        gap_postings_list = [postings_list[0]]
        for i in range(1, len(postings_list)):
            gap_postings_list.append(postings_list[i] - postings_list[i-1])
        return VBEPostings.vb_encode(gap_postings_list)

    @staticmethod
    def encode_tf(tf_list):
        """
        Encode list of term frequencies menjadi stream of bytes

        Parameters
        ----------
        tf_list: List[int]
            List of term frequencies

        Returns
        -------
        bytes
            bytearray yang merepresentasikan nilai raw TF kemunculan term di setiap
            dokumen pada list of postings
        """
        return VBEPostings.vb_encode(tf_list)

    @staticmethod
    def vb_decode(encoded_bytestream):
        """
        Decoding sebuah bytestream yang sebelumnya di-encode dengan
        variable-byte encoding.
        """
        n = 0
        numbers = []
        decoded_bytestream = array.array('B')
        decoded_bytestream.frombytes(encoded_bytestream)
        bytestream = decoded_bytestream.tolist()
        for byte in bytestream:
            if byte < 128:
                n = 128 * n + byte
            else:
                n = 128 * n + (byte - 128)
                numbers.append(n)
                n = 0
        return numbers

    @staticmethod
    def decode(encoded_postings_list):
        """
        Decodes postings_list dari sebuah stream of bytes. JANGAN LUPA
        bytestream yang di-decode dari encoded_postings_list masih berupa
        gap-based list.

        Parameters
        ----------
        encoded_postings_list: bytes
            bytearray merepresentasikan encoded postings list sebagai keluaran
            dari static method encode di atas.

        Returns
        -------
        List[int]
            list of docIDs yang merupakan hasil decoding dari encoded_postings_list
        """
        decoded_postings_list = VBEPostings.vb_decode(encoded_postings_list)
        total = decoded_postings_list[0]
        ori_postings_list = [total]
        for i in range(1, len(decoded_postings_list)):
            total += decoded_postings_list[i]
            ori_postings_list.append(total)
        return ori_postings_list

    @staticmethod
    def decode_tf(encoded_tf_list):
        """
        Decodes list of term frequencies dari sebuah stream of bytes

        Parameters
        ----------
        encoded_tf_list: bytes
            bytearray merepresentasikan encoded term frequencies list sebagai keluaran
            dari static method encode_tf di atas.

        Returns
        -------
        List[int]
            List of term frequencies yang merupakan hasil decoding dari encoded_tf_list
        """
        return VBEPostings.vb_decode(encoded_tf_list)

class EliasGammaPostings:
    """
    Bit-level postings compression using Elias-Gamma coding.

    Elias-Gamma represents a positive integer n as:
        unary(floor(log2(n))) + binary(n - 2^floor(log2(n)))

    That is, k = floor(log2(n)) zero-bits, a stop bit of 1, then the
    k least-significant bits of n (the binary suffix after the implicit
    leading 1).

    Examples:
        n=1  -> "1"           (1 bit)
        n=2  -> "010"         (3 bits)
        n=3  -> "011"         (3 bits)
        n=4  -> "00100"       (5 bits)
        n=9  -> "0001001"     (7 bits)

    Postings are gap-encoded before compression (same strategy as VBE).
    TF values are encoded directly (no gap encoding needed).

    Because bit-packing with zero-padding makes the end of a stream
    ambiguous, the count of encoded integers is stored as a 4-byte
    big-endian prefix, so the decoder knows exactly how many numbers
    to read.

    On-disk format: [4 bytes: count] [ceil(total_bits/8) bytes: packed bits]

    ASUMSI: postings_list untuk sebuah term MUAT di memori!
    """

    @staticmethod
    def _encode_number(n, bits):
        """
        Append the Elias-Gamma encoding of n (n >= 1) to bits.

        Parameters
        ----------
        n : int
            Positive integer to encode.
        bits : list
            Bit buffer to append to (mutated in place).
        """
        k = n.bit_length() - 1      # floor(log2(n))
        for _ in range(k):          # k zero-bits (unary prefix)
            bits.append(0)
        bits.append(1)              # stop bit
        for i in range(k - 1, -1, -1):  # k-bit binary suffix (MSB first)
            bits.append((n >> i) & 1)

    @staticmethod
    def _decode_number(bits, pos):
        """
        Decode one Elias-Gamma number from bits starting at pos.

        Parameters
        ----------
        bits : list of int
            Flat bit array.
        pos : int
            Current read position.

        Returns
        -------
        (int, int)
            Decoded value and updated position.
        """
        k = 0
        while pos < len(bits) and bits[pos] == 0:
            k += 1
            pos += 1
        pos += 1        # skip stop bit
        n = 1           # implicit leading 1
        for _ in range(k):
            n = (n << 1) | bits[pos]
            pos += 1
        return n, pos

    @staticmethod
    def _bits_to_bytes(bits):
        """
        Pack a bit list into a byte string (MSB first).
        The last byte is zero-padded on the right if needed.
        """
        result = bytearray()
        for i in range(0, len(bits), 8):
            chunk = bits[i:i + 8]
            byte = 0
            for b in chunk:
                byte = (byte << 1) | b
            byte <<= (8 - len(chunk))   # right-pad last chunk
            result.append(byte)
        return bytes(result)

    @staticmethod
    def _bytes_to_bits(data):
        """Unpack a byte string into a flat bit list (MSB first)."""
        bits = []
        for byte in data:
            for i in range(7, -1, -1):
                bits.append((byte >> i) & 1)
        return bits

    @staticmethod
    def encode(postings_list):
        """
        Encode a postings list with gap encoding + Elias-Gamma compression.

        DocIDs may start at 0, but Elias-Gamma requires n >= 1, so the
        first gap (the first docID itself) is stored as (docID + 1).
        Subsequent gaps are always >= 1 because docIDs are strictly
        increasing.

        Parameters
        ----------
        postings_list : List[int]
            Sorted list of docIDs.

        Returns
        -------
        bytes
            4-byte count prefix followed by the packed bit stream.
        """
        gaps = [postings_list[0] + 1]           # +1: handle docID=0
        for i in range(1, len(postings_list)):
            gaps.append(postings_list[i] - postings_list[i - 1])   # >= 1

        bits = []
        for g in gaps:
            EliasGammaPostings._encode_number(g, bits)

        count_bytes = len(gaps).to_bytes(4, byteorder='big')
        return count_bytes + EliasGammaPostings._bits_to_bytes(bits)

    @staticmethod
    def decode(encoded_postings_list):
        """
        Decode an Elias-Gamma encoded postings list.

        Parameters
        ----------
        encoded_postings_list : bytes
            Output of encode().

        Returns
        -------
        List[int]
            Reconstructed sorted list of docIDs.
        """
        count = int.from_bytes(encoded_postings_list[:4], byteorder='big')
        bits  = EliasGammaPostings._bytes_to_bits(encoded_postings_list[4:])

        pos  = 0
        gaps = []
        for _ in range(count):
            n, pos = EliasGammaPostings._decode_number(bits, pos)
            gaps.append(n)

        postings = [gaps[0] - 1]                # undo the +1 offset
        for i in range(1, len(gaps)):
            postings.append(postings[-1] + gaps[i])
        return postings

    @staticmethod
    def encode_tf(tf_list):
        """
        Encode a TF list with Elias-Gamma compression (no gap encoding).

        TF values are always >= 1, so no offset is needed.

        Parameters
        ----------
        tf_list : List[int]
            List of term frequencies.

        Returns
        -------
        bytes
            4-byte count prefix followed by the packed bit stream.
        """
        bits = []
        for tf in tf_list:
            EliasGammaPostings._encode_number(tf, bits)

        count_bytes = len(tf_list).to_bytes(4, byteorder='big')
        return count_bytes + EliasGammaPostings._bits_to_bytes(bits)

    @staticmethod
    def decode_tf(encoded_tf_list):
        """
        Decode an Elias-Gamma encoded TF list.

        Parameters
        ----------
        encoded_tf_list : bytes
            Output of encode_tf().

        Returns
        -------
        List[int]
            Reconstructed list of term frequencies.
        """
        count = int.from_bytes(encoded_tf_list[:4], byteorder='big')
        bits  = EliasGammaPostings._bytes_to_bits(encoded_tf_list[4:])

        pos     = 0
        tf_list = []
        for _ in range(count):
            n, pos = EliasGammaPostings._decode_number(bits, pos)
            tf_list.append(n)
        return tf_list


if __name__ == '__main__':

    postings_list = [34, 67, 89, 454, 2345738]
    tf_list = [12, 10, 3, 4, 1]
    for Postings in [StandardPostings, VBEPostings, EliasGammaPostings]:
        print(Postings.__name__)
        encoded_postings_list = Postings.encode(postings_list)
        encoded_tf_list = Postings.encode_tf(tf_list)
        print("byte hasil encode postings: ", encoded_postings_list)
        print("ukuran encoded postings   : ", len(encoded_postings_list), "bytes")
        print("byte hasil encode TF list : ", encoded_tf_list)
        print("ukuran encoded postings   : ", len(encoded_tf_list), "bytes")

        decoded_posting_list = Postings.decode(encoded_postings_list)
        decoded_tf_list = Postings.decode_tf(encoded_tf_list)
        print("hasil decoding (postings): ", decoded_posting_list)
        print("hasil decoding (TF list) : ", decoded_tf_list)
        assert decoded_posting_list == postings_list, "hasil decoding tidak sama dengan postings original"
        assert decoded_tf_list == tf_list, "hasil decoding tidak sama dengan postings original"
        print()
