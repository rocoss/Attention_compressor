import sys
python3 = sys.version_info.major >= 3

class ArithmeticCoderBase(object):

    def __init__(self, numbits):
        if numbits < 1:
            raise ValueError("State size out of range")
        self.num_state_bits = numbits
        self.full_range = 1 << self.num_state_bits
        self.half_range = self.full_range >> 1  # Non-zero
        self.quarter_range = self.half_range >> 1  # Can be zero
        self.minimum_range = self.quarter_range + 2  # At least 2
        self.maximum_total = self.minimum_range
        # Bit mask of num_state_bits ones, which is 0111...111.
        self.state_mask = self.full_range - 1
        self.low = 0
        self.high = self.state_mask

    def update(self, freqs, symbol):
        # State check
        low = self.low
        high = self.high

        range = high - low + 1

        total = int(freqs[-1])
        symlow = int(freqs[symbol-1]) if symbol > 0 else 0
        symhigh = int(freqs[symbol])

        newlow  = low + symlow  * range // total
        newhigh = low + symhigh * range // total - 1
        self.low = newlow
        self.high = newhigh

        while ((self.low ^ self.high) & self.half_range) == 0:
            self.shift()
            self.low  = ((self.low  << 1) & self.state_mask)
            self.high = ((self.high << 1) & self.state_mask) | 1
        while (self.low & ~self.high & self.quarter_range) != 0:
            self.underflow()
            self.low = (self.low << 1) ^ self.half_range
            self.high = ((self.high ^ self.half_range) << 1) | self.half_range | 1

    def shift(self):
        raise NotImplementedError()

    def underflow(self):
        raise NotImplementedError()

class ArithmeticEncoder(ArithmeticCoderBase):

    def __init__(self, numbits, bitout):
        super(ArithmeticEncoder, self).__init__(numbits)
        self.output = bitout
        self.num_underflow = 0

    def write(self, freqs, symbol):
        self.update(freqs, symbol)

    def finish(self):
        self.output.write(1)


    def shift(self):
        bit = self.low >> (self.num_state_bits - 1)
        self.output.write(bit)

        # Write out the saved underflow bits
        for _ in range(self.num_underflow):
            self.output.write(bit ^ 1)
        self.num_underflow = 0


    def underflow(self):
        self.num_underflow += 1


class ArithmeticDecoder(ArithmeticCoderBase):

    def __init__(self, numbits, bitin):
        super(ArithmeticDecoder, self).__init__(numbits)
        # The underlying bit input stream.
        self.input = bitin
        # The current raw code bits being buffered, which is always in the range [low, high].
        self.code = 0
        for _ in range(self.num_state_bits):
            self.code = self.code << 1 | self.read_code_bit()

    def read(self, freqs):

        total = int(freqs[-1])
        range = self.high - self.low + 1
        offset = self.code - self.low
        value = ((offset + 1) * total - 1) // range
        start = 0
        end = len(freqs)
        #end = freqs.get_symbol_limit()
        while end - start > 1:
            middle = (start + end) >> 1
            low = int(freqs[middle-1]) if middle > 0 else 0
            #if freqs.get_low(middle) > value:
            if low > value:
                end = middle
            else:
                start = middle

        symbol = start
        self.update(freqs, symbol)
        return symbol


    def shift(self):
        self.code = ((self.code << 1) & self.state_mask) | self.read_code_bit()


    def underflow(self):
        self.code = (self.code & self.half_range) | ((self.code << 1) & (self.state_mask >> 1)) | self.read_code_bit()

    def read_code_bit(self):
        temp = self.input.read()
        if temp == -1:
            temp = 0
        return temp

class BitInputStream(object):

    def __init__(self, inp):
        self.input = inp
        self.currentbyte = 0
        self.numbitsremaining = 0

    def read(self):
        if self.currentbyte == -1:
            return -1
        if self.numbitsremaining == 0:
            temp = self.input.read(1)
            if len(temp) == 0:
                self.currentbyte = -1
                return -1
            self.currentbyte = temp[0] if python3 else ord(temp)
            self.numbitsremaining = 8
        assert self.numbitsremaining > 0
        self.numbitsremaining -= 1
        return (self.currentbyte >> self.numbitsremaining) & 1

    def read_no_eof(self):
        result = self.read()
        if result != -1:
            return result
        else:
            raise EOFError()


    def close(self):
        self.input.close()
        self.currentbyte = -1
        self.numbitsremaining = 0

class BitOutputStream(object):

    def __init__(self, out):
        self.output = out
        self.currentbyte = 0
        self.numbitsfilled = 0

    def write(self, b):
        if b not in (0, 1):
            raise ValueError("Argument must be 0 or 1")
        self.currentbyte = (self.currentbyte << 1) | b
        self.numbitsfilled += 1
        if self.numbitsfilled == 8:
            towrite = bytes((self.currentbyte,)) if python3 else chr(self.currentbyte)
            self.output.write(towrite)
            self.currentbyte = 0
            self.numbitsfilled = 0

    def close(self):
        while self.numbitsfilled != 0:
            self.write(0)
        self.output.close()
