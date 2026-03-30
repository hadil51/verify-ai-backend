from collections import OrderedDict
from datetime import datetime


class MRZ(object):
    def __init__(self, mrz_lines):
        self._parse(mrz_lines)
        self.aux = {}

    @staticmethod
    def from_ocr(mrz_ocr_string):
        result = MRZ(MRZOCRCleaner.apply(mrz_ocr_string))
        result.aux["text"] = mrz_ocr_string
        result.aux["raw_text"] = mrz_ocr_string
        return result

    @staticmethod
    def _guess_type(mrz_lines):
        try:
            if len(mrz_lines) == 3:
                return "TD1"
            elif len(mrz_lines) == 2 and len(mrz_lines[0]) < 40 and len(mrz_lines[1]) < 40:
                return "MRVB" if mrz_lines[0][0].upper() == "V" else "TD2"
            elif len(mrz_lines) == 2:
                return "MRVA" if mrz_lines[0][0].upper() == "V" else "TD3"
            else:
                return None
        except Exception:
            return None

    def _parse(self, mrz_lines):
        self.mrz_type = MRZ._guess_type(mrz_lines)
        try:
            if self.mrz_type == "TD1":
                self.valid = self._parse_td1(*mrz_lines)
            elif self.mrz_type == "TD2":
                self.valid = self._parse_td2(*mrz_lines)
            elif self.mrz_type == "TD3":
                self.valid = self._parse_td3(*mrz_lines)
            elif self.mrz_type == "MRVA":
                self.valid = self._parse_mrv(*mrz_lines, length=44)
            elif self.mrz_type == "MRVB":
                self.valid = self._parse_mrv(*mrz_lines, length=36)
            else:
                self.valid = False
                self.valid_score = 0
        except Exception:
            self.mrz_type = None
            self.valid = False
            self.valid_score = 0

    def to_dict(self):
        result = OrderedDict()
        result["mrz_type"] = self.mrz_type
        result["valid_score"] = self.valid_score
        if "raw_text" in self.aux:
            result["raw_text"] = self.aux["raw_text"]
        if self.mrz_type is not None:
            result["type"] = self.type
            result["country"] = self.country
            result["number"] = self.number
            result["date_of_birth"] = self.date_of_birth
            result["expiration_date"] = self.expiration_date
            result["nationality"] = self.nationality
            result["sex"] = self.sex
            result["names"] = self.names
            result["surname"] = self.surname
            if self.mrz_type == "TD1":
                result["optional1"] = self.optional1
                result["optional2"] = self.optional2
            elif self.mrz_type in ["TD2", "MRVA", "MRVB"]:
                result["optional1"] = self.optional1
            else:
                result["personal_number"] = self.personal_number
            result["check_number"] = self.check_number
            result["check_date_of_birth"] = self.check_date_of_birth
            result["check_expiration_date"] = self.check_expiration_date
            if self.mrz_type not in ["MRVA", "MRVB"]:
                result["check_composite"] = self.check_composite
            if self.mrz_type == "TD3":
                result["check_personal_number"] = self.check_personal_number
            result["valid_number"] = self.valid_check_digits[0]
            result["valid_date_of_birth"] = self.valid_check_digits[1]
            result["valid_expiration_date"] = self.valid_check_digits[2]
            if self.mrz_type not in ["MRVA", "MRVB"]:
                result["valid_composite"] = self.valid_check_digits[3]
            if self.mrz_type == "TD3":
                result["valid_personal_number"] = self.valid_check_digits[4]
            if "method" in self.aux:
                result["method"] = self.aux["method"]
        return result

    def _parse_td1(self, a, b, c):
        len_a, len_b, len_c = len(a), len(b), len(c)
        if len(a) < 30:
            a = a + "<" * (30 - len(a))
        if len(b) < 30:
            b = b + "<" * (30 - len(b))
        if len(c) < 30:
            c = c + "<" * (30 - len(c))
        self.type = a[0:2]
        self.country = a[2:5]
        self.number = a[5:14]
        self.check_number = a[14]
        self.optional1 = a[15:30]
        self.date_of_birth = b[0:6]
        self.check_date_of_birth = b[6]
        self.sex = b[7]
        self.expiration_date = b[8:14]
        self.check_expiration_date = b[14]
        self.nationality = b[15:18]
        self.optional2 = b[18:29]
        self.check_composite = b[29]
        surname_names = c.split("<<", 1)
        if len(surname_names) < 2:
            surname_names += [""]
        self.surname, self.names = surname_names
        self.names = self.names.replace("<", " ").strip()
        self.surname = self.surname.replace("<", " ").strip()

        self.valid_number = MRZCheckDigit.compute(self.number) == self.check_number
        self.valid_date_of_birth = MRZCheckDigit.compute(self.date_of_birth) == self.check_date_of_birth and MRZ._check_date(self.date_of_birth)
        self.valid_expiration_date = MRZCheckDigit.compute(self.expiration_date) == self.check_expiration_date and MRZ._check_date(self.expiration_date)
        self.valid_composite = MRZCheckDigit.compute(a[5:30] + b[0:7] + b[8:15] + b[18:29]) == self.check_composite
        self.valid_check_digits = [
            self.valid_number,
            self.valid_date_of_birth,
            self.valid_expiration_date,
            self.valid_composite,
        ]
        self.valid_line_lengths = [len_a == 30, len_b == 30, len_c == 30]
        self.valid_misc = [a[0] in "IAC"]
        self.valid_score = 10 * sum(self.valid_check_digits) + sum(self.valid_line_lengths) + sum(self.valid_misc) + 1
        self.valid_score = 100 * self.valid_score // (40 + 3 + 1 + 1)
        return self.valid_score == 100

    def _parse_td2(self, a, b):
        len_a, len_b = len(a), len(b)
        if len(a) < 36:
            a = a + "<" * (36 - len(a))
        if len(b) < 36:
            b = b + "<" * (36 - len(b))
        self.type = a[0:2]
        self.country = a[2:5]
        surname_names = a[5:36].split("<<", 1)
        if len(surname_names) < 2:
            surname_names += [""]
        self.surname, self.names = surname_names
        self.names = self.names.replace("<", " ").strip()
        self.surname = self.surname.replace("<", " ").strip()
        self.number = b[0:9]
        self.check_number = b[9]
        self.nationality = b[10:13]
        self.date_of_birth = b[13:19]
        self.check_date_of_birth = b[19]
        self.sex = b[20]
        self.expiration_date = b[21:27]
        self.check_expiration_date = b[27]
        self.optional1 = b[28:35]
        self.check_composite = b[35]
        self.valid_number = MRZCheckDigit.compute(self.number) == self.check_number
        self.valid_date_of_birth = MRZCheckDigit.compute(self.date_of_birth) == self.check_date_of_birth and MRZ._check_date(self.date_of_birth)
        self.valid_expiration_date = MRZCheckDigit.compute(self.expiration_date) == self.check_expiration_date and MRZ._check_date(self.expiration_date)
        self.valid_composite = MRZCheckDigit.compute(b[0:10] + b[13:20] + b[21:35]) == self.check_composite
        self.valid_check_digits = [
            self.valid_number,
            self.valid_date_of_birth,
            self.valid_expiration_date,
            self.valid_composite,
        ]
        self.valid_line_lengths = [len_a == 36, len_b == 36]
        self.valid_misc = [a[0] in "ACI"]
        self.valid_score = 10 * sum(self.valid_check_digits) + sum(self.valid_line_lengths) + sum(self.valid_misc) + 1
        self.valid_score = 100 * self.valid_score // (40 + 2 + 1 + 1)
        return self.valid_score == 100

    def _parse_td3(self, a, b):
        len_a, len_b = len(a), len(b)
        if len(a) < 44:
            a = a + "<" * (44 - len(a))
        if len(b) < 44:
            b = b + "<" * (44 - len(b))
        self.type = a[0:2]
        self.country = a[2:5]
        surname_names = a[5:44].split("<<", 1)
        if len(surname_names) < 2:
            surname_names += [""]
        self.surname, self.names = surname_names
        self.names = self.names.replace("<", " ").strip()
        self.surname = self.surname.replace("<", " ").strip()
        self.number = b[0:9]
        self.check_number = b[9]
        self.nationality = b[10:13]
        self.date_of_birth = b[13:19]
        self.check_date_of_birth = b[19]
        self.sex = b[20]
        self.expiration_date = b[21:27]
        self.check_expiration_date = b[27]
        self.personal_number = b[28:42]
        self.check_personal_number = b[42]
        self.check_composite = b[43]
        self.valid_number = MRZCheckDigit.compute(self.number) == self.check_number
        self.valid_date_of_birth = MRZCheckDigit.compute(self.date_of_birth) == self.check_date_of_birth and MRZ._check_date(self.date_of_birth)
        self.valid_expiration_date = MRZCheckDigit.compute(self.expiration_date) == self.check_expiration_date and MRZ._check_date(self.expiration_date)
        self.valid_composite = MRZCheckDigit.compute(b[0:10] + b[13:20] + b[21:43]) == self.check_composite
        self.valid_personal_number = (
            ((self.check_personal_number == "<" or self.check_personal_number == "0") and self.personal_number == "<<<<<<<<<<<<<<")
            or MRZCheckDigit.compute(self.personal_number) == self.check_personal_number
        )
        self.valid_check_digits = [
            self.valid_number,
            self.valid_date_of_birth,
            self.valid_expiration_date,
            self.valid_composite,
            self.valid_personal_number,
        ]
        self.valid_line_lengths = [len_a == 44, len_b == 44]
        self.valid_misc = [a[0] in "P"]
        self.valid_score = 10 * sum(self.valid_check_digits) + sum(self.valid_line_lengths) + sum(self.valid_misc) + 1
        self.valid_score = 100 * self.valid_score // (50 + 2 + 1 + 1)
        return self.valid_score == 100

    @staticmethod
    def _check_date(ymd):
        try:
            datetime.strptime(ymd, "%y%m%d")
            return True
        except ValueError:
            return False

    def _parse_mrv(self, a, b, length=44):
        len_a, len_b = len(a), len(b)
        if len(a) < length:
            a = a + "<" * (44 - len(a))
        if len(b) < length:
            b = b + "<" * (44 - len(b))
        self.type = a[0:2]
        self.country = a[2:5]
        surname_names = a[5:length].split("<<", 1)
        if len(surname_names) < 2:
            surname_names += [""]
        self.surname, self.names = surname_names
        self.names = self.names.replace("<", " ").strip()
        self.surname = self.surname.replace("<", " ").strip()
        self.number = b[0:9]
        self.check_number = b[9]
        self.nationality = b[10:13]
        self.date_of_birth = b[13:19]
        self.check_date_of_birth = b[19]
        self.sex = b[20]
        self.expiration_date = b[21:27]
        self.check_expiration_date = b[27]
        self.optional1 = b[28:length]
        self.valid_check_digits = [
            MRZCheckDigit.compute(self.number) == self.check_number,
            MRZCheckDigit.compute(self.date_of_birth) == self.check_date_of_birth,
            MRZCheckDigit.compute(self.expiration_date) == self.check_expiration_date,
        ]
        self.valid_line_lengths = [len_a == length, len_b == length]
        self.valid_misc = [a[0] == "V"]
        self.valid_score = 10 * sum(self.valid_check_digits) + sum(self.valid_line_lengths) + sum(self.valid_misc) + 1
        self.valid_score = 100 * self.valid_score // (30 + 2 + 1 + 1)
        self.valid_number, self.valid_date_of_birth, self.valid_expiration_date = self.valid_check_digits
        return self.valid_score == 100


class MRZOCRCleaner(object):
    def __init__(self):
        td1 = ["a*" + "A" * 3 + "*" * 9 + "N" + "*" * 15, "n" * 7 + "A" + "n" * 7 + "A" * 3 + "*" * 11 + "n", "A" * 30]
        td2 = ["a" + "A" * 35, "*" * 9 + "n" + "A" * 3 + "n" * 7 + "A" + "n" * 7 + "*" * 7 + "n" * 1]
        td3 = ["a" + "A" * 43, "*" * 9 + "n" + "A" * 3 + "n" * 7 + "A" + "n" * 7 + "*" * 14 + "n" * 2]
        mrv = ["a" + "A" * 43, "*" * 9 + "n" + "A" * 3 + "n" * 7 + "A" + "n" * 7 + "*" * 16]
        self.FORMAT = {"TD1": td1, "TD2": td2, "TD3": td3, "MRVA": mrv, "MRVB": mrv}
        a = {"0": "O", "1": "I", "2": "Z", "4": "A", "5": "S", "6": "G", "8": "B"}
        n = {"B": "8", "C": "0", "D": "0", "G": "6", "I": "1", "O": "0", "Q": "0", "S": "5", "Z": "2"}
        self.FIXERS = {"a": a, "A": a, "n": n, "N": n, "*": {}}

    def _split_lines(self, mrz_ocr_string):
        return [ln for ln in mrz_ocr_string.replace(" ", "").split("\n") if (len(ln) >= 20 or "<<" in ln)]

    def __call__(self, mrz_ocr_string):
        lines = self._split_lines(mrz_ocr_string)
        tp = MRZ._guess_type(lines)
        if tp is not None:
            for i in range(len(lines)):
                lines[i] = self._fix_line(lines[i], tp, i)
        return lines

    def _fix_line(self, line, type, line_idx):
        ln = list(line)
        for j in range(len(ln)):
            ln[j] = self._fix_char(ln[j], type, line_idx, j)
        return "".join(ln)

    def _fix_char(self, char, type, line_idx, char_idx):
        fmt = self.FORMAT[type][line_idx]
        if char_idx >= len(fmt):
            return char
        fixer = self.FIXERS[fmt[char_idx]]
        char = char.upper()
        return fixer.get(char, char)

    @staticmethod
    def apply(txt):
        if getattr(MRZOCRCleaner, "__instance__", None) is None:
            MRZOCRCleaner.__instance__ = MRZOCRCleaner()
        return MRZOCRCleaner.__instance__(txt)


class MRZCheckDigit(object):
    def __init__(self):
        self.CHECK_CODES = dict()
        for i in range(10):
            self.CHECK_CODES[str(i)] = i
        for i in range(ord("A"), ord("Z") + 1):
            self.CHECK_CODES[chr(i)] = i - 55
        self.CHECK_CODES["<"] = 0
        self.CHECK_WEIGHTS = [7, 3, 1]

    def __call__(self, txt):
        if txt == "":
            return ""
        res = sum([self.CHECK_CODES.get(c, -1000) * self.CHECK_WEIGHTS[i % 3] for i, c in enumerate(txt)])
        if res < 0:
            return ""
        return str(res % 10)

    @staticmethod
    def compute(txt):
        if getattr(MRZCheckDigit, "__instance__", None) is None:
            MRZCheckDigit.__instance__ = MRZCheckDigit()
        return MRZCheckDigit.__instance__(txt)

