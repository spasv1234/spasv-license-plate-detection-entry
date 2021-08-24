import re

prefix_pattern = re.compile(
    r'(A[A-Z]|CB|E[A-Z]|F[A-Z]|FB[A-Z]|G[A-Z]|GB[A-Z]|P[A-Z]|Q[A-Z]|Q[A-Z][A-Z]|S[A-Z][A-Z]|SH[A-Z]|TR[A-Z]|W[A-Z]|X[A-Z]|Y[A-Z])')
number_pattern = re.compile(r'[0-9]{1,4}')
postfix_pattern = re.compile(r'[AZYXUTSRPMLKJHGEDCB]')


# This function remove any remaining noise on the license plate detection
# Whitespace will be removed and borders that are likely to be mistaken as characters removed
def remove_noise(raw_input):
    # Removes all whitespaces
    raw_input = raw_input.replace(" ", "")
    raw_input = raw_input.replace("I", "1")
    raw_input = raw_input.replace("O", "0")

    # These patterns are used to identify any border of the license plate that are read as characters by ocr
    start_border_pattern = re.compile(r'^[!Iil1|/\\_\-~=\[\{\]\}]')
    end_border_pattern = re.compile(r'[!Iil1|/\\_\-~=\[\{\]\}]$')

    # Find patterns at the start of string
    matches = start_border_pattern.finditer(raw_input)
    for match in matches:
        print(match)
        raw_input = raw_input[match.end():]
        print("start border removed: " + raw_input)

    # Find patterns at the end of string
    matches = end_border_pattern.finditer(raw_input)
    for match in matches:
        print(match)
        raw_input = raw_input[:match.start()]
        print("end border removed: " + raw_input)

    raw_input = re.sub(r'[-~_=]', '', raw_input)
    result = raw_input
    return result


# This function checks license plate against singapore license plate format,
# Taking prefix,numbers and postfix into account
# It does not take into account checksum letter calculation to determine validity of license plate
def validate_license_plate_format(license_plate_number):
    correct_license_plate_pattern = re.compile(
        r'^(A[A-Z]|CB|E[A-Z]|F[A-Z]|FB[A-Z]|G[A-Z]|GB[A-Z]|P[A-Z]|Q[A-Z]|Q[A-Z][A-Z]|S[A-Z][A-Z]|SH[A-Z]|TR[A-Z]|W[A-Z]|X[A-Z]|Y[A-Z])([0-9]{1,4})([AZYXUTSRPMLKJHGEDCB])$')

    matches = correct_license_plate_pattern.findall(license_plate_number)
    for match in matches:
        license_plate_number = ''.join(match)
        print(license_plate_number + " is in the right format")
        return True


def validate_license_plate_with_checksum(license_plate_number):
    if validate_license_plate_format(license_plate_number):
        checksum_number_list = []  # store the 6 values needed to calculate checksum

        # Find regex patterns and matches them to prefix,number,and checksum letter
        correct_license_plate_pattern = re.compile(
            r'^(A[A-Z]|CB|E[A-Z]|F[A-Z]|FB[A-Z]|G[A-Z]|GB[A-Z]|P[A-Z]|Q[A-Z]|Q[A-Z][A-Z]|S[A-Z][A-Z]|SH[A-Z]|TR[A-Z]|W[A-Z]|X[A-Z]|Y[A-Z])([0-9]{1,4})([AZYXUTSRPMLKJHGEDCB])$')
        matches = correct_license_plate_pattern.findall(license_plate_number)
        print(matches)
        prefix = str(matches[0][0])
        number = str(matches[0][1])
        checksum_letter = str(matches[0][2])

        # Allocate first 2 checksum value according to length of prefix
        # -64 because A is 65 in ASCII
        # Always take last 2 char of prefix
        # When prefix only have 1 char, the first checksum value will be 0
        if len(prefix) > 1:
            checksum_number_list.append(ord(prefix[-2]) - 64)
            checksum_number_list.append(ord(prefix[-1]) - 64)
        elif len(prefix) == 1:
            checksum_number_list.append(0)
            checksum_number_list.append(ord(prefix[-1]) - 64)

        # Add numbers to checksum value
        # Pad with 0 when number has less than 4 digit
        while len(number) < 4:
            number = "0" + number

        for i in range(0, len(number)):
            checksum_number_list.append(int(number[i]))

        # Checksum  value calculation
        checksum_value = 0
        checksum_value += checksum_number_list[0] * 9
        checksum_value += checksum_number_list[1] * 4
        checksum_value += checksum_number_list[2] * 5
        checksum_value += checksum_number_list[3] * 4
        checksum_value += checksum_number_list[4] * 3
        checksum_value += checksum_number_list[5] * 2
        checksum_value %= 19

        # Map checksum letter to corresponding number
        # check if checksum letter is correct
        checksum_letter_map = {0: "A", 1: "Z", 2: "Y", 3: "X", 4: "U", 5: "T", 6: "S", 7: "R", 8: "P", 9: "M", 10: "L",
                               11: "K", 12: "J", 13: "H", 14: "G", 15: "E", 16: "D", 17: "C", 18: "B"}
        if checksum_letter == checksum_letter_map[checksum_value]:
            print("License plate checksum validation: Correct")
            return True
        else:
            return False
