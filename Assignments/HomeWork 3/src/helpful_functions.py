import random


def swap(ix, jx, ax, ay):
    """The function swaps elemtents at index ix & ij for both arrays ax & ay"""
    tempx, tempy = ax[ix], ay[ix]
    ax[ix] = ax[jx]
    ay[ix] = ay[jx]
    ax[jx] = tempx
    ay[jx] = tempy


def randomize_data(arr_a, arr_b):
    """The function randomizes the array"""
    for ix in range(0, len(arr_a)-1):
        j = random.randint(ix+1, len(arr_a)-1)
        swap(ix, j, arr_a, arr_b)


def get_opposite_character(st, index):
    """The function provides the matching Gene Code, matching is A<-->C, B<-->D"""
    if st[index] is 'A':
        return 'C'
    elif st[index] is 'C':
        return 'A'
    elif st[index] is 'D':
        return 'B'
    elif st[index] is 'B':
        return 'D'


def sticky_count_wrapper(fwd_str):
    """Wrapper function is used to count the matching number of characters in the sticky gnome"""
    length = len(fwd_str)
    count = 0
    rev_index = length-1
    for index in range(length/2):
        # print fwd_str[index], " ", fwd_str[rev_index - index]
        if get_opposite_character(fwd_str, index) is not fwd_str[rev_index-index]:
            # print "Breaking the Code :",get_opposite_character(fwd_str, index), " is not equal to  ",
            # fwd_str[rev_index - index]
            break
        count += 1
    return count


def sticky_count(st):
    """Function takes in the string and outputs the number of characters that are matching in the reverse"""
    fwd_str = st.strip(" ")
    return sticky_count_wrapper(fwd_str=fwd_str)


def get_sticky_class(st):
    """This function return the sticky class corresponding to Genetic String received"""
    #   { classes and their corresponding output mapping
    # [NONSTICK -> 0],
    # [12 - STICKY -> 1],
    # [34 - STICKY -> 2],
    # [56 - STICKY -> 3],
    # [78 - STICKY ->4],
    # [STICK_PALINDROME ->5]
    # }.

    count = sticky_count(st=st)
    if count is 0:
        # print  "NONSTICK ",
        return 0
    elif count in [1, 2]:
        # print "12 - STICKY ",
        return 1
    elif count in [3, 4]:
        # print"34 - STICKY ",
        return 2
    elif count in [5, 6]:
        # print "56 - STICKY ",
        return 3
    elif 6 < count < 20:
        # print "78 - STICKY",
        return 4
    elif count is 20:
        # print "STICK_PALINDROME ",
        return 5

# print sticky_count("DDAADCBDBBADACACBDBDBDBDACACBCDDBDABCCBB") #Test String to test the function
# print get_sticky_class("DDAADCBDBBADACACBDBDBDBDACACBCDDBDABCCBB") #Test String to test the function -->Stick-Pal
# print get_sticky_class("DDAADCBDBBADACACBDAABDBDACACBCDDBDABCCBB") #Test String to test the function -->78-Sticky
# print get_sticky_class("DDAADCBABBADACACBDAABDBDACACBCDDBDABCCBB") #Test String to test the function -->78-Sticky
# print get_sticky_class("DDAADCAABBADACACBDAABDBDACACBCDDBDABCCBB") #Test String to test the function -->56-Sticky
# print get_sticky_class("DDAADABABBADACACBDAABDBDACACBCDDBDABCCBB") #Test String to test the function -->56-Sticky
# print get_sticky_class("DDADCCBABBADACACBDAABDBDACACBCDDBDABCCBB") #Test String to test the function -->34-Sticky
# print get_sticky_class("DDADDCBABBADACACBDAABDBDACACBCDDBDABCCBB") #Test String to test the function -->34-Sticky
# print get_sticky_class("DDBADCBABBADACACBDAABDBDACACBCDDBDABCCBB") #Test String to test the function -->12-Sticky
# print get_sticky_class("DAAADCBABBADACACBDAABDBDACACBCDDBDABCCBB") #Test String to test the function -->12-Sticky
# print get_sticky_class("ADAADCBABBADACACBDAABDBDACACBCDDBDABCCBB") #Test String to test the function -->NONSTICK
# print (get_opposite_character("ABCADC", 2))
