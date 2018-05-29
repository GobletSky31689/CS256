import math
import random
import sys

ACT_TANH = "tanh"
ACT_RELU = "relu"
ACT_THRESHOLD = "threshold"
ACTIVATION = sys.argv[1]

ALGO_PERCEPTRON = "perceptron"
ALGO_WINNOW = "winnow"
TRAINING_ALGO = sys.argv[2]

FUNC_NESTED_BOOL = "NBF"
FUNC_THRESHOLD = "TF"
GROUND_FILE = sys.argv[3]

DIST_BOOL = "bool"
DIST_SPHERE = "sphere"
DISTRIBUTION = sys.argv[4]

NUM_TRAIN = int(sys.argv[5])
NUM_TEST = int(sys.argv[6])
EPSILON = float(sys.argv[7])

TRAINING_DATA = []
TESTING_DATA = []
TRAINING_UPDATES = []


def generate_input_vector(count, dist=DISTRIBUTION):
    inp_vector = []
    if dist == DIST_BOOL:
        for _ in range(count):
            inp_vector.append(random.randint(0, 1))
    elif dist == DIST_SPHERE:
        magnitude = 0
        for _ in range(count):
            # if TRAINING_ALGO == ALGO_WINNOW:
            #     random_in = random.random()
            # else:
            random_in = random.random() * 2 - 1
            inp_vector.append(random_in)
            magnitude = magnitude + random_in*random_in
        magnitude = math.sqrt(magnitude)
        for i in range(count):
            inp_vector[i] = inp_vector[i]/magnitude
    return inp_vector


def get_tf_output(const_vector, limit, input_vector):
    total = 0
    for i in range(len(const_vector)):
        total = total + const_vector[i] * input_vector[i]
    if total >= limit:
        return 1
    return 0


def get_bool_output(bool_vector, input_vector):
    exp_str = ""
    for i in range(len(bool_vector)):
        if bool_vector[i][1].isdigit():
            if bool_vector[i][0] == '-':
                exp_str = exp_str + "not "
            exp_str = exp_str + str(input_vector[abs(int(bool_vector[i])-1)]) + " "
        else:
            exp_str = exp_str + bool_vector[i].lower() + " "
    if eval(exp_str):
        return 1
    return 0


def read_nested_always_zero_func():
    for _ in xrange(NUM_TRAIN):
        train_input = generate_input_vector(5, dist=DIST_BOOL)
        train_output = 0
        train_sample = (train_input, train_output)
        # print train_sample
        TRAINING_DATA.append(train_sample)
    for _ in xrange(NUM_TEST):
        test_input = generate_input_vector(5, dist=DIST_BOOL)
        test_output = 0
        test_sample = (test_input, test_output)
        TESTING_DATA.append(test_sample)


def read_nested_bool_func(func_str):
    bool_vector = func_str.split(" ")
    max_index = 0
    for item in bool_vector:
        if item[1].isdigit() and max_index < abs(int(item)):
            max_index = abs(int(item))

    for _ in xrange(NUM_TRAIN):
        train_input = generate_input_vector(max_index, dist=DIST_BOOL)
        train_output = get_bool_output(bool_vector, train_input)
        train_sample = (train_input, train_output)
        # print train_sample
        TRAINING_DATA.append(train_sample)
    for _ in xrange(NUM_TEST):
        test_input = generate_input_vector(max_index, dist=DIST_BOOL)
        test_output = get_bool_output(bool_vector, test_input)
        test_sample = (test_input, test_output)
        TESTING_DATA.append(test_sample)


def read_threshold_func(limit_str, func_str):
    limit = int(limit_str)
    const_vector = map(int, func_str.split(" "))

    for _ in xrange(NUM_TRAIN):
        train_input = generate_input_vector(len(const_vector))
        train_output = get_tf_output(const_vector, limit, train_input)
        train_sample = (train_input, train_output)
        TRAINING_DATA.append(train_sample)
    for _ in xrange(NUM_TEST):
        test_input = generate_input_vector(len(const_vector))
        test_output = get_tf_output(const_vector, limit, test_input)
        test_sample = (test_input, test_output)
        TESTING_DATA.append(test_sample)


def dot_product(x_vector, y_vector):
    total = 0
    for i in range(len(x_vector)):
        total += x_vector[i]*y_vector[i]
    return total


def vector_diff(x_vector, y_vector):
    new_vector = []
    for i in range(len(x_vector)):
        new_vector.append(x_vector[i]-y_vector[i])
    return new_vector


def winnow_demote(weight_vector, input_vector, alpha):
    new_vector = []
    for i in range(len(weight_vector)):
        new_vector.append(weight_vector[i]/(alpha**input_vector[i]))
    return new_vector


def winnow_promote(weight_vector, input_vector, alpha):
    new_vector = []
    for i in range(len(weight_vector)):
        new_vector.append(weight_vector[i]*(alpha**input_vector[i]))
    return new_vector


def vector_add(x_vector, y_vector):
    new_vector = []
    for i in range(len(x_vector)):
        new_vector.append(x_vector[i]+y_vector[i])
    return new_vector


def train():
    if TRAINING_ALGO == ALGO_PERCEPTRON:
        return train_perceptron()
    elif TRAINING_ALGO == ALGO_WINNOW:
        return train_winnow()


# x=dot product of input vector and weight vector
def activation_func(x, theta):
    if DISTRIBUTION == DIST_SPHERE or ACTIVATION == ACT_THRESHOLD:
        if x >= theta:
            return 1
        return 0
    if ACTIVATION == ACT_RELU:
        # return max(0, x-theta)
        # Modify relu so that it binary
        if x-theta <= 0:
            return 0
        else:
            return 1
    if ACTIVATION == ACT_TANH:
        # return 1 / (1 + math.exp(-(x-theta)))
        # Modify relu so that it binary
        act = 1 / (1 + math.exp(-(x-theta)))
        if act < 0.5:
            return 0
        else:
            return 1


def train_winnow():
    num_inputs = len(TRAINING_DATA[0][0])
    node_weights = [1 for _ in range(num_inputs)]
    alpha_rate = 2
    threshold_theta = num_inputs
    for train_sample in TRAINING_DATA:
        dot_p = dot_product(train_sample[0], node_weights)
        output = activation_func(dot_p, threshold_theta)
        if output == train_sample[1]:
            # Correct guess, no update
            # print str(train_sample) + ":no update"
            pass
        elif output == 1:
            # False positive
            node_weights = winnow_demote(node_weights, train_sample[0], alpha_rate)
            TRAINING_UPDATES.append((node_weights+[threshold_theta], train_sample[0]+[train_sample[1]]))
            # print str(train_sample) + ":update"
        elif output == 0:
            node_weights = winnow_promote(node_weights, train_sample[0], alpha_rate)
            TRAINING_UPDATES.append((node_weights+[threshold_theta], train_sample[0]+[train_sample[1]]))
            # print str(train_sample) + ":update"
        # print node_weights, threshold_theta
    return node_weights, threshold_theta


def train_perceptron():
    num_inputs = len(TRAINING_DATA[0][0])
    node_weights = [0 for _ in range(num_inputs)]
    threshold_theta = 0
    for train_sample in TRAINING_DATA:
        dot_p = dot_product(train_sample[0], node_weights)
        output = activation_func(dot_p, threshold_theta)
        if output == train_sample[1]:
            # Correct guess, no update
            # print str(train_sample) + ":no update"
            pass
        elif output == 1:
            # False positive
            node_weights = vector_diff(node_weights, train_sample[0])
            threshold_theta += 1
            # print str(train_sample) + ":update"
            TRAINING_UPDATES.append((node_weights+[threshold_theta], train_sample[0]+[train_sample[1]]))
        elif output == 0:
            # False negative
            node_weights = vector_add(node_weights, train_sample[0])
            threshold_theta -= 1
            # print str(train_sample) + ":update"
            TRAINING_UPDATES.append((node_weights+[threshold_theta], train_sample[0]+[train_sample[1]]))
    return node_weights, threshold_theta


def parse_ground_file():
    ground_file = open(GROUND_FILE, "r")
    ground_func = []
    for line in ground_file:
        ground_func.append(line.rstrip('\n'))
    if ground_func[0] == FUNC_THRESHOLD:
        read_threshold_func(ground_func[1], ground_func[2])
    elif ground_func[0] == FUNC_NESTED_BOOL:
        if len(ground_func) == 1:
            read_nested_always_zero_func()
        else:
            read_nested_bool_func(ground_func[1])
    else:
        return False
    return True


# RUN_NAME = 'Runs2/'
# SET_NAME = RUN_NAME+'Set3/'
# DIR_NAME = SET_NAME+'Experiment10/'
#
#
# def write_prob_to_file():
#     train_file = open(DIR_NAME + 'train_file.txt', "w")
#     train_file.write('\n'.join(map(str, TRAINING_DATA)))
#     train_file.close()
#     test_file = open(DIR_NAME + "test_file.txt", "w")
#     test_file.write('\n'.join(map(str, TESTING_DATA)))
#     test_file.close()
#     result_file = open(DIR_NAME + "result_file.txt", "w")
#     result_file.write("WEIGHT: %s \n" % ', '.join(map(str, NODE_WEIGHTS)))
#     result_file.write("THETA: " + str(THRESHOLD_THETA) + "\n\n")
#     g_file = open(GROUND_FILE, "r")
#     g_func = []
#     for line in g_file:
#         g_func.append(line.rstrip('\n'))
#     g_file.close()
#     result_file.write("Ground Function:\n" + '\n'.join(map(str, g_func)))
#     result_file.write("\n\nArguments: " + ' '.join(map(str, sys.argv[1:])))
#     result_file.write("\n\nTraining Updates: " + str(len(TRAINING_UPDATES)) + "\n"
#                       + '\n'.join(map(str, TRAINING_UPDATES)))
#     result_file.close()

valid = parse_ground_file()
if valid:
    NODE_WEIGHTS, THRESHOLD_THETA = train()
    total_error = 0.0
    for test in TESTING_DATA:
        result = activation_func(dot_product(test[0], NODE_WEIGHTS), THRESHOLD_THETA)
        total_error = total_error + abs(result-test[1])
        print ','.join(map(str, test[0])) + ":" + str(result) + ":" + str(test[1]) + ":" + str(abs(result-test[1]))
    avg_error = total_error/NUM_TEST
    print "Average Error:" + str(avg_error)
    print "Epsilon:" + str(EPSILON)
    if avg_error < EPSILON:
        print "TRAINING SUCCEEDED"
    else:
        print "TRAINING FAILED"
    # write_prob_to_file()
else:
    print "NOT PARSEABLE"
