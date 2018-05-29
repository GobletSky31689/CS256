from ast import literal_eval as make_tuple
from math import sqrt
import matplotlib as mpl
mpl.use('TkAgg')
from matplotlib import pyplot as plt, lines as mlines
import math

SET_NAME = ''
DIR_NAME = SET_NAME+'Experiment10/'  # sys.argv[1]


TRAINING_DATA = []


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


def dot_product(x_vector, y_vector):
    total = 0
    for i in range(len(x_vector)):
        total += x_vector[i]*y_vector[i]
    return total


def read_nested_bool_func(func_str):
    bool_vector = func_str.split(" ")
    max_index = 0
    for item in bool_vector:
        if item[1].isdigit() and max_index < abs(int(item)):
            max_index = abs(int(item))
    n = max_index
    for i in range(1 << n):
        s = bin(i)[2:]
        s = '0' * (n-len(s))+s
        train_input = map(int, list(s))
        train_output = get_bool_output(bool_vector, train_input)
        TRAINING_DATA.append((train_input, train_output))
    return max_index


def plot_line(wts, l_col='black'):
    n = sqrt(sum(wt1**2 for wt1 in wts))
    ww = [wts[0]/n, wts[1]/n, wts[2]/n]
    a = ww[0]
    b = ww[1]
    c = ww[2]
    #  For a given eq, ax + by = c
    #  the normal vector is [a,b]
    if a == 0:
        a = 1
    if b == 0:
        b = 1
    pt1 = [1.0, (c-a)/b]
    if (c-a)/b == (c-b)/a:
        pt2 = [c/a, 0.0]
    else:
        pt2 = [(c-b)/a, 1.0]
    # print pt1, pt2
    # plt.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], '--k')
    return newline(pt1, pt2, plt.gca(), l_col)


def get_results_file_str():
    results = []
    result_file = open(DIR_NAME+'result_file.txt', 'r')
    for line in result_file:
        results.append(line.rstrip('\n'))
    result_file.close()
    return results


def get_updating_rules():
    start = 0
    results = get_results_file_str()
    for ind in range(len(results)):
        # print results[i]
        if results[ind].startswith("Training"):
            # print results[i]
            start = ind+1
            break
    return results[start:]


def newline(p1, p2, ax, l_col):
    xmin, xmax = ax.get_xbound()
    if p2[0] == p1[0]:
        xmin = xmax = p1[0]
        ymin, ymax = ax.get_ybound()
    else:
        ymax = p1[1]+(p2[1]-p1[1])/(p2[0]-p1[0])*(xmax-p1[0])
        ymin = p1[1]+(p2[1]-p1[1])/(p2[0]-p1[0])*(xmin-p1[0])
    lin = mlines.Line2D([xmin, xmax], [ymin, ymax], color=l_col)
    # backup_x, backup_y = ax.get_xbound(), ax.get_ybound()
    lin.set_linewidth(0.2)
    ax.add_line(lin)
    ax.set_xlim([-0.5, 1.5])
    ax.set_ylim([-0.5, 1.5])
    return lin


def plot_points(file_name, color1, color2, limit=None, alpha=1.0):
    test_file = open(file_name, 'r')
    samples = []
    for li in test_file:
        samples.append(make_tuple(li.rstrip('\n')))
    test_file.close()

    positives_xs = []
    positives_ys = []
    negatives_xs = []
    negatives_ys = []
    for item in samples:
        if item == limit:
            plt.scatter(positives_xs, positives_ys, color=color1, s=2)
            plt.scatter(negatives_xs, negatives_ys, color=color2, s=2)
            return
        if item[1] == 1:
            positives_xs.append(item[0][0])
            positives_ys.append(item[0][1])
        else:
            negatives_xs.append(item[0][0])
            negatives_ys.append(item[0][1])
    plt.scatter(positives_xs, positives_ys, color=color1, s=2, alpha=alpha)
    plt.scatter(negatives_xs, negatives_ys, color=color2, s=2, alpha=alpha)


plot_points(DIR_NAME+'train_file.txt', 'blue', 'red')
# plot_points('test_file.txt', 'red', 'green')


EXPERIMENT_RESULTS = []


NUM_INPUTS = read_nested_bool_func(get_results_file_str()[5])


def calculate_error(wt):
    error_count = 0
    for item in TRAINING_DATA:
        if dot_product(item[0], wt[:-1]) >= wt[-1]:
            my_output = 1
        else:
            my_output = 0
        if my_output != item[1]:
            error_count += 1
    return error_count*1.0/len(TRAINING_DATA)


m = 0
for rule in get_updating_rules():
    m += 1
    x = make_tuple(rule)
    wt = x[0]
    error = calculate_error(wt)
    EXPERIMENT_RESULTS.append((m, error))
    # if NUM_INPUTS == 2:
    #     l = plot_line(wt, 'black')
    #     plt.text(0, 1, 'Error Count: %s' % error,
    #              horizontalalignment='left',
    #              verticalalignment='top',
    #              transform=plt.gca().transAxes)
    #     plt.savefig(DIR_NAME+'/Update%s.png' % m)
    #     plt.gca().lines.remove(l)


experiment_result = open(DIR_NAME+"experiment_result.txt", "w")
experiment_result.write('\n'.join(map(str, EXPERIMENT_RESULTS)))
experiment_result.close()

plt.clf()
xses = []
yses = []
for res in EXPERIMENT_RESULTS:
    xses.append(res[0])
    yses.append(res[1]/math.pow(2, NUM_INPUTS))
plt.gca().set_ylim([0, 1])
plt.plot(xses, yses, 'r-')
plt.xlabel("No of Updates (m)")
plt.ylabel("Error (delta)")
plt.savefig(DIR_NAME+'m_vs_delta.png')
# plt.show()
