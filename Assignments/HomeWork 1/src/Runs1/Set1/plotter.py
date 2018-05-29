from ast import literal_eval as make_tuple
from math import sqrt, pi
import matplotlib as mpl
import numpy as np
from shapely.geometry import LineString
from shapely.geometry import Point
import numpy.linalg as la
mpl.use('TkAgg')
from matplotlib import pyplot as plt


SET_NAME = ''
DIR_NAME = SET_NAME+'Experiment10/'  # sys.argv[1]


def plot_line(wts, l_col='black'):
    n = sqrt(sum(wt1**2 for wt1 in wts))
    ww = [wts[0]/n, wts[1]/n, wts[2]/n]
    a = ww[0]
    b = ww[1]
    c = ww[2]
    l_x = np.array(range(-2, 3, 1))
    #  For a given eq, ax + by = c
    #  y = (c - a*x)/b
    l_y = (c - a*l_x)/b
    return plt.plot(l_x, l_y, color=l_col)


def get_final_weights():
    result_file = open(DIR_NAME+'result_file.txt', 'r')
    results = []
    for line in result_file:
        results.append(line.rstrip('\n').split(" "))
    result_file.close()
    w_x = float(results[0][1].replace(",", ""))
    w_y = float(results[0][2])
    theta = int(results[1][1])
    return [w_x, w_y, theta]


def get_ideal_weights():
    result_file = open(DIR_NAME+'result_file.txt', 'r')
    results = []
    for line in result_file:
        results.append(line.rstrip('\n').split(" "))
    result_file.close()
    const_vector = map(int, results[6])
    const_vector.append(int(results[5][0]))
    return const_vector


def get_updating_rules():
    start = 0
    results = []
    result_file = open(DIR_NAME+'result_file.txt', 'r')
    for line in result_file:
        results.append(line.rstrip('\n'))
    result_file.close()
    for ind in range(len(results)):
        if results[ind].startswith("Training"):
            start = ind+1
            break
    return results[start:]


def py_ang(v1, v2):
    cosang = np.dot(v1, v2)
    sinang = la.norm(np.cross(v1, v2))
    return np.arctan2(sinang, cosang)


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


def get_linestring(wt):
    a = wt[0]
    b = wt[1]
    c = wt[2]
    return LineString([(2, (c-2*a)/b), (-2, (c+2*a)/b)])


def calculate_incorrect_arc_ratio(wt):
    p = Point(0, 0)
    circle = p.buffer(1).boundary
    line1 = get_linestring(wt)
    pts_ideal = circle.intersection(get_linestring(get_ideal_weights()))
    i = circle.intersection(line1)
    if i.geoms and len(i.geoms) == 2:
        pt1 = [i.geoms[0].x, i.geoms[0].y]
        pt2 = [i.geoms[1].x, i.geoms[1].y]
        ideal1 = [pts_ideal.geoms[0].x, pts_ideal.geoms[0].y]
        angle1 = py_ang(pt1, ideal1)
        angle2 = py_ang(pt2, ideal1)
        ratio1 = min(angle2, angle1)
        ideal2 = [pts_ideal.geoms[1].x, pts_ideal.geoms[1].y]
        angle1 = py_ang(pt1, ideal2)
        angle2 = py_ang(pt2, ideal2)
        ratio2 = min(angle2, angle1)
        return (ratio1+ratio2)/(2*pi)
    return 1


# plot_points(DIR_NAME+'train_file.txt', 'blue', 'red')
# plot_points(DIR_NAME+'test_file.txt', 'red', 'green')

# plot_line(get_final_weights(), 'red')
# plot_line(get_ideal_weights(), 'blue')


EXPERIMENT_RESULTS = []

m = 0
for rule in get_updating_rules():
    m += 1
    x = make_tuple(rule)
    sample = x[1]
    # if sample[2] == 0:
    #     plt.scatter([sample[0]], [sample[1]], color='red', s=20)
    # else:
    #     plt.scatter([sample[0]], [sample[1]], color='green', s=20)
    wt = x[0]
    l = plot_line(wt, 'black')
    plot_line(get_ideal_weights(), 'blue')
    plot_points(DIR_NAME+'train_file.txt', 'blue', 'red', alpha=0.3)
    arc = calculate_incorrect_arc_ratio(wt)
    plt.text(0, 1, 'Incorrect Arc Ratio: %s' % arc,
             horizontalalignment='left',
             verticalalignment='top',
             transform=plt.gca().transAxes)
    EXPERIMENT_RESULTS.append((m, arc))
    # plt.gca().set_ylim([-1.5, 1.5])
    # plt.gca().set_xlim([-1.5, 1.5])
    plt.axis('equal')
    # plt.savefig(DIR_NAME+'/Update%s.png' % m)
    plt.clf()


experiment_result = open(DIR_NAME+"experiment_result.txt", "w")
experiment_result.write('\n'.join(map(str, EXPERIMENT_RESULTS)))
experiment_result.close()

plt.clf()
xses = []
yses = []
for res in EXPERIMENT_RESULTS:
    xses.append(res[0])
    yses.append(res[1])
plt.gca().set_ylim([0, 1])
plt.plot(xses, yses, 'r-')
plt.xlabel("No of Updates (m)")
plt.ylabel("Error (delta)")
plt.savefig(DIR_NAME+'m_vs_delta.png')
# plt.show()
