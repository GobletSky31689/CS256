from ast import literal_eval as make_tuple
import matplotlib as mpl
mpl.use('TkAgg')
from matplotlib import pyplot as plt


SET_NAME = 'Set3'

# Updates could be less than equal to 40, but we take array to be size 40
m = [0]*40

for i in range(1, 11):
    result_file = open(SET_NAME+'/Experiment%s' % i + '/experiment_result.txt', 'r')
    result = result_file.readlines()
    for line in result:
        x = make_tuple(line.rstrip('\n'))
        if x[0] >= len(m):
            break
        m[x[0]] = m[x[0]] + x[1]
    result_file.close()

for i in range(len(m)):
    m[i] = m[i]/10
m = m[1:]
xses = []
yses = []
for i in range(len(m)):
    xses.append(i+1)
    yses.append(m[i])
plt.gca().set_ylim([0, 1])
plt.plot(xses, yses, 'r-')
plt.xlabel("No of Updates (m)")
plt.ylabel("Average Error (delta)")
plt.savefig(SET_NAME+'_result_threshold.png')
# plt.show()
