import matplotlib as mpl
from ast import literal_eval as make_tuple
mpl.use('TkAgg')
from matplotlib import pyplot as plt


TRIAL_NAME = 'Trial 1'
FILE_NAME = 'cross_only'


results = []
result_file = open(TRIAL_NAME + '/' + FILE_NAME+'.txt', 'r')
for line in result_file:
    results.append(make_tuple(line.rstrip('\n')))
result_file.close()
title = results[0]
results = results[1:]

plt.clf()
xses = []
validation__cost_yses = []
training_cost_yses = []
test_accuracy_yses = []
train_accuracy_yses = []

max_train_cost = float(results[0][1])
max_validation_cost = float(results[0][2])

for res in results:
    xses.append(float(res[0]))
    training_cost_yses.append(float(res[1])/max_train_cost)
    validation__cost_yses.append(float(res[2])/max_validation_cost)
    train_accuracy_yses.append(float(res[3]))
    test_accuracy_yses.append(float(res[4]))

plt.plot(xses, training_cost_yses, 'r-', label='training cost (normalized)')
plt.plot(xses, validation__cost_yses, 'b-', label='validation cost (normalized)')
plt.plot(xses, train_accuracy_yses, 'g-', label='training accuracy')
plt.plot(xses, test_accuracy_yses, 'y-', label='testing accuracy')
plt.legend(loc='center right')
plt.title(title)
plt.gca().set_ylim([0, 1.2])
plt.xlabel("No of Updates (epochs)")
plt.ylabel("Accuracy")
plt.savefig(TRIAL_NAME + '/' + 'epochs_vs_accuracy_'+FILE_NAME+'.png')
