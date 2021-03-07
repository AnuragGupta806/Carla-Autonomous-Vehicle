from tkinter import Label
import  matplotlib.pyplot as plt
import  csv

def difference_calculate(array):

    diff_array = []
    for i in range(1, len(array)):
        k = (array[i] - array[i-1])*100
        diff_array.append(k)

    return diff_array


stanley_s = []
pure_s = []
mpl_s = []

with open('D:\\Carla All\\Stanley\\stanley.csv','r') as csvfile:
    plots =csv.reader(csvfile,delimiter=',')
    count = 0
    for column in plots:
        try:
            f1 = (float)(column[0])
            # print(f1)
        except:
            continue
        count+=1
        stanley_s.append((float)(column[2]))
        if count > 2000:
            break

with open('D:\\Carla All\\Stanley\\pure.csv','r') as csvfile:
    plots =csv.reader(csvfile,delimiter=',')
    count = 0
    for column in plots:
        try:
            f1 = (float)(column[0])
            # print(f1)
        except:
            continue
        count+=1
        pure_s.append((float)(column[2]))
        if count > 2000:
            break

with open('D:\\Carla All\\Stanley\\mpl.csv','r') as csvfile:
    plots =csv.reader(csvfile,delimiter=',')
    count = 0
    for column in plots:
        try:
            f1 = (float)(column[0])
            # print(f1)
        except:
            continue
        count += 1
        mpl_s.append((float)(column[2]))
        print((float)(column[2]))
        # print('Values {} and {}'.format( column[0], column[1]))
        # print(type(column[0]))
        if count > 2000:
            break

diff_stanley = difference_calculate(stanley_s)
diff_mpl = difference_calculate(mpl_s)
diff_pure = difference_calculate(pure_s)

plt.plot(diff_stanley, label='Stanley')
plt.xlabel('Number of Iteration')
plt.ylabel('Change in the throttle value * 10')
plt.plot(diff_pure, label='PurePursuit')
plt.plot(diff_mpl, label='MPL')
plt.title('Graph to compare Stability of each Controller')
plt.legend()
plt.show()

