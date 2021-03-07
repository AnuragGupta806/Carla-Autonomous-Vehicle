import  matplotlib.pyplot as plt
import  csv
original_x=[]
original_y=[]

stanley_x = []
stanley_y = []

pure_x = []
pure_y = [] 

mpl_x = []
mpl_y = []


with open('D:\\Carla All\\Stanley\\racetrack.csv','r') as csvfile:
    plots =csv.reader(csvfile,delimiter=',')
    count = 0
    for column in plots:
        try:
            f1 = (float)(column[0])
            # print(f1)
        except:
            continue
        
        original_x.append((float)(column[0]))
        original_y.append((float)(column[1]))
        # print('Values {} and {}'.format( column[0], column[1]))
        # print(type(column[0]))
        if count > 1000:
            break


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
        stanley_x.append((float)(column[0]))
        stanley_y.append((float)(column[1]))
        if count > 1000:
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
        pure_x.append((float)(column[0]))
        pure_y.append((float)(column[1]))
        if count > 1000:
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
        mpl_x.append((float)(column[0]))
        mpl_y.append((float)(column[1]))
        # print('Values {} and {}'.format( column[0], column[1]))
        # print(type(column[0]))
        if count > 1000:
            break



for i in range(len(stanley_x)) :
    stanley_x[i] = original_x[i] - stanley_x[i]
    stanley_y[i] = original_y[i] - stanley_y[i]


for i in range(len(pure_x)) :
    pure_x[i] = original_x[i] - pure_x[i]
    pure_y[i] = original_y[i] - pure_y[i]

for i in range(len(mpl_x)):
    mpl_x[i] = original_x[i] - mpl_x[i]
    mpl_y[i] = original_y[i] - mpl_y[i]

sum_error1 = [0]
sum_error2 = [0]
sum_error3 = [0]



for i in range(1, 1000):
    k = (stanley_x[i]**2 + stanley_y[i]**2)**0.5
    k = k/1000
    k += sum_error1[i-1]
    sum_error1.append(k)
    print(k)

for i in range(1, 1000):
    k = (pure_x[i]**2 + pure_y[i]**2)**0.5
    k = k/1000
    k += sum_error2[i-1]
    sum_error2.append(k)
    print(k)


for i in range(1, 1000):
    k = (mpl_x[i]**2 + mpl_y[i]**2)**0.5
    k = k/1000
    k += sum_error3[i-1]
    sum_error3.append(k)
    print(k)


plt.plot(sum_error1, label='Error in Stanley Controller')
plt.xlabel('Number of Iteration')
plt.ylabel('Error Sum in Each Controller')
plt.plot(sum_error2, label='Error in PurePursuit')
plt.plot(sum_error3, color = 'black',label='Error in MPL')
plt.title('Sum of error in Stantley Controller output')
plt.legend()
plt.show()

