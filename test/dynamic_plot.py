from dSPEECH import config
import matplotlib.pyplot as plt
import random

plt.ion()  # turning interactive mode on

# preparing the data
y = [random.randint(1, 10) for i in range(20)]
x = [*range(1, 21)]

# plotting the first frame
fig,ax=plt.subplots()
ax.plot(x,y)
#graph = plt.plot(x, y)[0]
#plt.ylim(0, 10)
#plt.pause(1)

def test():
    y.append(random.randint(1, 10))
    x.append(x[-1] + 1)

    # removing the older graph
    #graph.remove()
    ax.clear()

    # plotting newer graph
    #graph = plt.plot(x, y, color='g')[0]
    #plt.xlim(x[0], x[-1])
    ax.plot(x,y)

    # calling pause function for 0.25 seconds
    plt.pause(0.25)


# the update loop
while (True):
    # updating the data
    test()
