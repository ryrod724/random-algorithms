# Perceptron Learning Algorithm
import random

# begin with randomly generated linear target function 
x0=random.uniform(-1, 1); x1=random.uniform(-1, 1)
y0=random.uniform(-1, 1); y1=random.uniform(-1, 1)
def f(x):
    return (y1-y0)/(x1-x0)*(x-x0)+y0

# generate dataset of N points and label them according to the target function
N = 100 
data =[]
for i in range(N):
    x = random.uniform(-1, 1)
    y = random.uniform(-1, 1)
    if y > f(x):
        data.append((x, y, 1))
    else:
        data.append((x, y, -1))

w = [0,0,0] #initial weights
iter = 0 #number of iterations
while True:
    iter += 1
    mis = []
    for (x, y, label) in data:
        if label * (w[0] + w[1]*x + w[2]*y) <= 0: # when wrong sign
            mis.append((x, y, label))
    if len(mis) == 0:
        print("final weights:", w)
        print("iterations required:", iter)
        break
    else:
        n = random.randint(0, len(mis)-1)
        (x, y, label) = mis[n]
        # update weights
        w[0] += label * 1
        w[1] += label * x
        w[2] += label * y
    
# Verification of the final weights
M = 1000; err = 0;
for i in range(M):
    x = random.uniform(-1, 1)
    y = random.uniform(-1, 1)
    if y > f(x):
        label = 1
    else:
        label = -1
    if label * (w[0] + w[1]*x + w[2]*y) <= 0:
        err += 1
print('Probability of misclassification:', err/M)
 
