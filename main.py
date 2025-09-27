import numpy as np
from matplotlib import pyplot
i = 5
h = 15
h1 = 7
o = 1

biasCoeff = .5

np.random.seed(2)

alpha = .01
#model = {"W1":[], "W2":[], "W3":[]}
model = dict()
model["W1"] = np.random.uniform(-1, 1, (i, h))
model["W2"] = np.random.uniform(-1, 1, (h, h1))
model["W3"] = np.random.uniform(-1, 1, (h1, o))
#print("model", model)
biases = [np.zeros(h), np.zeros(h1), np.zeros(o)]


errList = []
runningError = 0
sum = {"W1":np.zeros_like(model["W1"]),"W2":np.zeros_like(model["W2"]),"W3":np.zeros_like(model["W3"]),}
sumB = [np.zeros(h), np.zeros(h1), np.zeros(o)]


def forwardPropagation(input):
    h = np.dot(input, model["W1"]) + biases[0]
    h[h < 0] *= 0.01
    h1 = np.dot(h, model["W2"]) + biases[1]
    h1[h1 < 0] *= 0.01
    o = np.dot(h1, model["W3"]) + biases[2]
    #o[o < 0] *= 0.01
    return o, h, h1


batchsize = 100
def backwardPropagation(i, h, h1, o, error):
    #error = (o-n)*2
    d3 = np.array([h1]).T*error
    #d3 = np.array([np.dot(np.array([h1]).T, error)]).T
    e1 = np.dot(model["W3"], error)
    e1[h1 < 0] *= 0.01
    d2 = np.dot(np.array([h]).T, [e1])
    e = np.dot(model["W2"], e1)
    e[h < 0] *= 0.01
    d1 = np.dot(np.array([i]).T, [e])
#    print(f'''
#d3 {d3}
#e1 {e1}
#d2 {d2}
#e {e}
#d1 {d1}
#          ''')
    return {"W1": d1, "W2": d2, "W3": d3}, [e, e1, error]



def genInO(bias = True):
    r = np.random.random()
    inL = np.zeros(5)
    out = np.random.randint(0,6)
    if np.random.uniform(0,1,1) > r and bias:
        out *= biasCoeff
        out = int(out)
    for i in range(out):
        inL[i] = r
    return inL, out

def reinforce(k):
    for j in range(k):
        global sum, sumB, runningError, model, biases
        it, ot = genInO()
        oL, hL, h1L = forwardPropagation(it)
        #print(oL)
        reward = 0  
        if it[0]<.5 and it[0] > 0 and oL > 2.3:
            reward = oL[0]*2
        elif oL > 5:
            reward = (4-oL[0])
        runningError = runningError*0.99 + (oL-ot)**2*0.01
        errList.append(runningError)    
        dm, b = backwardPropagation(it, hL, h1L, oL, np.array([-reward]))

        for k, v in dm.items():
            #print("v", v)
            #print("model", model[k])
            #print(np.std(model[k]))
            sum[k] += v
        for i in range(len(b)):
            sumB[i] += b[i]
        if j % batchsize == 0:
            for k, v in sum.items():
                model[k] -= alpha * v / batchsize
            for i in range(len(sumB)):
                biases[i] -= alpha * sumB[i] / batchsize
            sum = {"W1":np.zeros_like(model["W1"]),"W2":np.zeros_like(model["W2"]),"W3":np.zeros_like(model["W3"]),}
            sumB = [np.zeros(h), np.zeros(h1), np.zeros(o)]

#for i in range(10):
#    it, ot = genInO()
#    print(it, ot)

def test(a, b):
    it = [0,0,0,0,0]
    for i in range(b):
        it[i] = a
    ot = b
    o, h, h1 = forwardPropagation(it)
    print(it, ot, o)


def learn(num, bias = True):
    global runningError, sum, sumB
    for j in range(num): 
        it, ot = genInO(bias)
        #print("it, ot", it, ot)
        oL, hL, h1L = forwardPropagation(it)
        #print((o-ot)**2)
        runningError = runningError*0.99 + (oL-ot)**2*0.01
        errList.append(runningError)
        dm, b = backwardPropagation(it, hL, h1L, oL, (oL - ot)*2)
        for k, v in dm.items():
            #print("v", v)
            #print("model", model[k])
            #print(np.std(model[k]))
            sum[k] += v
        for i in range(len(b)):
            sumB[i] += b[i]
        if j % batchsize == 0:
            for k, v in sum.items():
                model[k] -= alpha * v / batchsize
            for i in range(len(sumB)):
                biases[i] -= alpha * sumB[i] / batchsize
            sum = {"W1":np.zeros_like(model["W1"]),"W2":np.zeros_like(model["W2"]),"W3":np.zeros_like(model["W3"]),}
            sumB = [np.zeros(h), np.zeros(h1), np.zeros(o)]
    sum = {"W1":np.zeros_like(model["W1"]),"W2":np.zeros_like(model["W2"]),"W3":np.zeros_like(model["W3"]),}
    sumB = [np.zeros(h), np.zeros(h1), np.zeros(o)]
learn(300000)

pyplot.plot(errList)
pyplot.show()

test(.1, 5)
test(.8, 5)
test(1, 5)
test(.1, 3)
test(.1, 1)
test(.8, 3)


for i in range(1):
    print("reinforcing...")
    
    for j in range(30):
        reinforce(104)
        learn(102, False)
    #learn(300000)
    
    pyplot.plot(errList)
    pyplot.show()
    
    for i in range(5):
        it, ot = genInO()
        o, h, h1 = forwardPropagation(it)
        #print (it, ot, o)
    test(.1, 5)
    test(.8, 5)
    test(1, 5)
    test(.1, 3)
    test(.1, 1)
    test(.8, 3)

