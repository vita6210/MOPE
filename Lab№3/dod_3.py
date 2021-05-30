import random
import numpy as np
from numpy.linalg import solve

def lab():
    x1min = -20
    x1max = 15
    x2min = -35
    x2max = 10
    x3min = 10
    x3max = 20

    ymax = 200 + (x1max + x2max + x3max) / 3
    ymin = 200 + (x1min + x2min + x3min) / 3

    xn = [[1, 1, 1, 1],
          [-1, -1, 1, 1],
          [-1, 1, -1, 1],
          [-1, 1, 1, -1]]

    y1 = [random.randint(int(ymin), int(ymax)) for i in range(4)]
    y2 = [random.randint(int(ymin), int(ymax)) for i in range(4)]
    y3 = [random.randint(int(ymin), int(ymax)) for i in range(4)]

    Y = [[y1[0], y2[0], y3[0]],
         [y1[1], y2[1], y3[1]],
         [y1[2], y2[2], y3[2]],
         [y1[3], y2[3], y3[3]]]
    print("Матриця планування Y (m=3):")
    for i in range(4):
        print(Y[i])

    x1 = [10, 10, 60, 60]
    x2 = [-25, 10, -25, 10]
    x3 = [10, 15, 15, 10]

    print("-------------------------- Перевірка за критерієм Кохрена --------------------------")
    Y_average = []
    for i in range(len(Y)):
        Y_average.append(np.mean(Y[i], axis=0))
    print("Середні значення відгуку за рядками:", Y_average[0], Y_average[1], Y_average[2], Y_average[3])

    mx1 = np.average(x1)
    mx2 = np.average(x2)
    mx3 = np.average(x3)
    my = np.average(Y_average)

    a1 = (x1[0] * Y_average[0] + x1[1] * Y_average[1] + x1[2] * Y_average[2] + x1[3] * Y_average[3]) / 4
    a2 = (x2[0] * Y_average[0] + x2[1] * Y_average[1] + x2[2] * Y_average[2] + x2[3] * Y_average[3]) / 4
    a3 = (x3[0] * Y_average[0] + x3[1] * Y_average[1] + x3[2] * Y_average[2] + x3[3] * Y_average[3]) / 4

    a11 = (x1[0] * x1[0] + x1[1] * x1[1] + x1[2] * x1[2] + x1[3] * x1[3]) / 4
    a22 = (x2[0] * x2[0] + x2[1] * x2[1] + x2[2] * x2[2] + x2[3] * x2[3]) / 4
    a33 = (x3[0] * x3[0] + x3[1] * x3[1] + x3[2] * x3[2] + x3[3] * x3[3]) / 4

    a12 = (x1[0] * x2[0] + x1[1] * x2[1] + x1[2] * x2[2] + x1[3] * x2[3]) / 4
    a13 = (x1[0] * x3[0] + x1[1] * x3[1] + x1[2] * x3[2] + x1[3] * x3[3]) / 4
    a23 = (x2[0] * x3[0] + x2[1] * x3[1] + x2[2] * x3[2] + x2[3] * x3[3]) / 4

    a32, a31, a21 = a23, a13, a12

    Deter1 = [[1, mx1, mx2, mx3], [mx1, a11, a12, a13], [mx2, a12, a22, a23], [mx3, a13, a23, a33]]
    Deter2 = [my, a1, a2, a3]
    B = [round(i, 4) for i in solve(Deter1, Deter2)]

    ypr1 = B[0] + B[1] * x1[0] + B[2] * x2[0] + B[3] * x3[0]
    ypr2 = B[0] + B[1] * x1[1] + B[2] * x2[1] + B[3] * x3[1]
    ypr3 = B[0] + B[1] * x1[2] + B[2] * x2[2] + B[3] * x3[2]
    ypr4 = B[0] + B[1] * x1[3] + B[2] * x2[3] + B[3] * x3[3]
    print("Отримані практичні значення:", ypr1, ypr2, ypr3, ypr4)

    dispersions = []
    for i in range(len(Y)):
        a = 0
        for k in Y[i]:
            a += (k - np.mean(Y[i], axis=0)) ** 2
        dispersions.append(a / len(Y[i]))
    print("Дисперсії:", dispersions)

    Gp = max(dispersions) / sum(dispersions)

    Gt = 0.7679
    if Gp < Gt:
        print("Дисперсія однорідна")
    else:
        print("Дисперсія неоднорідна")

    print("------------- Перевірка значущості коефіцієнтів за критерієм Стьюдента -------------")
    sb = sum(dispersions) / len(dispersions)
    sbs = (sb / (4 * 3)) ** 0.5

    beta0 = (Y_average[0] * 1 + Y_average[1] * 1 + Y_average[2] * 1 + Y_average[3] * 1) / 4
    beta1 = (Y_average[0] * (-1) + Y_average[1] * (-1) + Y_average[2] * 1 + Y_average[3] * 1) / 4
    beta2 = (Y_average[0] * (-1) + Y_average[1] * 1 + Y_average[2] * (-1) + Y_average[3] * 1) / 4
    beta3 = (Y_average[0] * (-1) + Y_average[1] * 1 + Y_average[2] * 1 + Y_average[3] * (-1)) / 4

    T = [abs(beta0) / sbs, abs(beta1) / sbs, abs(beta2) / sbs, abs(beta3) / sbs]
    print("Значення t:", T[0], T[1], T[2], T[3])

    d = 0
    res = [0] * 4
    Tf = 2.306
    coefs1 = []
    coefs2 = []
    for i in range(4):
        if T[i] <= Tf:
            coefs2.append(B[i])
            res[i] = 0
        else:
            coefs1.append(B[i])
            res[i] = B[i]
            d += 1
    print("Значущі коефіцієнти регресії:", coefs1)
    print("Незначущі коефіцієнти регресії:", coefs2)

    y_st = []
    for i in range(4):
        y_st.append(res[0] + res[1] * x1[i] + res[2] * x2[i] + res[3] * x3[i])
    print("Значення з отриманими коефіцієнтами:", y_st)

    print("-------------------- Перевірка адекватності за критерієм Фішера --------------------")
    Sad = 3 * sum([(y_st[i] - Y_average[i]) ** 2 for i in range(4)]) / (4 - d)
    Fp = Sad / sb
    Ft = 4.5
    print("Fp =", Fp)
    if (Fp < Ft):
        print("Рівняння регресії адекватне при рівні значимості 0.05")
    else:
        print("Рівняння регресії неадекватне при рівні значимості 0.05")
        lab()

lab()