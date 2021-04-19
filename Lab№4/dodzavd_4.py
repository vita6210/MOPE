import random
import sys

import numpy as np
from numpy.linalg import solve
from scipy.stats import f,t
n = 8
m = 3
x1min = -30
x1max = 0
x2min = -15
x2max = 35
x3min = -30
x3max = 35

y_max = 200 + (x1max + x2max + x3max) / 3
y_min = 200 + (x1min + x2min + x3min) / 3

xn = [[1, 1, 1, 1, 1, 1, 1, 1],
      [-1, -1, 1, 1, -1, -1, 1, 1],
      [-1, 1, -1, 1, -1, 1, -1, 1],
      [-1, 1, 1, -1, 1, -1, -1, 1]]

x1x2_norm, x1x3_norm, x2x3_norm, x1x2x3_norm = [0] * n, [0] * n, [0] * n, [0] * n

for i in range(n):
    x1x2_norm[i] = xn[1][i] * xn[2][i]
    x1x3_norm[i] = xn[1][i] * xn[3][i]
    x2x3_norm[i] = xn[2][i] * xn[3][i]
    x1x2x3_norm[i] = xn[1][i] * xn[2][i] * xn[3][i]

y1 = [random.randint(int(y_min), int(y_max)) for i in range(n)]
y2 = [random.randint(int(y_min), int(y_max)) for i in range(n)]
y3 = [random.randint(int(y_min), int(y_max)) for i in range(n)]

# матриця планування
y_matrix = [[y1[0], y2[0], y3[0]],
            [y1[1], y2[1], y3[1]],
            [y1[2], y2[2], y3[2]],
            [y1[3], y2[3], y3[3]],
            [y1[4], y2[4], y3[4]],
            [y1[5], y2[5], y3[5]],
            [y1[6], y2[6], y3[6]],
            [y1[7], y2[7], y3[7]]]

print("Матриця планування y : \n")
for i in range(n):
    print(y_matrix[i])

x0 = [1, 1, 1, 1, 1, 1, 1, 1]

# заміна -1 на х1_мін, 1 на х1_макс
x1 = [-15, -15, 35, 35, -15, -15, 35, 35]

# заміна -1 на х2_мін, 1 на х2_макс
x2 = [-30, 0, -30, 0, -30, 0, -30, 0]

# заміна -1 на х3_мін, 1 на х3_макс
x3 = [-30, 35, 35, -30, 35, -30, -30, 35]

# заповнення нулями х1х2, х1х3, х1х2х3
x1x2, x1x3, x2x3, x1x2x3 = [0] * n, [0] * n, [0] * n, [0] * n
# заповнення х1х2, х1х3, х1х2х3 добутками
for i in range(n):
    x1x2[i] = x1[i] * x2[i]
    x1x3[i] = x1[i] * x3[i]
    x2x3[i] = x2[i] * x3[i]
    x1x2x3[i] = x1[i] * x2[i] * x3[i]
Y_average = []
for i in range(len(y_matrix)):
    Y_average.append(np.mean(y_matrix[i], axis=0))

list_for_b = [xn[0], xn[1], xn[2], xn[3], x1x2_norm, x1x3_norm, x2x3_norm, x1x2x3_norm]
list_for_a = list(zip(x0, x1, x2, x3, x1x2, x1x3, x2x3, x1x2x3))

print("Матриця планування X:")
for i in range(n):
    print(list_for_a[i])
bi = []
for k in range(n):
    S = 0
    for i in range(n):
        S += (list_for_b[k][i] * Y_average[i]) / n
    bi.append(round(S, 3))
# розрахунок аі(система рівнянь) через функцію solve, вивід рівняння регресії
ai = [round(i, m) for i in solve(list_for_a, Y_average)]
print("Рівняння регресії: \n" "y = {} + {}*x1 + {}*x2 + {}*x3 + {}*x1x2 + {}*x1x3 + {}*x2x3 "
      "+ {}*x1x2x3".format(ai[0], ai[1], ai[2], ai[3], ai[4], ai[5], ai[6], ai[7]))
# вивід даних
print("Рівняння регресії для нормованих факторів: \n" "y = {} + {}*x1 + {}*x2 + {}*x3 + {}*x1x2 + {}*x1x3 +"
      " {}*x2x3 + {}*x1x2x3".format(bi[0], bi[1], bi[2], bi[3], bi[4], bi[5], bi[6], bi[7]))

print("Перевірка за критерієм Кохрена")
print("Середні значення відгуку за рядками:", "\n", +Y_average[0], Y_average[1], Y_average[2], Y_average[3],
      Y_average[4], Y_average[5], Y_average[6], Y_average[7])
dispersions = []
for i in range(len(y_matrix)):
    a = 0
    for k in y_matrix[i]:
        a += (k - np.mean(y_matrix[i], axis=0)) ** 2
    dispersions.append(a / len(y_matrix[i]))
Gp = max(dispersions) / sum(dispersions)
Gt = 0.5157
if Gp < Gt:
    print("Дисперсія однорідна")
else:
    print("Дисперсія неоднорідна")
print(" Перевірка значущості коефіцієнтів за критерієм Стьюдента")
sb = sum(dispersions) / len(dispersions)
sbs = (sb / (n * m)) ** 0.5

t_list = [abs(bi[i]) / sbs for i in range(0, n)]

d = 0
res = [0] * n
coef_1 = []
coef_2 = []
F3 = (m - 1) * n
for i in range(n):
    if t_list[i] < t.ppf(q=0.975, df=F3):
        coef_2.append(bi[i])
        res[i] = 0
    else:
        coef_1.append(bi[i])
        res[i] = bi[i]
        d += 1

print("Значущі коефіцієнти регресії:", coef_1)
print("Незначущі коефіцієнти регресії:", coef_2)

if len(coef_1) == 2:
    print("Модель неадекватна")
    sys.exit()

# значення y з коефіцієнтами регресії
y_st = []
for i in range(n):
    y_st.append(res[0] + res[1] * xn[1][i] + res[2] * xn[2][i] + res[3] * xn[3][i] + res[4] * x1x2_norm[i] \
                + res[5] * x1x3_norm[i] + res[6] * x2x3_norm[i] + res[7] * x1x2x3_norm[i])
print("Значення з отриманими коефіцієнтами:\n", y_st)

# критерій Фішера
print("\nПеревірка адекватності за критерієм Фішера\n")
Sad = m * sum([(y_st[i] - Y_average[i]) ** 2 for i in range(n)]) / (n - d)
Fp = Sad / sb
F4 = n - d
# перевірка за допомогою scipy
if Fp < f.ppf(q=0.95, dfn=F4, dfd=F3):
    print("Рівняння регресії адекватне при рівні значимості 0.05")
else:
    print("Рівняння регресії неадекватне при рівні значимості 0.05")
