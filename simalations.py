from heatingProject import HeatingModel, parameteresFunc, drawUsage, drawHeatingModel, drawHeatingModel2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

m1 = parameteresFunc(2, 2, 2, 2, 298, 294, 285, 296, 297, 295)
m2 = parameteresFunc(0, 5, 5, 5, 298, 294, 285, 296, 297, 295)
m3 = parameteresFunc(1, 4, 2, 1, 298, 294, 285, 296, 297, 295)
m4 = parameteresFunc(3, 4, 3, 3, 298, 294, 285, 296, 297, 295)

m12 = parameteresFunc(2, 2, 2, 2, 290, 297, 285, 291, 292, 290)
m22 = parameteresFunc(0, 5, 5, 5, 290, 297, 285, 291, 292, 290)
m32 = parameteresFunc(1, 4, 2, 1, 290, 297, 285, 291, 292, 290)
m42 = parameteresFunc(3, 4, 3, 3, 290, 297, 285, 291, 292, 290)

m13 = parameteresFunc(2, 2, 2, 2, 285, 283, 280, 300, 299, 298)
m23 = parameteresFunc(0, 5, 5, 5, 285, 283, 280, 300, 299, 298)
m33 = parameteresFunc(1, 4, 2, 1, 285, 283, 280, 300, 299, 298)
m43 = parameteresFunc(3, 4, 3, 3, 285, 283, 280, 300, 299, 298)

m14 = parameteresFunc(2, 2, 2, 2, 295, 287, 285, 295, 295, 280)
m24 = parameteresFunc(0, 5, 5, 5, 295, 287, 285, 295, 295, 280)
m34 = parameteresFunc(1, 4, 2, 1, 295, 287, 285, 295, 295, 280)
m44 = parameteresFunc(3, 4, 3, 3, 295, 287, 285, 295, 295, 280)

#model1 = HeatingModel(m1)
#drawHeatingModel2(m1)
#drawUsage(m1, m2, m3, m4, [298, 294, 285, 296, 297, 295])
#drawUsage(m12, m22, m32, m42, [290, 297, 285, 291, 292, 290])
drawUsage(m13, m23, m33, m43, [285, 283, 280, 300, 299, 298])
#drawUsage(m14, m24, m34, m44, [295, 287, 285, 295, 295, 280])
plt.savefig("zuzycie3.png")
plt.show()