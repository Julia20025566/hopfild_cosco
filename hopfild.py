import numpy as np


class Hopfild:  # класс нейронной сети

    def __init__(self, n, k):  # инициализация
        self.N = n  # количество нейронов
        self.K = k  # максимальное количество эпох распознавания сигнала
        self.W = np.zeros((n, n))  # матрица взаимодействий (весов)

        self.simvols = {"#": 1,
                   "_": -1}
        self.simvols_revers = {1: '#',
                          -1: '_'}

        self.t = "##########" \
                "##########" \
                "____##____" \
                "____##____" \
                "____##____" \
                "____##____" \
                "____##____" \
                "____##____" \
                "____##____" \
                "____##____"

        self.g = "" \
                 "#___##___#" \
                 "##__##__##" \
                 "_##_##_##_" \
                 "__#_##_#__" \
                 "___####___" \
                 "___####___" \
                 "__#_##_#__" \
                 "_##_##_##_" \
                 "##__##__##" \
                 "#___##___#"

        self.m = "##______##" \
                 "###____###"\
                 "####__####" \
                 "##_####_##" \
                 "##__##__##" \
                 "##______##" \
                 "##______##"\
                 "##______##" \
                 "##______##" \
                 "##______##"

        self.p = "##########" \
                "##########" \
                "##______##" \
                "##______##" \
                "##______##" \
                "##########" \
                "##########" \
                "##________" \
                "##________" \
                "##________"

        self.o = "##########" \
                "##########" \
                "##______##" \
                "##______##" \
                "##______##" \
                "##______##" \
                "##______##" \
                "##______##" \
                "##########" \
                "##########"

        self.n = "##______##" \
                "##______##" \
                "##______##" \
                "##______##" \
                "##########" \
                "##########" \
                "##______##" \
                "##______##" \
                "##______##" \
                "##______##"

        self.k = "##______##" \
                "##_____##_" \
                "##____##__" \
                "##__##____" \
                "####______" \
                "####______" \
                "##__##____" \
                "##____##__" \
                "##_____##_" \
                "##______##"

        self.a = "____##____" \
                "____##____" \
                "___####___" \
                "___####___" \
                "__##__##__" \
                "__##__##__" \
                "__######__" \
                "_########_" \
                "_##____##_" \
                "##______##"

        self.y = "##______##" \
                 "_##_____##" \
                 "__##___###" \
                 "___##__##_" \
                 "____##_##_" \
                 "_____###__" \
                 "_____##___" \
                 "____##____" \
                 "___##_____" \
                 "__##______"

        self.list = []
        self.list.append(self.a)
        self.list.append(self.t)
        self.list.append(self.g)
        self.list.append(self.n)
        self.list.append(self.k)
        self.list.append(self.y)

    def remember(self, M):  # метод запоминания образов
        for X in M:  # перебор массива образов
            for i in range(self.N):
                for j in range(self.N):
                    if i == j:
                        self.W[i][j] = 0  # диагональные элементы матрицы полагаются равными нулю
                    else:
                        self.W[i][j] += X[i] * X[j]  # можно опустить деление на N

    def associations(self, signal):  # распознавание образа
        X = signal.copy()  # текущее состояние
        stop = 0

        while (stop < self.K):
            pre_X = X.copy()  # предыдущее состояние
            for i in range(self.N):
                a_i = 0
                for j in range(self.N):
                    a_i += self.W[i][j] * pre_X[j]
                X[i] = self.signum(a_i)

            if (pre_X == X).all():  # выход из цикла, если значения стабилизировались
                return X
            stop += 1
        return X

    def signum(self, a):  # функция активации
        return 1 if a >= 0 else -1

    def create_image(self, base, dic):  # Создает бинарный вектор образа
        a = np.array([])
        for i in base:
            a = np.append(a, dic[i])
        return a

    def parse_image(self, img, dic, n):  # Выводит образ по бинарному вектору
        a = ''
        for i in range(len(img)):
            if i % n == 0:
                print(a)
                a = ''
            a += dic[img[i]]

    def print_images(self, img1, img2, dic, n):
        a1 = ''
        a2 = ''
        c = ''
        d = ''
        for i in range(len(img1)+1):

            if i % n == 0:
                c += a1 + '\n'
                d += a2 + '\n'
                a1 = ''
                a2 = ''
            if i != len(img1):
                a1 += dic[img1[i]]
                a2 += dic[img2[i]]

        c += "\n" + d
        return c

    def compare_result_with_instances(self, vector, instances_list):
        is_equal = False

        for index in range(len(instances_list)):
            instance = instances_list[index]

            if np.array_equal(vector, instance):
                is_equal = True
                break
        return is_equal

    def teach_neurons(self):
        images = np.array([self.create_image(self.t, self.simvols),
                           self.create_image(self.n, self.simvols),
                           self.create_image(self.k, self.simvols),
                           self.create_image(self.a, self.simvols),
                           self.create_image(self.g, self.simvols),
                           self.create_image(self.y, self.simvols),
                           ])
        self.remember(images)

    def get_letter(self, signal):
        res = self.associations(signal)
        k = self.print_images(signal, res, self.simvols_revers, 10)
        bol = self.compare_result_with_instances(k, self.list)
        return k, bol



