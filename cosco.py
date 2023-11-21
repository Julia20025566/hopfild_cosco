from functools import reduce
import numpy as np
# нейронная сеть коско
# гетероассоциативная память

# каждый вектор ассоциирован с другим вектором
# x - вектор входного образца
# y - натуральные числа в двоичной записи переведенные в биполярную кодировку
# 1 => 001 => [-1, -1, 1]

class Cosco:  # класс нейронной сети

    def __init__(self, n, k):  # инициализация
        self.instances = []
        self.N = n  # количество нейронов
        self.K = k  # максимальное количество эпох распознавания сигнала
        self.W = np.zeros((n, n))  # матрица взаимодействий (весов)

        self.simvols = {"#": 1,
                   "_": -1}
        self.simvols_revers = {1: '#',
                          -1: '_'}

        self.vectors = []

        self.numbers_list = [
            [-1, -1, 1],
            [-1, 1, -1],
            [-1, 1, 1],
            [1, -1, -1],
            [1, -1, 1],
            [1, 1, -1]
        ]

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

        self.g = "#___##___#" \
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
                 "_##_____##"\
                 "__##___###" \
                 "___##__##_" \
                 "____##_##_" \
                 "_____###__" \
                 "_____##___" \
                 "____##____" \
                 "___##_____" \
                 "__##______"

        self.letters = []
        self.letters.append(self.t)
        self.letters.append(self.n)
        self.letters.append(self.k)
        self.letters.append(self.a)
        self.letters.append(self.g)
        self.letters.append(self.y)

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
        for i in range(len(img1) + 1):

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

    # транспонированный вектор y умножаем на вектор x
    def get_matrix_from_instance(self, x, y):
        matrix = []

        for valY in y:
            matrix.append([valX * valY for valX in x])

        return np.array(matrix)

    def get_all_matrices(self, instances_list):
        return [self.get_matrix_from_instance(v['x'], v['y']) for v in instances_list]

    def get_main_matrix(self, instances_list):
        return reduce(lambda m1, m2: m1 + m2, self.get_all_matrices(instances_list))

    # умножаем матрицу на вектор входящего образца
    def multiply_matrix_by_instance(self, matrix, vector):
        a = matrix
        b = np.array([vector]).T
        return a.dot(b)

    # активационная функция
    def activation_func(self, vector):
        return list(map(lambda x: -1 if x[0] < 0 else 1, vector))

    # сравниваем с образцами в памяти
    def compare_result_with_instances(self, vector, instances_list, entry_type):
        is_equal = False
        ind = 0

        for index in range(len(instances_list)):
            instance = instances_list[index]['x' if entry_type == 'y' else 'y']

            if np.array_equal(vector, instance):
                is_equal = True
                ind = index
                print('Успех! ' + str(vector) + ' = #' + str(index + 1))
                break

        return is_equal, vector, ind

    def iterating_to_result(self, matrix, entry_instance, entry_type, counter=1000):
        counter = counter - 1
        vector = self.multiply_matrix_by_instance(matrix, entry_instance)
        activated_vector = self.activation_func(vector)
        self.is_equal, self.vec, self.index = self.compare_result_with_instances(activated_vector, self.instances, entry_type)

        e_type = 'y' if entry_type == 'x' else 'x'
        m = matrix.T

        if counter < 0 and not self.is_equal:
            print('не удалось вычислить образец')
            return
        if not self.is_equal:
            self.iterating_to_result(m, activated_vector, e_type, counter)

    def teach_neurons(self):
        images = np.array([self.create_image(self.t, self.simvols),
                           self.create_image(self.n, self.simvols),
                           self.create_image(self.k, self.simvols),
                           self.create_image(self.a, self.simvols),
                           self.create_image(self.g, self.simvols),
                           self.create_image(self.y, self.simvols),
                           ])
        for index in range(len(images)):
            self.instances.append({'x': images[index], 'y': self.numbers_list[index]})
        self.matrix_W = self.get_main_matrix(self.instances)

    def get_letter(self, signal):
        self.iterating_to_result(self.matrix_W, signal, 'x')

        c = []
        if self.is_equal:
            img = self.create_image(self.letters[self.index], self.simvols)
            c = self.print_images(signal, img, self.simvols_revers, 10)
        print("hi")
        return self.vec, c, self.is_equal



