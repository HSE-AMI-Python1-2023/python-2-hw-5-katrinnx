import pytest
import numpy as np

from differential_evolution import DifferentialEvolution


# CONSTANTS

def rastrigin(array, A=10):
    return A * 2 + (array[0] ** 2 - A * np.cos(2 * np.pi * array[0])) + (
            array[1] ** 2 - A * np.cos(2 * np.pi * array[1]))


BOUNDS = np.array([[-20, 20], [-20, 20]])
FOBJ = rastrigin

"""
Ваша задача добиться 100% покрытия тестами DifferentialEvolution
Различные этапы тестирования логики разделяйте на различные функции
Запуск команды тестирования:
pytest -s test_de.py --cov-report=json --cov
"""


def test_fobj():
    assert FOBJ([1, 2, 3], 1) == 5  # проверяем корректную работу функции


de = DifferentialEvolution(FOBJ, BOUNDS)
de._init_population()


def test_initialization():
    assert de.fobj == FOBJ  # проверяем, что указанные параметры сохранены корректно
    assert de.bounds.all() == BOUNDS.all()
    assert de.mutation_coefficient == 0.8  # а также что используются предполагаемые аргументы по умолчанию
    assert de.crossover_coefficient == 0.7
    assert de.population_size == 20
    assert de.dimensions == 2


def test_init_population():
    assert np.array(
        [[de.min_bound,
          de.max_bound]]).all() == BOUNDS.T.all()  # проверяем корректность работы функции инициализации
    assert de.diff.all() >= 0  # разница должна быть неотрицательной
    assert de.fitness.shape == (20,)  # по умолчанию размер популяции 20 => размер массива fitness должен быть (20,)
    assert de.best_idx >= 0  # индекс должен быть неотрицательным


def test_iterate():
    ftns1 = de.fitness

    de.iterate()

    ftns2 = de.fitness

    assert ftns1.all() <= ftns2.all()  # проверяем, что после итерации массив fitness стал не хуже


# рассмотрим 4 случая для проверки if в функции _evaluate: когда значение fitness[0] должно и не должно измениться;
# если оно меняется, то нужно также рассмотреть случаи, когда best меняется и не меняется

def test_evaluate_changing(result_of_evolution=5, population_index=0):
    de.fitness[0] = 8  # задаем параметры так, чтобы мутант был лучше предыдущей особи

    de._evaluate(result_of_evolution, population_index)

    assert de.fitness[0] == 5


def test_evaluate_changing_best(result_of_evolution=5, population_index=0):
    de.best_idx = 1  # задаем параметры так, чтобы мутант был лучше всех
    de.fitness[population_index] = 8
    de.fitness[de.best_idx] = 6
    de.trial = np.array([1, 2])
    de.trial_denorm = np.array([2, 4])

    de._evaluate(result_of_evolution, population_index)

    assert de.best_idx == 0


def test_evaluate_not_changing_best(result_of_evolution=5, population_index=0):  # 1 случай
    de.best_idx = 1  # задаем параметры так, чтобы мутант не был лучше всех (но был лучше предыдущего)
    de.fitness[population_index] = 8
    de.fitness[de.best_idx] = 4
    de.trial = np.array([1, 2])
    de.trial_denorm = np.array([2, 4])

    de._evaluate(result_of_evolution, population_index)

    assert de.best_idx == 1


def test_evaluate_not_changing(result_of_evolution=5, population_index=0):
    de.fitness[0] = 3  # задаем параметры так, чтобы мутант был хуже

    de._evaluate(result_of_evolution, population_index)

    assert de.fitness[0] == 3


def test_mutation():
    de._mutation()

    assert de.mutant.shape == (2,)  # проверяем корректность размера мутанта


# в функции _crossover нужно рассмотреть 2 случая: когда cross_point состоит только из False и не только
def test_crossover_only_false():
    de.crossover_coefficient = 0  # при таком параметре cross_points будет состоять только из False => проверяем условие

    de._crossover()

    assert de.cross_points.any() == True  # функция кроссовер должна работать так, чтобы хотя бы одно полученное значение было истинно


def test_crossover_only_true():
    de.crossover_coefficient = 1  # при таком параметре cross_points будет состоять только из False => проверяем условие

    de._crossover()

    assert de.cross_points.any() == True  # функция кроссовер должна работать так, чтобы хотя бы одно полученное значение было истинно


def test_recombination():
    trial, trial_denorm = de._recombination(0)

    assert trial.shape == (2,)  # проверяем корректность размера объектов, полученных в результате функции recombination
    assert trial_denorm.shape == (2,)
