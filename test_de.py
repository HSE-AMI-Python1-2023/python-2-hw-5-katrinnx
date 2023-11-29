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


def test_initialization():
    assert de.fobj == FOBJ  # проверяем, что указанные параметры сохранены корректно
    assert de.bounds.all() == BOUNDS.all()
    assert de.mutation_coefficient == 0.8  # а также что используются предполагаемые аргументы по умолчанию
    assert de.crossover_coefficient == 0.7
    assert de.population_size == 20
    assert de.dimensions == 2


def test_init_population():
    de._init_population()
    assert np.array(
        [[de.min_bound, de.max_bound]]).all() == BOUNDS.T.all()  # проверяем корректность работы функции инициализации
    assert de.diff.all() >= 0  # разница должна быть неотрицательной
    assert de.fitness.shape == (20,)  # по умолчанию размер популяции 20 => размер массива fitness должен быть (20,)
    assert de.best_idx >= 0  # индекс должен быть неотрицательным


def test_iterate():
    ftns1 = de.fitness
    bst1 = de.best
    de.iterate()
    ftns2 = de.fitness
    bst2 = de.best
    assert ftns1.all() <= ftns2.all()  # проверяем, что после итерации массив fitness стал не хуже
    assert bst1.all() <= bst2.all()  # а также что лучший элемент стал не хуже


def test_mutation():
    de._init_population()
    de._mutation()
    assert de.mutant.shape == (2,)  # проверяем корректность размера мутанта


def test_crossover():
    de._crossover()
    assert de.cross_points.any() == True  # функция кроссовер должна работать так, чтобы хотя бы одно полученное значение было истинно


def test_recombination():
    trial, trial_denorm = de._recombination(0)
    assert trial.shape == (2,)  # проверяем корректность размера объектов, полученных в результате функции recombination
    assert trial_denorm.shape == (2,)
    