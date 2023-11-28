import pytest
import numpy as np

from differential_evolution import DifferentialEvolution


# CONSTANTS

def rastrigin(array, A=10):
    return A * 2 + (array[0] ** 2 - A * np.cos(2 * np.pi * array[0])) + (
            array[1] ** 2 - A * np.cos(2 * np.pi * array[1]))


BOUNDS = np.array([[-20, 20], [-20, 20]])
FOBJ = rastrigin


def test_fobj():
    assert FOBJ([1, 2, 3], 1) == 5


"""
Ваша задача добиться 100% покрытия тестами DifferentialEvolution
Различные этапы тестирования логики разделяйте на различные функции
Запуск команды тестирования:
pytest -s test_de.py --cov-report=json --cov
"""

de = DifferentialEvolution(FOBJ, BOUNDS)


def test_initialization():
    assert de.fobj == FOBJ
    # assert np.array(de.bounds) == BOUNDS
    assert de.mutation_coefficient == 0.8
    assert de.crossover_coefficient == 0.7
    assert de.population_size == 20
    assert de.dimensions == 2  # Предполагаем, что у нас есть 3 измерения в примере


def test_init_population():
    de._init_population()
    assert de.diff.all() >= 0

def test_iterate():
    ftns1 = de.fitness
    bst1 = de.best
    de.iterate()
    ftns2 = de.fitness
    bst2 = de.best
    assert ftns1.all() <= ftns2.all()
    assert bst1.all() <= bst2.all()

def test_mutation():
    de._init_population()
    de._mutation()
    assert de.mutant.shape == (2,)

# print(np.array(de.cross_points))
# def test_crossover():
#     de._crossover()
#     assert de.cross_points).any() == True

def test_recombination():
    de._init_population()
    # de._mutation()
    de._crossover()
    trial, trial_denorm = de._recombination(0)
    assert trial.shape == (2,)




# pytest --cov=logging_de.py


# coverage run -m pytest test_de.py
