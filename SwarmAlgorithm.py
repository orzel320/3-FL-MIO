import numpy as np
from FuzzyLogicWithACO import evaluateCustomSystem, applyParametersToSystem, Fuzzy
from typing import List

POPULATION_SIZE = 100
PARAMETERS_NUMBER = 3
TRUE_BOUNDS = [(1.0, 3.0), (1.0, 3.0), (0.0, 10.0)]
SCOPES = [(1.2, 2.8), (1.2, 2.8), (1.0, 9.0)]
BEST_POSITION = np.array([(min_val + max_val) / 2.0 for min_val, max_val in SCOPES])

C_MAX = 1.0
C_MIN = 0.00001
MAX_EPOCHS = 100

# Do zoptymalizowania są 3 reguły, w których występują 3 osobne funkcje, z których każda jest trójkątem, czyli przyjmuje 3 parametry. Daje to w sumie 27 parametrów. Jednak można zoptymalizować procoes, uznając, że nie ma logicznego sensu, aby funkcje skrajne zaczynały się na innych miejscach niż wartości skrajne. Jedyne co może mieć różną pozycję, to środek funkcji środkowej. Należy pamiętać, że sumy funkcji powinny się dodawać do 1, więc kąt nachylenia funkcji musi być dostosowany pod położenie środka funkcji środkowej. Ostatecznie daje to 3 parametry do optymalizacji, czyli pozycję środka funkcji środkowej dla każdej z 3 reguł. Każdy osobnik zawiera 9 zmiennych, bo trzeba uwzględnić jeszcze prędkości oraz najlepsze znalezione dotychczas pozycje osobników.
def initialize_population():
    population = np.random.rand(POPULATION_SIZE, PARAMETERS_NUMBER)
    for i, (min_val, max_val) in enumerate(SCOPES):
        population[:, i] = population[:, i] * (max_val - min_val) + min_val
    return population

def swarm_algorithm(model_to_fit, train_data_X, train_data_Y) -> List[float]:
    global BEST_POSITION
    population = initialize_population()
    
    for epoch in range(MAX_EPOCHS):
        BEST_POSITION = find_new_best_position(population, model_to_fit, train_data_X, train_data_Y)
        
        c = C_MAX - epoch * ((C_MAX - C_MIN) / MAX_EPOCHS)
        population = apply_goa_forces(population, c)
        
        applyParametersToSystem(model_to_fit, change_parameters_from_swarms_to_system(BEST_POSITION))
        
        current_score = np.sum((train_data_Y - evaluateCustomSystem(
            Fuzzy.ControlSystemSimulation(model_to_fit), 
            train_data_X
        ))**2)
        
        print(f"Epoch {epoch:<3} Current best position: {BEST_POSITION}, with score: {current_score:.4f}")
        
    return list(change_parameters_from_swarms_to_system(BEST_POSITION))

def change_parameters_from_swarms_to_system(swarm_params):
    result_vector = np.zeros((27,))
    for i in range(3):
        min_edge = TRUE_BOUNDS[i][0]
        max_edge = TRUE_BOUNDS[i][1]
        center = swarm_params[i]

        result_vector[i * 9]     = min_edge
        result_vector[i * 9 + 1] = min_edge
        result_vector[i * 9 + 2] = center

        result_vector[i * 9 + 3] = min_edge
        result_vector[i * 9 + 4] = center
        result_vector[i * 9 + 5] = max_edge

        result_vector[i * 9 + 6] = center
        result_vector[i * 9 + 7] = max_edge
        result_vector[i * 9 + 8] = max_edge

    return result_vector

def find_new_best_position(population, model_to_fit, train_data_X, train_data_Y):
    results_for_current_population = [
        np.sum((train_data_Y - evaluateCustomSystem(
            Fuzzy.ControlSystemSimulation(
                applyParametersToSystem(
                    model_to_fit,
                    change_parameters_from_swarms_to_system(swarm)
                )
            ),
            train_data_X,
        ))**2)
        for swarm in population
    ] + [
        np.sum((train_data_Y - evaluateCustomSystem(
            Fuzzy.ControlSystemSimulation(
                applyParametersToSystem(
                    model_to_fit,
                    change_parameters_from_swarms_to_system(BEST_POSITION)
                )
            ),
            train_data_X,
        ))**2)
    ]

    best_index = np.argmin(results_for_current_population)
    return population[best_index].copy() if best_index < POPULATION_SIZE else BEST_POSITION.copy()

def apply_goa_forces(population, c):
    new_population = np.zeros_like(population)
    bounds_range = np.array([max_val - min_val for min_val, max_val in SCOPES])
    
    for i in range(POPULATION_SIZE):
        diffs = population - population[i] 
        dists = np.linalg.norm(diffs, axis=1) 
        dists[dists == 0] = 1e-10  
        
        s_r = 0.5 * np.exp(-dists / 1.5) - np.exp(-dists)
        
        force = np.sum(c * (bounds_range / 2.0) * s_r[:, None] * (diffs / dists[:, None]), axis=0)
        new_population[i] = c * force + BEST_POSITION

    for i, (min_val, max_val) in enumerate(SCOPES):
        new_population[:, i] = np.clip(new_population[:, i], min_val, max_val)
        
    return new_population