import random
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np

PROBABILITY_LIST = [2, 1, 2, 2, 0, 0, 0, 0, 0, 0]

SURVIVAL_RATE = 0.1
MUTATION_RATE = 0.2
CROSSOVER_RATE = 0.7
NUM_OF_GENERATIONS = 1000
POPULATION_SIZE = 100
NUM_OF_JOBS = 10
NUM_OF_OPERATIONS_I = [10, 10, 10, 10, 10, 10, 10, 10, 10, 10]
NUM_OF_MACHINES = 10
LAST_OPERATIONS = []
op_machine = []
op_duration = []

# cuvamo poslednje operacije svakog posla
temp_op = 0
for i in range(NUM_OF_JOBS):
    temp_op += NUM_OF_OPERATIONS_I[i]
    LAST_OPERATIONS.append(temp_op)


# racuna ukupan broj operacija
def get_total_operations():
    total = 0
    for job_op in NUM_OF_OPERATIONS_I:
        total += job_op
    return total


# odredjuje kom poslu pripada operacija
def get_job(operation):
    job = 1
    s = 0
    for job_op in NUM_OF_OPERATIONS_I:
        s += job_op
        if operation <= s:
            return job
        job += 1
    return job


# odredjuje prethodnu operaciju koja se trebala izvrsiti, vraca 0 ako je prva operacija posla
def get_dependent_op(operation):
    job = get_job(operation)
    start_point = 1
    for ops in range(job - 1):
        start_point += NUM_OF_OPERATIONS_I[ops]

    if operation == start_point:
        return 0
    else:
        return operation - 1

#################################################
# operacije su u opsegu od 1 do ukupan broj operacija  [operacija,masina]
def create_individual():
    op_total = get_total_operations()
    id = [i for i in range(op_total)]  # svi moguci id-jevi
    individual = [None] * op_total
    operation = 1
    for j in range(NUM_OF_JOBS):
        ids = random.choices(id, k=NUM_OF_OPERATIONS_I[j])  # biramo j id-jeva i izbacujemo ih iz liste
        id.remove(ids)
        for n in ids:
            individual[n] = [operation, op_machine[operation - 1][random.randrange(0, len(op_machine[operation - 1]))]]
            operation += 1  # redom dodajemo operacije jednog posla na random izabrane id-jeve i cuvamo redoslijed
    return individual


def generate_population(size):
    population = []
    for i in range(size):
        population.append(create_individual())
    return population


def selection(population):
    while True:
        id1 = random.randint(0, len(population) - 1)
        id2 = random.randint(0, len(population) - 1)
        if id1 != id2:
            break

    return population[id1], population[id2]


def crossover(children, p1, p2):
    # biranje crossover pointa i provjera da li zadovoljava uslove
    p1_prim = None
    p2_prim = None

    while True:
        fail = False
        checked = [False for _ in range(NUM_OF_JOBS)]
        point1 = random.randint(int(np.ceil(get_total_operations() / 2)), len(p1[0]) - 2)
        point2 = random.randint(point1, len(p1[0]) - 1)
        sp1 = p1[point1:point2 + 1]
        for op in range(len(sp1[0])-1, -1,  -1):
            if sp1[0] in LAST_OPERATIONS:
                checked[get_job(sp1[0])-1] = True
            else:
                if not checked[get_job(sp1[0]) - 1]:
                    fail = True
                    break
        if not fail:
            p2_prim = p2[:]
            # brisanje operacija u p2 koje sadrzi p1 u dijelu od point1 do point2
            for id1 in range(point1, point2 + 1):
                for id2 in range(len(p2_prim[0])):
                    if p2_prim[0][id2] == p1[0][id1]:
                        p2_prim.pop(id2)
                        break
            p2_prim += sp1
            break
        else:
            continue
    while True:
        fail = False
        checked = [False for _ in range(NUM_OF_JOBS)]
        point1 = random.randint(int(np.ceil(get_total_operations() / 2)), len(p2[0]) - 2)
        point2 = random.randint(point1, len(p2[0]) - 1)
        sp2 = p2[point1:point2 + 1]
        for op in range(len(sp2[0])-1, -1,  -1):
            if sp2[0] in LAST_OPERATIONS:
                checked[get_job(sp2[0])-1] = True
            else:
                if not checked[get_job(sp2[0]) - 1]:
                    fail = True
                    break
        if not fail:
            p1_prim = p1[:]
            # brisanje operacija u p2 koje sadrzi p1 u dijelu od point1 do point2
            for id1 in range(point1, point2 + 1):
                for id2 in range(len(p1_prim[0])):
                    if p1_prim[0][id2] == p2[0][id1]:
                        p1_prim.pop(id2)
                        break
            p1_prim += sp2
            break
        else:
            continue

    children.append(p1_prim)
    children.append(p2_prim)
    return p1_prim, p2_prim
    # for id1 in range(point1, point2 + 1):
    #     p2_prim.append(p1[0][id1])


def fitness(individual):
    pass
def main():
    global op_machine
    global op_duration

    with open("input.txt", "r") as f:
        lines = f.readlines()
        for line in lines:
            new_line = line.strip()
            data = new_line.split("  ")
            # print(data)
            for i in range(0, len(data), 2):
                op_machine.append(int(data[i]))
                op_duration.append(int(data[i + 1]))

    # kreiranje pocetne populacije
    population = []
    for i in range(POPULATION_SIZE):
        print(i)
        individual = create_individual()
        population.append(individual)

    for i in range(NUM_OF_GENERATIONS):
        new_population = []
        while True:
            parent1, parent2 = selection(population)
            crossover(new_population, parent1[:], parent2[:])
            if len(new_population) == POPULATION_SIZE:
                break
        population = new_population

        for i in population:
            print(i)
        exit()


if __name__ == "__main__":
    main()
