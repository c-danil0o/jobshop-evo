import random
import numpy as np
from matplotlib import pyplot as plt

SURVIVAL_RATE = 0.08
MUTATION_RATE = 0.7
CROSSOVER_RATE = 0.7
NUM_OF_GENERATIONS = 6
POPULATION_SIZE = 100
NUM_OF_JOBS = 10
NUM_OF_OPERATIONS_I = [10, 10, 10, 10, 10, 10, 10, 10, 10, 10]
NUM_OF_MACHINES = 10
LAST_OPERATIONS = []
op_machine = []  # masine se oznacavaju od 1 pa nadalje
op_duration = []
colors = ["#35ff69", "#44ccff", "#7494ea", "#d138bf", "#3891a6", "#4c5b5c", "#fde74c", "#db5461", "#e8969d",
          "#95f9e3",
          "#49d49d", "#558564", "#ea638c", "#b33c86", "#190e4f", "#03012c", "#002a22"]
# cuvamo poslednje operacije svakog posla
temp_op = 0
for JOB in range(NUM_OF_JOBS):
    temp_op += NUM_OF_OPERATIONS_I[JOB]
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


def get_index_of_operation(individual, operation):
    i = 0
    for op in individual:
        if operation == op[0]:
            return i
        i += 1


#################################################
# operacije su u opsegu od 1 do ukupan broj operacija  [operacija,masina]
def create_individual():
    op_total = get_total_operations()
    id = [i for i in range(op_total)]  # svi moguci id-jevi
    individual = [None] * op_total
    operation = 1
    for j in range(NUM_OF_JOBS):
        ids = random.sample(id, k=NUM_OF_OPERATIONS_I[j])  # biramo j id-jeva i izbacujemo ih iz liste
        ids.sort()
        for n in ids:
            individual[n] = [operation, op_machine[operation - 1][random.randrange(0, len(op_machine[operation - 1]))]]
            id.remove(n)
            operation += 1  # redom dodajemo operacije jednog posla na random izabrane id-jeve i cuvamo redoslijed
    # print(fitness(individual))
    return individual


def generate_population():
    population = []
    for i in range(POPULATION_SIZE):
        population.append(create_individual())
    return population


def selection(population):
    population.sort(key=lambda x: fitness(x) * random.random())

    # while True:
    #     id1 = random.randint(0, len(population) - 1)
    #     id2 = random.randint(0, len(population) - 1)
    #     if id1 != id2:
    #         break

    return population[0], population[1]


# uniform i order crossover
def crossover(children, p1, p2):
    # biranje crossover pointa i provjera da li zadovoljava uslove
    p1_prim = []
    p2_prim = []
    total = get_total_operations()
    while True:
        fail = False
        checked = [False for _ in range(NUM_OF_JOBS)]
        point1 = random.randint(int(np.ceil(get_total_operations() / 2)), len(p1) - 2)
        # point2 = random.randint(point1, len(p1) - 1)
        point2 = random.randint(point1, len(p2) - 1 - (int((1 - CROSSOVER_RATE) * (len(p2) - point1))))
        sp1 = p1[point1:point2 + 1]
        for i in range(len(sp1) - 1, -1, -1):
            if sp1[i][0] in LAST_OPERATIONS:
                checked[get_job(sp1[i][0]) - 1] = True
            else:
                if not checked[get_job(sp1[i][0]) - 1]:
                    fail = True
                    break
        if not fail:
            p2_prim = p2[:]
            # brisanje operacija u p2 koje sadrzi p1 u dijelu od point1 do point2
            for id1 in range(point1, point2 + 1):
                for id2 in range(len(p2_prim)):
                    if p2_prim[id2] == p1[id1]:
                        p2_prim.pop(id2)
                        break
            p2_prim += sp1
            break
        else:
            continue
    while True:
        fail = False
        checked = [False for _ in range(NUM_OF_JOBS)]
        point1 = random.randint(int(np.ceil(get_total_operations() / 2)), len(p2) - 2)
        point2 = random.randint(point1, len(p2) - 1)
        sp2 = p2[point1:point2 + 1]
        for i in range(len(sp2) - 1, -1, -1):
            if sp2[i][0] in LAST_OPERATIONS:
                checked[get_job(sp2[i][0]) - 1] = True
            else:
                if not checked[get_job(sp2[i][0]) - 1]:
                    fail = True
                    break
        if not fail:
            p1_prim = p1[:]
            # brisanje operacija u p2 koje sadrzi p1 u dijelu od point1 do point2
            for id1 in range(point1, point2 + 1):
                for id2 in range(len(p1_prim)):
                    if p1_prim[id2] == p2[id1]:
                        p1_prim.pop(id2)
                        break
            p1_prim += sp2
            break
        else:
            continue

    ###uniform crossover####
    # for op1 in p1_prim:
    #     for op2 in p2_prim:
    #         # ako je ista operacija
    #         # mijenjamo masine
    #         if op1[0] == op2[0]:
    #             tmp = op1[1]
    #             op1[1] = op2[1]
    #             op2[1] = tmp
    #             break

    ####mutacija jedinke#####
    if random.random() <= MUTATION_RATE:
        mutation(p1_prim)
    ## mutation(p1_prim)
    if random.random() <= MUTATION_RATE:
        mutation(p2_prim)
    ## mutation(p2_prim)

    children.append(p1_prim)
    if len(children) == POPULATION_SIZE:
        return p1_prim, p2_prim

    children.append(p2_prim)
    return p1_prim, p2_prim
    # for id1 in range(point1, point2 + 1):
    #     p2_prim.append(p1[0][id1])


# fitness jedinke predstavlja vrijednost promjenljive end_time koja oznacava makespan
def fitness(individual):
    global op_duration

    # vrijeme kada je masina dostupna
    machines_available = [0] * NUM_OF_MACHINES
    # vrijeme kada se naredna operacija posla moze izvrsiti
    jobs_available = [0] * NUM_OF_JOBS

    end_time = 0  # vrijeme zavrsetka operacije koja se poslednja izvrsi-makespan
    for op in individual:
        job = get_job(op[0])
        machine_time = machines_available[op[1] - 1]
        job_time = jobs_available[job - 1]

        start_time = 0
        # vrijeme kada se operacija moze izvrsiti na masini
        if machine_time > job_time:
            start_time = machine_time
        else:
            start_time = job_time

        # vrijeme kada ce masina biti opet dostupna
        machines_available[op[1] - 1] = start_time + op_duration[op[0] - 1]
        # vrijeme kada ce se naredna operacija posla moci izvrsiti
        jobs_available[job - 1] = start_time + op_duration[op[0] - 1]

        # ako je zavrsetak izvrsavanja operacije nakon end_time onda to postaje end_time
        if (start_time + op_duration[op[0] - 1]) > end_time:
            end_time = start_time + op_duration[op[0] - 1]

    return end_time


def mutation(individual):
    # biranje operacije
    while True:
        id = random.randint(0, len(individual) - 1)
        if get_dependent_op(individual[id][0]) != 0:
            po = get_index_of_operation(individual, get_dependent_op(individual[id][0]))
            if (po + 1) != id:
                x = random.randint(po + 1, id)
                break

    # zamjena operacija
    tmp = individual[id]
    individual.pop(id)
    individual.insert(x, tmp)


def generate_graph_data(individual):
    machines = [[] for _ in range(NUM_OF_MACHINES)]
    # vrijeme za svaku masinu
    machines_available = [0] * NUM_OF_MACHINES

    # vrijeme kada se naredna operacija posla moze izvrsiti
    jobs_available = [0] * NUM_OF_JOBS

    for op in individual:
        job = get_job(op[0])
        machine_time = machines_available[op[1] - 1]
        job_time = jobs_available[job - 1]
        start_time = 0

        if machine_time > job_time:
            start_time = machine_time
        else:
            start_time = job_time

        machines[op[1] - 1].append(
            (op[0], start_time, start_time + op_duration[op[0] - 1], op_duration[op[0] - 1]))

        # vrijeme kada ce masina biti opet dostupna
        machines_available[op[1] - 1] = start_time + op_duration[op[0] - 1]
        # vrijeme kada ce se naredna operacija posla moci izvrsiti
        jobs_available[job - 1] = start_time + op_duration[op[0] - 1]

    return machines


def generate_graph(data):
    global colors
    fig, ax = plt.subplots()
    ax.set_ylim(0, NUM_OF_MACHINES * 10 + 15)
    ax.set_xlim(0, 1500)
    ax.set_xlabel("time")
    ax.set_ylabel("machine")
    y_height = 10
    yticks = []
    j = 15
    for i in range(NUM_OF_MACHINES):
        yticks.append(j)
        j += y_height
    ax.set_yticks(yticks)
    ax.set_yticklabels(range(1, NUM_OF_MACHINES + 1))
    mach = 0
    for machine in data:
        mach += 1
        for op in machine:
            ax.broken_barh([(op[1], op[3])], (y_height * mach, y_height - 1), facecolor=colors[get_job(op[0]) - 1])
    plt.show()


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
                op_machine.append([int(data[i]) + 1])
                op_duration.append(int(data[i + 1]))

    # kreiranje pocetne populacije
    population = []
    for _ in range(POPULATION_SIZE):
        individual = create_individual()
        population.append(individual)

    for _ in range(NUM_OF_GENERATIONS):
        new_population = []

        # najbolje jedinke na pocetku liste
        population.sort(key=lambda x: fitness(x))

        # elitizam - najbolje jednike prezivljavaju
        for i in range(int(POPULATION_SIZE * SURVIVAL_RATE)):
            new_population.append(population[i])

        while True:
            parent1, parent2 = selection(population)
            crossover(new_population, parent1[:], parent2[:])
            if len(new_population) == POPULATION_SIZE:
                break
        population = new_population
        print(population[0], fitness(population[0]))

    print(population[0], fitness(population[0]))
    data = generate_graph_data(population[0])
    for mach in data:
        print(mach)
    generate_graph(data)


if __name__ == "__main__":
    main()
    # print(fitness([[1,2],[3,2],[2,3],[4,3]]))
    # 2 3 4 1
