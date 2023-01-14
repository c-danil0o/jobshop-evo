import copy
import random
import numpy as np
from matplotlib import pyplot as plt

# parametri za podatke iz ulaznog fajla

NUM_OF_JOBS = 10
NUM_OF_OPERATIONS_I = [10, 10, 10, 10, 10, 10, 10, 10, 10, 10]
NUM_OF_MACHINES = 10

# parametri za podatke iz dict example
# NUM_OF_JOBS = 5
# NUM_OF_OPERATIONS_I = [3, 4, 3, 3, 3]
# NUM_OF_MACHINES = 5

SURVIVAL_RATE = 0.08
MUTATION_RATE = 0.7
CROSSOVER_RATE = 0.8
NUM_OF_GENERATIONS = 500
POPULATION_SIZE = 100
LAST_OPERATIONS = []
op_machine = []  # masine se oznacavaju od 1 pa nadalje
op_duration = []
NO = 9999999
colors = ["#35ff69", "#44ccff", "#7494ea", "#d138bf", "#3891a6", "#4c5b5c", "#fde74c", "#db5461", "#e8969d",
          "#95f9e3",
          "#49d49d", "#558564", "#ea638c", "#b33c86", "#190e4f", "#03012c", "#002a22"]

# ulazni podaci
# svaki key predstavlja jednu masinu koja u sebi sadrzi rjecnik formata operacija : trajanje
# ukoliko se operacija ne moze izvrsiti na toj masini trajanje postavljamo na konstantu NO
example = {
    1: {
        11: 2,
        12: 5,
        13: 4,
        21: 8,
        22: NO,
        23: NO,
        24: NO,
        31: 3,
        32: NO,
        33: 7,
        41: 3,
        42: NO,
        43: 8,
        51: 2,
        52: 7,
        53: 7
    },
    2: {
        11: 2,
        12: 5,
        13: 4,
        21: 7,
        22: 8,
        23: NO,
        24: NO,
        31: 3,
        32: 6,
        33: NO,
        41: 9,
        42: 5,
        43: 2,
        51: 7,
        52: 4,
        53: NO
    },
    3: {
        11: 2,
        12: 5,
        13: 4,
        21: NO,
        22: 2,
        23: 3,
        24: 4,
        31: 9,
        32: 8,
        33: 6,
        41: 4,
        42: 3,
        43: 5,
        51: NO,
        52: NO,
        53: NO
    },
    4: {
        11: 2,
        12: 5,
        13: 4,
        21: 8,
        22: 7,
        23: 6,
        24: 9,
        31: 4,
        32: 2,
        33: 3,
        41: NO,
        42: NO,
        43: NO,
        51: 5,
        52: 5,
        53: 3
    },
    5: {
        11: 2,
        12: 5,
        13: 4,
        21: 3,
        22: 12,
        23: NO,
        24: 6,
        31: 7,
        32: NO,
        33: 8,
        41: 6,
        42: 7,
        43: NO,
        51: NO,
        52: 2,
        53: 4
    }
}

# cuvamo poslednje operacije svakog posla
temp_op = 0
for JOB in range(NUM_OF_JOBS):
    temp_op += NUM_OF_OPERATIONS_I[JOB]
    LAST_OPERATIONS.append(temp_op)


def read_data_dict():
    global example
    global op_machine
    global op_duration

    for i in range(get_total_operations()):
        op_machine.append([1, 2, 3, 4, 5])
        op_duration.append([])

    for machine in example:
        op_id = 0
        for operation in example[machine]:
            op_duration[op_id].append(example[machine][operation])
            op_id += 1


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
            temp = []
            for tt in op_machine[operation - 1]:
                if op_duration[operation - 1][tt - 1] != NO:
                    temp.append(tt)
            # individual[n] = [operation, op_machine[operation - 1][random.randrange(0, len(op_machine[operation - 1]))]]
            individual[n] = [operation, temp[random.randrange(0, len(temp))]]
            id.remove(n)
            operation += 1  # redom dodajemo operacije jednog posla na random izabrane id-jeve i cuvamo redoslijed
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
        # sp1 = copy.deepcopy(p1[point1:point2 + 1])
        sp1 = p1[point1:point2 + 1]
        for i in range(len(sp1) - 1, -1, -1):
            if sp1[i][0] in LAST_OPERATIONS:
                checked[get_job(sp1[i][0]) - 1] = True
            else:
                if not checked[get_job(sp1[i][0]) - 1]:
                    fail = True
                    break
        if not fail:
            # p2_prim = copy.deepcopy(p2)
            p2_prim = p2[:]
            # brisanje operacija u p2 koje sadrzi p1 u dijelu od point1 do point2
            for id1 in range(point1, point2 + 1):
                for id2 in range(len(p2_prim)):
                    if p2_prim[id2][0] == p1[id1][0]:
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
        # sp2 = copy.deepcopy(p2[point1:point2 + 1])
        sp2 = p2[point1:point2 + 1]
        for i in range(len(sp2) - 1, -1, -1):
            if sp2[i][0] in LAST_OPERATIONS:
                checked[get_job(sp2[i][0]) - 1] = True
            else:
                if not checked[get_job(sp2[i][0]) - 1]:
                    fail = True
                    break
        if not fail:
            # p1_prim = copy.deepcopy(p1)
            p1_prim = p1[:]
            # brisanje operacija u p2 koje sadrzi p1 u dijelu od point1 do point2
            for id1 in range(point1, point2 + 1):
                for id2 in range(len(p1_prim)):
                    if p1_prim[id2][0] == p2[id1][0]:
                        p1_prim.pop(id2)
                        break
            p1_prim += sp2
            break
        else:
            continue

    # uniform crossover####
    for op1 in p1_prim:
        for op2 in p2_prim:
            # ako je ista operacija
            # mijenjamo masine
            if op1[0] == op2[0]:
                tmp = op1[1]
                op1[1] = op2[1]
                op2[1] = tmp
                break

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
        machines_available[op[1] - 1] = start_time + op_duration[op[0] - 1][op[1] - 1]
        # vrijeme kada ce se naredna operacija posla moci izvrsiti
        jobs_available[job - 1] = start_time + op_duration[op[0] - 1][op[1] - 1]

        # ako je zavrsetak izvrsavanja operacije nakon end_time onda to postaje end_time
        if (start_time + op_duration[op[0] - 1][op[1] - 1]) > end_time:
            end_time = start_time + op_duration[op[0] - 1][op[1] - 1]

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
        else:
            x = random.randint(0, id)
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
            (op[0], start_time, start_time + op_duration[op[0] - 1][op[1] - 1], op_duration[op[0] - 1][op[1] - 1]))

        # vrijeme kada ce masina biti opet dostupna
        machines_available[op[1] - 1] = start_time + op_duration[op[0] - 1][op[1] - 1]
        # vrijeme kada ce se naredna operacija posla moci izvrsiti
        jobs_available[job - 1] = start_time + op_duration[op[0] - 1][op[1] - 1]

    return machines


def generate_graph(data, time):
    global colors
    fig, ax = plt.subplots()
    ax.set_ylim(0, NUM_OF_MACHINES * 10 + 15)
    # ax.set_xlim(0, max(data, key = lambda x : x[-1][2]*1.15))
    ax.set_xlim(0, time * 1.15)
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

    plt.savefig("pictue.png")
    plt.show()


def read_input_data():
    global op_machine
    global op_duration

    op_machine = [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10] for _ in range(100)]
    # op_duration = [[] for _ in range(10)]

    with open("input.txt", "r") as f:
        lines = f.readlines()
        for line in lines:
            new_line = line.strip()
            data = new_line.split("  ")
            for i in range(0, len(data), 2):
                temp = [NO for _ in range(10)]
                temp[int(data[i])] = int(data[i + 1])
                op_duration.append(temp)


def make_output_file(data):
    with open("output.txt", "w") as f:
        for mach_id in range(len(data)):
            f.write("Machine " + str(mach_id + 1) + str(data[mach_id]) + "\n")


def main():
    read_input_data()
    # read_data_dict()
    # kreiranje pocetne populacije
    population = []
    for _ in range(POPULATION_SIZE):
        individual = create_individual()
        population.append(individual)

    counter = 0
    for _ in range(NUM_OF_GENERATIONS):
        new_population = []
        # najbolje jedinke na pocetku liste
        population.sort(key=lambda x: fitness(x))
        print("population " + str(counter + 1), fitness(population[0]))
        # elitizam - najbolje jednike prezivljavaju
        for i in range(int(POPULATION_SIZE * SURVIVAL_RATE)):
            new_population.append(population[i])

        while True:
            parent1, parent2 = selection(population)
            crossover(new_population, copy.deepcopy(parent1), copy.deepcopy(parent2))
            # crossover(new_population, parent1, parent2)
            if len(new_population) == POPULATION_SIZE:
                break
        population = new_population
        counter += 1
    population.sort(key=lambda x: fitness(x))
    data = generate_graph_data(population[0])
    for mach in data:
        print(mach)
    make_output_file(data)
    generate_graph(data, fitness(population[0]))


if __name__ == "__main__":
    main()

