# jobshop-evo

A solution to the JobShop problem using genetic alghoritm

## Description

Shop scheduling problems belong to the class of multi-stage scheduling problems, where
each job consists of a set of operations.
In such a shop scheduling problem, a set of n jobs J1, J2, . . . , Jn has to be processed on
a set of m machines M1, M2, . . . , Mm. The processing of a job Ji on a particular machine M j
is denoted as an operation and abbreviated by (i, j). Each job Ji consists of a number ni
of operations. For the deterministic scheduling problems, the processing time pi j of each
operation (i, j) is given in advance.

The alghoritm is implemented in python. Parameters for the genetic alghoritm such as number of generations, mutation rate, generation size, etc, can be changed in the source code for different scenarios. 
Heuristic for finding the best solution is simply total time of all jobs executed on a machine.
Time for each generation is shown at the end of a output line.
Example of a input data is found in the input file. 
Results graph is plotted via matplotlib.

## Getting Started

### Dependencies

* python3
* matplotlib
* numpy



### Executing program

* python3 main.py

## Authors

.Danilo Cvijetic
.Teodor Vidakovic
