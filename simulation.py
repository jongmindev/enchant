import simulation
import sys
import numpy

numpy.set_printoptions(linewidth=sys.maxsize)

p = numpy.zeros(20, dtype=float)
r = numpy.zeros(20, dtype=float)

p[0], p[1], p[2] = 100, 100, 100

for i in range(3, 10):
    p[i] = 120 - 10 * i

p[10] = 25

for i in range(11, 20):
    p[i] = 20 - 2 * (i - 11)

p = p/100

r = 1 - p

success_probability = p.copy()
failure_probability = r.copy()


# transition matrix 입력
transition_matrix = numpy.zeros((21, 21), dtype=float)
# 성공확률 : upper off-diagonal
for i in range(len(success_probability)):
    transition_matrix[i][i+1] = success_probability[i]
# 실패확률 : lower off-diagonal
for i in range(1, len(failure_probability)):
    transition_matrix[i][i-1] = failure_probability[i]
# absorbing states
transition_matrix[20][20] = 1

print(transition_matrix)


# transition matrix 유효성 검사
def transition_matrix_validation(_transition_matrix):
    for line in range(len(_transition_matrix)):
        if (sum(_transition_matrix[line]) - 1.0) > 1/10000:
            raise ValueError("Wrong transition matrix")
        # print(f"{i}th row sum : {sum(_transition_matrix[i])}")


transition_matrix_validation(transition_matrix)

# fundamental matrix 입력
sub_matrix_q = numpy.array(transition_matrix[:20, :20])

print("=======sub Q========")
print(sub_matrix_q)

fundamental_matrix = numpy.linalg.inv(numpy.identity(len(sub_matrix_q), dtype=float) - sub_matrix_q)

print("===========fundamental N===============")
numpy.set_printoptions(precision=2)
print(fundamental_matrix)
# print(numpy.identity(len(sub_matrix_q), dtype=float) - sub_matrix_q)

try_for_20 = numpy.zeros(20, dtype=float)

for row in range(len(fundamental_matrix)):
    expectation_to_absorbed = sum(fundamental_matrix[row])
    print(f"start from {row}, expected try is {expectation_to_absorbed}")
    try_for_20[row] = expectation_to_absorbed


try_for_next_step = numpy.zeros(20, dtype=float)
try_validation = numpy.zeros(20, dtype=float)

for start in range(len(try_for_next_step))[::-1]:
    sub_q_n = numpy.array(transition_matrix[:start+1, :start+1])
    fundamental_n = numpy.linalg.inv(numpy.identity(len(sub_q_n), dtype=float) - sub_q_n)
    try_for_next_step[start] = sum(fundamental_n[-1])
    try_validation[start] = sum(try_for_next_step)

print(try_for_20)
print(try_validation)
print(try_for_20 - try_validation)

try_error = numpy.zeros(20, dtype=float)
for i in range(len(try_error)):
    try_error[i] = try_for_20[i] / try_validation[i] - 1

print(try_error)

print(try_for_next_step)

for start in range(len(try_for_next_step)):
    print(f"start from {start}, expected try for the next level is {try_for_next_step[start]}")