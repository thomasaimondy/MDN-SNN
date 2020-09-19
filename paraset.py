def make_paraset(nps):
    params_M = [[0.060, 0.219, -0.065, 0.003],
                [0.060, 0.219, -0.010, 0.050],
                [0.060, 0.219, -0.135, -0.051]]

    params_T = [[-0.009, 0.246, -0.058, 0.065],
                [-0.009, 0.246, 0.064, 0.091],
                [0.005, 0.158, -0.058, 0.065],
                [0.005, 0.158, 0.064, 0.091],
                [0.007, 0.205, -0.058, 0.065],
                [0.007, 0.205, 0.064, 0.091]]

    if nps == 'M1':
        paraset = [params_M[0]]
    elif nps == 'M2':
        paraset = [params_M[1]]
    elif nps == 'M3':
        paraset = [params_M[2]]
    elif nps == 'T1':
        paraset = [params_T[0]]
    elif nps == 'T2':
        paraset = [params_T[1]]
    elif nps == 'T3':
        paraset = [params_T[2]]
    elif nps == 'T4':
        paraset = [params_T[3]]
    elif nps == 'T5':
        paraset = [params_T[4]]
    elif nps == 'T6':
        paraset = [params_T[5]]

    return paraset