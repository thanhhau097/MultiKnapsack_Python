# READ DATA

from __future__ import print_function
import sys
import json
from ortools.sat.python import cp_model
import numpy as np
import time

class MultiKnapsack:
    path = 'data/MinMaxTypeMultiKnapsackInput-1000.json'

    def __init__(self):
        with open(self.path) as f:
            data = json.load(f)

        self.w = [item['w'] for item in data['items']]
        self.p = [item['p'] for item in data['items']]
        self.t = [item['t'] for item in data['items']]
        self.r = [item['r'] for item in data['items']]
        self.D = [item['binIndices'] for item in data['items']]
        self.n = len(self.w)

        self.LW = [pack['minLoad'] for pack in data['bins']]
        self.W = [pack['capacity'] for pack in data['bins']]
        self.P = [pack['p'] for pack in data['bins']]
        self.T = [pack['t'] for pack in data['bins']]
        self.R = [pack['r'] for pack in data['bins']]
        self.m = len(self.W)

        self.nt = int(max(self.t)+1)
        self.nr = int(max(self.r)+1)

        # # ROUND
        # coef = 1000
        # w = [int(coef*t)+1 for t in w]
        # p = [int(coef*t)+1 for t in p]
        # LW = [int(coef*t) for t in LW]
        # W = [int(coef*t) for t in W]
        # P = [int(coef*t) for t in P]

        # ADD NEW BIN
        D = [t+[self.m] for t in self.D]
        self.m += 1
        self.LW.append(0)
        self.W.append(sum(self.w))
        self.P.append(sum(self.p))
        self.T.append(self.nt)
        self.R.append(self.nr)

        # remove LW
        # LW = [0 for t in LW]

        self.w = np.array(self.w)
        self.p = np.array(self.p)
        self.t = np.array(self.t) + 1
        self.r = np.array(self.r) + 1 # t and r must be not equal to 0, for using np multiply
        self.LW = np.array(self.LW)
        self.W = np.array(self.W)
        self.P = np.array(self.P)
        self.T = np.array(self.T)
        self.R = np.array(self.R)

        self.x = np.zeros([self.n, self.m])
        for i in range(self.n):
            self.x[i, self.m - 1] = 1

        # MODEL
        self.W_violations = np.array([0] * self.m)
        self.LW_violations = np.array([0] * self.m)
        self.P_violations = np.array([0] * self.m)
        self.T_violations = np.array([0] * self.m)
        self.R_violations = np.array([0] * self.m)

        #
        self.n_satisfied_items = 0
        self.current_violations = 0

    def get_violations_with(self, violations):
        '''
        Convert negative violations in to real violations
        :param violations: current violations (which can have negative values)
        :return:
        '''
        value = np.array(violations.copy())
        value[value < 0] = 0
        return value

    def update_W_violations(self):
        '''
        Update weight 1 violations

        W_violations will have negative values (easier in calculating getAssignDelta).
        If we need to convert into real violations, we can use get_violations_with(self.W_violations)
        :return:
        '''
        result = self.w.reshape([self.n, 1]).transpose() @ self.x - self.W # tong cua item trong bin tru W cua bin
        # result[result < 0] = 0
        # result = self.get_violations_with(result)
        self.W_violations = np.reshape(result, [-1])

    def update_LW_violations(self):
        '''
        Update lower weight 1 violations

        There are 2 cases:
        1. No item -> No violations
        2. 1+ item: LW_violations_bin = LW_bin - total_weight_of_items_in_bin
        :return:
        '''
        result = self.LW - self.w.reshape([self.n, 1]).transpose() @ self.x
        # result = self.get_violations_with(result)
        n_items_in_bins = np.sum(self.x, axis=0)
        n_items_in_bins[n_items_in_bins > 0] = 1
        self.LW_violations = np.reshape(np.multiply(result, n_items_in_bins), [-1])

    def update_P_violations(self):
        result = self.p.reshape([self.n, 1]).transpose() @ self.x - self.W
        # result = self.get_violations_with(result)
        self.P_violations = np.reshape(result, [-1])

    # t and r must be not equal to 0, for using np multiply
    def update_T_violations(self):
        result = np.multiply(np.transpose([self.t] * self.m), self.x)

        T_current = np.zeros(self.m)
        for i in range(result.shape[1]):
            T_current[i] = np.count_nonzero(np.unique(result[:, i]))

        result = T_current - self.T
        # result = self.get_violations_with(result)
        self.T_violations = result

    def update_R_violations(self):
        result = np.multiply(np.transpose([self.r] * self.m), self.x)
        R_current = np.zeros(self.m)
        for i in range(result.shape[1]):
            R_current[i] = np.count_nonzero(np.unique(result[:, i]))

        result = R_current - self.R
        # result = self.get_violations_with(result)
        self.R_violations = result

    def update_violations(self):
        self.update_W_violations()
        self.update_LW_violations()
        self.update_P_violations()
        self.update_T_violations()
        self.update_R_violations()
        self.current_violations = np.sum(self.get_violations_with(self.W_violations)
                                         + self.get_violations_with(self.LW_violations)
                                         + self.get_violations_with(self.P_violations)
                                         + self.get_violations_with(self.T_violations)
                                         + self.get_violations_with(self.R_violations))

    def get_violations(self):
        return self.current_violations

    def print_violations(self):
        print("W violations:", self.W_violations, "\t total: ", np.sum(self.get_violations_with(self.W_violations)))
        print("LW violations:", self.LW_violations, "\t total: ", np.sum(self.get_violations_with(self.LW_violations)))
        print("P violations:", self.P_violations, "\t total: ", np.sum(self.get_violations_with(self.P_violations)))
        print("T violations:", self.T_violations, "\t total: ", np.sum(self.get_violations_with(self.T_violations)))
        print("R violations:", self.R_violations, "\t total: ", np.sum(self.get_violations_with(self.R_violations)))
        print("Total violations:", self.get_violations())


    def create_x_example(self):
        '''
        Random take each item into random bin.
        :return:
        '''
        self.x = np.zeros([self.n, self.m])

        for i in range(self.n):
            index = np.random.randint(self.m)
            self.x[i, index] = 1

    def get_swap_delta(self, i1, j1, i2, j2):
        return self.get_assign_delta(i1, j1, self.x[i2, j2]) + self.get_assign_delta(i2, j2, self.x[i1, j1])

    def set_swap_propagate(self, i1, j1, i2, j2):
        temp = self.x[i1, j1]
        self.x[i1, j1] = self.x[i2, j2]
        self.x[i2, j2] = temp
        self.update_violations()

    def set_value_propagate(self, i, j, k):
        self.x[i, j] = k
        self.update_violations()

    def get_assign_delta(self, i, j, k):
        # gia su gan x[i][j] = k (k = 0 hoac k = 1)
        # anh huong den cac constraint nhu nao?
        # anh huong W, LW, P, T, R cua bin j
        # phai tinh ca 2 gia tri truoc va sau khi thay doi
        # truoc khi thay doi thi da co san o W_violations, LW_violations, ...
        # old_value = self.x[i, j]
        # new value self.x[i, j] = k

        # anh huong nhu nao den current violation?
        # new_W_bin = self.w.reshape([self.n, 1]).transpose() @ self.x[:, j] # ton thoi gian?
        delta_w = self.w[i] * k - self.w[i] * self.x[i, j]
        temp_W_violations = self.W_violations[j] + delta_w
        delta_W_violations = self.get_violations_with([temp_W_violations])[0] - self.get_violations_with(self.W_violations[j])

        # LW: xet them truong hop neu bin chua chua vat nao thi delta_LW_violations = LW - self.w[i] * k
        if np.sum(self.x[:, j]) == 0:
            delta_LW_violations = self.LW[j] - self.w[i] * k
        else:
            delta_lw = self.w[i] * k - self.w[i] * self.x[i, j]
            temp_LW_violations = self.LW_violations[j] - delta_lw
            delta_LW_violations = self.get_violations_with([temp_LW_violations])[0] - self.get_violations_with(self.LW_violations[j])

        # P
        delta_p = self.p[i] * k - self.p[i] * self.x[i, j]
        temp_P_violations = self.P_violations[j] + delta_p
        delta_P_violations = self.get_violations_with([temp_P_violations])[0] - self.get_violations_with(self.P_violations[j])

        # T
        T_bin = np.multiply(np.transpose(self.t), self.x[:, j])
        T_bin = list(T_bin)

        n_uniques_old = np.count_nonzero(np.unique(T_bin))
        n_uniques = 0
        if self.x[i][j] == 0 and k == 1:
            T_bin.append(self.t[i])
            n_uniques = np.count_nonzero(np.unique(T_bin))
        elif self.x[i][j] == 1 and k == 0:
            T_bin.remove(self.t[i])
            n_uniques = np.count_nonzero(np.unique(T_bin))

        delta_t = n_uniques - n_uniques_old
        temp_T_violations = self.T_violations[j] + delta_t
        delta_T_violations = self.get_violations_with([temp_T_violations])[0] - self.get_violations_with(self.T_violations[j])

        # R
        R_bin = np.multiply(np.transpose(self.r), self.x[:, j])
        R_bin = list(R_bin)

        n_uniques_old = np.count_nonzero(np.unique(R_bin))
        n_uniques = 0
        if self.x[i][j] == 0 and k == 1:
            R_bin.append(self.r[i])
            n_uniques = np.count_nonzero(np.unique(R_bin))
        elif self.x[i][j] == 1 and k == 0:
            R_bin.remove(self.r[i])
            n_uniques = np.count_nonzero(np.unique(R_bin))

        delta_r = n_uniques - n_uniques_old
        temp_R_violations = self.R_violations[j] + delta_r
        delta_R_violations = self.get_violations_with([temp_R_violations])[0] - self.get_violations_with(
            self.R_violations[j])

        return delta_W_violations, delta_LW_violations, delta_P_violations, delta_T_violations, delta_R_violations

    def objective_function(self):
        # TODO
        # numbers of satisfied item and violations
        # current violations mostly focus on LW, we need to decrease the coefficient of LW
        return self.n_satisfied_items, self.get_violations()

    def swap_search(self):
        # TODO
        pass

    # def random_choice_x(self):
    #     for i in range(self.n):
    #         di = self.D[i]
    #         for d in di:
    #             self.x[i, d] = 1


solver = MultiKnapsack()

# start = time.time()
# solver.update_violations()
# solver.print_violations()
# print(time.time() - start)
# print("Before Update")
# solver.update_violations()
# solver.print_violations()

print("After Update")
solver.create_x_example()
solver.update_violations()
solver.print_violations()

print("BEFORE GET ASSIGN DELTA")
print(solver.x)

print("GET ASSIGN DELTA")
random_i = np.random.randint(solver.n)
random_j = np.random.randint(solver.m)
print("Set ", 223, 277, "to", 1)
print(solver.get_assign_delta(223, 277, 1))