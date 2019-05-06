# READ DATA

from __future__ import print_function
import json
import numpy as np
import time
from multiprocessing import Pool


class MultiKnapsack:
    path = 'data/MinMaxTypeMultiKnapsackInput-3000.json'

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

        # ADD NEW BIN
        self.D = [t+[self.m] for t in self.D]
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
        # for i in range(self.n):
        #     self.x[i, self.m - 1] = 1

        # MODEL
        self.W_violations = np.array([0] * self.m)
        self.LW_violations = np.array([0] * self.m)
        self.P_violations = np.array([0] * self.m)
        self.T_violations = np.array([0] * self.m)
        self.R_violations = np.array([0] * self.m)

        # upper bounds
        self.W_upper_bound, self.LW_upper_bound, self.P_upper_bound, self.T_upper_bound, self.R_upper_bound = 1, 1, \
                                                                                                              1, 1, 1

        #
        self.n_satisfied_items = 0
        self.current_violations = 0

    def set_upper_bounds(self, W_ub, LW_ub, P_ub, T_ub, R_ub):
        self.W_upper_bound, self.LW_upper_bound, self.P_upper_bound, self.T_upper_bound, self.R_upper_bound \
            = W_ub, LW_ub, P_ub, T_ub, R_ub


    def get_violations_with(self, violations: np.ndarray, upper_bound=1):
        '''
        Convert negative violations in to real violations
        :param violations: current violations (which can have negative values)
        :return:
        '''
        if type(violations) == np.ndarray:
            value = np.array(violations.copy())
            value[value < 0] = 0
            value[value > 0] = upper_bound # make all constraint are equal
            return value
        else:
            if violations < 0:
                return 0
            elif violations > 0:
                return upper_bound
            return violations

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

        Solution 1:
            There are 2 cases:
            1. No item -> No violations
            2. 1+ item: LW_violations_bin = LW_bin - total_weight_of_items_in_bin

        Solution 2: Calculate all violations
        :return:
        '''
        # solution 1
        result = self.LW - self.w.reshape([self.n, 1]).transpose() @ self.x
        # result = self.get_violations_with(result)
        n_items_in_bins = np.sum(self.x, axis=0)
        n_items_in_bins[n_items_in_bins > 0] = 1
        self.LW_violations = np.reshape(np.multiply(result, n_items_in_bins), [-1])

        # solution 2
        # result = self.LW - self.w.reshape([self.n, 1]).transpose() @ self.x
        # self.LW_violations = np.reshape(result, [-1])

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
        self.current_violations = np.sum(self.get_violations_with(self.W_violations, self.W_upper_bound)
                                         + self.get_violations_with(self.LW_violations, self.LW_upper_bound)
                                         + self.get_violations_with(self.P_violations, self.P_upper_bound)
                                         + self.get_violations_with(self.T_violations, self.T_upper_bound)
                                         + self.get_violations_with(self.R_violations, self.R_upper_bound))

    def get_violations(self):
        return self.current_violations

    def get_all_bin_violations(self):
        return self.get_violations_with(self.W_violations, self.W_upper_bound)\
               + self.get_violations_with(self.LW_violations, self.LW_upper_bound)\
               + self.get_violations_with(self.P_violations, self.P_upper_bound)\
               + self.get_violations_with(self.T_violations, self.T_upper_bound)\
               + self.get_violations_with(self.R_violations, self.R_upper_bound)

    def print_violations(self):
        print("W violations:", self.W_violations, "\t total: ", np.sum(self.get_violations_with(self.W_violations, self.W_upper_bound)))
        print("LW violations:", self.LW_violations, "\t total: ", np.sum(self.get_violations_with(self.LW_violations, self.LW_upper_bound)))
        print("P violations:", self.P_violations, "\t total: ", np.sum(self.get_violations_with(self.P_violations, self.P_upper_bound)))
        print("T violations:", self.T_violations, "\t total: ", np.sum(self.get_violations_with(self.T_violations, self.T_upper_bound)))
        print("R violations:", self.R_violations, "\t total: ", np.sum(self.get_violations_with(self.R_violations, self.R_upper_bound)))
        print("Total violations:", self.get_violations())


    def init_solution(self):
        '''
        Random take each item into random bin.
        :return:
        '''
        self.x = np.zeros([self.n, self.m])

        with open('solution/greedy_result.json') as f:
            line = f.readline()
            line = line.split("[")[1]
            line = line.split("]")[0]
            bins = line.split(" ")
            bins = [int(b) for b in bins]

            # data = json.load(f)
            # bins = list(data["binOfItem"])
            for i, b in enumerate(bins):
                print(i, b)
                self.x[i][b] = 1


        # for i in range(self.n):
        #     di = self.D[i]
        #     random_j = np.random.choice(di)
        #     self.x[i, random_j] = 1

    def get_swap_delta(self, i, j1, j2):
        # TODO
        # phai tinh dong thoi -> sua lai ham get assign delta de chap nhan list
        return self.get_assign_delta([i, i], [j1, j2], [self.x[i, j2], self.x[i, j1]])# + self.get_assign_delta([i], [j2], [self.x[i, j1]])

    def set_swap_propagate(self, i, j1, j2):
        temp = self.x[i, j1]
        self.x[i, j1] = self.x[i, j2]
        self.x[i, j2] = temp
        self.update_violations()

    def set_value_propagate(self, i, j, k):
        self.x[i, j] = k
        self.update_violations()

    def get_assign_delta(self, list_i, list_j, list_k):
        # NOTE: current version **just can be used for swap** or **one variable assignment**
        # gia su gan x[i][j] = k (k = 0 hoac k = 1)
        # anh huong den cac constraint nhu nao?
        # anh huong W, LW, P, T, R cua bin j
        # phai tinh ca 2 gia tri truoc va sau khi thay doi
        # truoc khi thay doi thi da co san o W_violations, LW_violations, ...
        # old_value = self.x[i, j]
        # new value self.x[i, j] = k

        # anh huong nhu nao den current violation?
        total_delta_W_violations, total_delta_LW_violations, total_delta_P_violations, \
        total_delta_T_violations, total_delta_R_violations = 0, 0, 0, 0, 0

        for (i, j, k) in zip(list_i, list_j, list_k):
            # W
            # new_W_bin = self.w.reshape([self.n, 1]).transpose() @ self.x[:, j] # ton thoi gian?
            delta_w = self.w[i] * k - self.w[i] * self.x[i, j]
            temp_W_violations = self.W_violations[j] + delta_w
            delta_W_violations = self.get_violations_with(temp_W_violations, self.W_upper_bound) \
                                 - self.get_violations_with(self.W_violations[j], self.W_upper_bound)

            # LW: xet them truong hop neu bin chua chua vat nao thi delta_LW_violations = LW - self.w[i] * k
            # if np.sum(self.x[:, j]) == 0:
            #     delta_LW_violations = self.get_violations_with(self.LW[j] - self.w[i] * k)
            # else:
            delta_lw = self.w[i] * k - self.w[i] * self.x[i, j]
            temp_LW_violations = self.LW_violations[j] - delta_lw
            delta_LW_violations = self.get_violations_with(temp_LW_violations, self.LW_upper_bound)\
                                  - self.get_violations_with(self.LW_violations[j], self.LW_upper_bound)

            # P
            delta_p = self.p[i] * k - self.p[i] * self.x[i, j]
            temp_P_violations = self.P_violations[j] + delta_p
            delta_P_violations = self.get_violations_with(temp_P_violations, self.P_upper_bound) \
                                 - self.get_violations_with(self.P_violations[j], self.P_upper_bound)

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
            delta_T_violations = self.get_violations_with(temp_T_violations, self.T_upper_bound) \
                                 - self.get_violations_with(self.T_violations[j], self.T_upper_bound)

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
            delta_R_violations = self.get_violations_with(temp_R_violations, self.R_upper_bound) \
                                 - self.get_violations_with(self.R_violations[j], self.R_upper_bound)

            total_delta_W_violations += delta_W_violations
            total_delta_LW_violations += delta_LW_violations
            total_delta_P_violations += delta_P_violations
            total_delta_T_violations += delta_T_violations
            total_delta_R_violations += delta_R_violations

        return total_delta_W_violations, total_delta_LW_violations, total_delta_P_violations, total_delta_T_violations, total_delta_R_violations

    def get_numbers_of_satisfied_items_and_violations(self):
        violations = self.get_all_bin_violations()[:-1]
        items_in_bins = np.sum(self.x, axis=0)[:-1]

        n_satisfied_items = np.sum(items_in_bins[violations == 0])
        return n_satisfied_items, violations

    def objective_function(self):
        # TODO
        # numbers of satisfied item and violations
        # current violations mostly focus on LW, we need to decrease the coefficient of LW
        return self.n_satisfied_items * 5 + self.get_violations()

    def swap_search(self):
        # TODO
        pass

    def search(self):
        # TODO
        # thay doi chien luoc, ban dau de LW lon, sau do dieu chinh LW nho lai = 1
        # can tim item tot de thay doi, tranh viec random
        # cach de dieu chinh
        while True:
            start_time = time.time()
            random_i = np.random.randint(self.n)
            current_j = np.where(self.x[random_i, :] == 1)[0][0]
            di = self.D[random_i].copy()
            # di = np.random.choice(di, min(50, len(di))) # random 50 bins
            # di[-1] = current_j

            start_time_get_swap = time.time()
            arr_delta_violations = []

            # # multiprocess
            # p = Pool(4)
            # p.map(self.append_violations, di)

            for d in di:
                # if d == current_j:
                #     di.remove(d)
                #     continue

                arr_delta_violations.append(np.sum(self.get_swap_delta(random_i, current_j, d)))

            end_time_get_swap = time.time()

            index_min = np.argmin(arr_delta_violations)
            # print(arr_delta_violations)
            # print("Len Di:", len(di), "\tindex min:", index_min)
            # if current_j == di[index_min]:
            #     continue

            start_time_propagate = time.time()
            # if current_j != di[index_min]:
            self.set_swap_propagate(random_i, current_j, di[index_min])
            end_time_propagate = time.time()

            n_items, violations = self.get_numbers_of_satisfied_items_and_violations()
            print("Violations:", np.sum(violations), "\t Items: ", n_items, "\t------>\t Take item ", random_i,
                  "\tfrom bin ", current_j, "\tto bin", di[index_min],
                  "\tget swap time:", "{:2.2}".format(end_time_get_swap - start_time_get_swap),
                  "\t propagate time:", "{:2.2}".format(end_time_propagate - start_time_propagate),
                  "\t in", "{:2.2}".format(time.time() - start_time), "s")


    def append_violations(self, bin):
        pass

    def search2(self):
        # chua check bin cuoi cung, vi violations = 0, can phai kiem tra truong hop nay
        print("Start Searching!")

        # bin cuoi
        bin_index = 1846

        # lay nhung item trong bin day ra
        item_indexes = np.where(self.x[:, bin_index] == 1)[0]
        print("Most violated bin:", bin_index, "\t with violations =", "\t with item: ",
              item_indexes)

        # changed = True
        # while(changed):
        #     changed = False

        for item in item_indexes:
            start_time = time.time()
            di = self.D[item]
            arr_delta_violations = []

            for d in di:
                arr_delta_violations.append(np.sum(self.get_swap_delta(item, bin_index, d)))

            index_min = np.argmin(arr_delta_violations)
            # if di[index_min] != bin_index:
            #     changed= True

            self.set_swap_propagate(item, bin_index, di[index_min])

            n_items, violations = self.get_numbers_of_satisfied_items_and_violations()
            print("Violations:", np.sum(violations), "\t Items: ", n_items, "\t------>\t Take item ", item,
                  "\tfrom bin ", bin_index, "\tto bin", di[index_min], "\t in", time.time() - start_time,
                  "seconds")

        while True:
            # tim bin vi pham nhieu nhat
            bin_violations = self.get_all_bin_violations() # he so bang nhau
            bin_index = np.argmax(bin_violations)

            # lay nhung item trong bin day ra
            item_indexes = np.where(self.x[:, bin_index] == 1)[0]
            print("Most violated bin:", bin_index, "\t with violations =", bin_violations[bin_index], "\t with item: ", item_indexes)


            # changed = True
            # while(changed):
            #     changed = False

            for item in item_indexes:
                start_time = time.time()
                di = self.D[item]
                arr_delta_violations = []

                for d in di:
                    arr_delta_violations.append(np.sum(self.get_swap_delta(item, bin_index, d)))

                index_min = np.argmin(arr_delta_violations)
                # if di[index_min] != bin_index:
                #     changed= True

                self.set_swap_propagate(item, bin_index, di[index_min])

                n_items, violations = self.get_numbers_of_satisfied_items_and_violations()
                print("Violations:", np.sum(violations), "\t Items: ", n_items, "\t------>\t Take item ", item,
                      "\tfrom bin ", bin_index, "\tto bin", di[index_min], "\t in", time.time() - start_time,
                      "seconds")



    def search_with_most_violated_bins(self):
        # vong lap: bin cuoi -> bin phat nhieu nhat -> bin cuoi -> ...
        pass


solver = MultiKnapsack()
solver.init_solution()
solver.set_upper_bounds(5, 20, 5, 5, 5)
solver.update_violations()
solver.search()
