from operator import itemgetter

import numpy as np


def link_same_line(l):
    # '''ctpn框选一行变多个，重新归纳'''
    l = sorted(l, key=lambda l: l[0][1] + l[0][5])
    for i in range(len(l) - 1):
        if l[i][0][0] > 50:
            mid_y = int((l[i][0][1] + l[i][0][5])/2)
            if l[i-1][0][5] > mid_y > l[i-1][0][1]:
                if l[i-1][0][0] < l[i][0][0]:
                    value = l[i-1][1] + l[i][1]
                    l[i-1] = l[i] = [l[i][0], value]
                else:
                    value = l[i][1] + l[i-1][1]
                    l[i-1] = l[i] = [l[i][0], value]
            elif l[i+1][0][5] > mid_y > l[i+1][0][1]:
                if l[i+1][0][0] < l[i][0][1]:
                    value = l[i+1][1]+l[i][1]
                    l[i] = l[i+1] = [l[i+1][0], value]
                else:
                    value = l[i][1] + l[i + 1][1]
                    l[i] = l[i + 1] = [l[i + 1][0], value]
    if l[-1][0][0] > 50:
        value = l[-2][1] + l[-1][1]
        l[-2] = l[-1] = [l[-2][0], value]
    new_l = []
    for t in range(len(l)-1):
        if np.all(l[t][0] == l[t+1][0]):
            pass
        else:
            new_l.append(l[t])
    new_l.append(l[-1])
    return new_l


def get_big_text(v, ft, template):
    with open('template_images/{}/{}/field.txt'.format(ft, template), 'r', encoding='utf8') as f:
        con = f.read()

    # k_list = ['名称', '住所', '法定代表人', '注册资本', '实收资本', '公司类型', '经营范围']
    k_list = con.split(' ')
    k_list.reverse()
    mm = []
    y1_y2 = []
    distences = []
    for i in range(1, len(v)):
        distences.append(v[i][0][1] - v[i - 1][0][5])
    print('distence:', distences)
    print('sum:', sum(distences))
    y1_y2.append([v[0][0][1], v[0][0][5]])
    for i in range(1, len(v)):
        y1_y2.append([v[i][0][1], v[i][0][5]])
        dis = v[i][0][1] - v[i-1][0][5]
        if dis < 10:
            v[i][0][1] = v[i-1][0][1]
            v[i][1] = v[i-1][1] + v[i][1]
            if i == len(v)-1 and len(k_list) > 0:
                mm.append([k_list.pop(), v[i][1]])
        if dis > 10 and len(k_list) > 0:
            mm.append([k_list.pop(), v[i-1][1]])
            if i == len(v)-1 and len(k_list) > 0:
                mm.append([k_list.pop(), v[i][1]])
    return mm, y1_y2


def get_tt(l):
    for i in range(len(l)):
        if l[i][0][0] > 50:
            sum_y = int((l[i][0][1] + l[i][0][5]))
            print(sum_y)
            dis_list = []
            sub_list = []
            for j in range(len(l)):
                if i != j:
                    dis_list.append(abs(sum_y -(l[j][0][1] + l[j][0][5])))
                    sub_list.append(j)
            print('123456')
            sub = min(dis_list)
            print('654321', sub, sub_list)
            sub = sub_list[sub]
            print('22222', sub)
            if l[i][0][0] > l[sub][0][0]:
                print('subsubsub', sub)
                value = l[sub][1]+l[i][1]
                l[i] = l[sub] = [l[sub][0], value]
            else:
                print('11111', i)
                value = l[i][1] + l[sub][1]
                l[i] = l[sub] = [l[i][0], value]
    new_l = []
    for t in range(len(l) - 1):
        if np.all(l[t][0] == l[t + 1][0]):
            pass
        else:
            new_l.append(l[t])
    new_l.append(l[-1])
    print('ggggggggg', new_l)
    return new_l