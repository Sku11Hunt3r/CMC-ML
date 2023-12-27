from typing import List


def sum_non_neg_diag(X: List[List[int]]) -> int:
    if len(X) == 0:
        return -1
    ans = 0
    kol_neg = 0
    for i in range(min(len(X), len(X[0]))):
        if X[i][i] >= 0:
            ans += X[i][i]
        else:
            kol_neg += 1
    if(kol_neg == min(len(X), len(X[0]))):
        return -1
    return ans


def are_multisets_equal(x: List[int], y: List[int]) -> bool:
    x.sort()
    y.sort()
    return x == y


def max_prod_mod_3(x: List[int]) -> int:
    tmp = x
    ans = []
    for i in range(len(x) - 1):
        if x[i]*tmp[i+1] % 3 == 0:
            ans.append(x[i]*tmp[i+1])
    if(len(ans) == 0):
        return -1
    return max(ans)


def convert_image(image: List[List[List[float]]], weights: List[float]) -> List[List[float]]:
    ans = [[0 for j in range(len(image[0]))] for i in range(len(image))]
    for i in range(len(image)):
        for j in range(len(image[0])):
            for k in range(len(weights)):
                ans[i][j] += image[i][j][k]*weights[k]
    return ans


def rle_scalar(x: List[List[int]], y:  List[List[int]]) -> int:
    newx = []
    newy = []
    ans = 0
    for i in x:
        newx += [i[0]] * i[1]
    for i in y:
        newy += [i[0]] * i[1]
    if(len(newx) != len(newy)):
        return -1
    for i in range(len(newx)):
        ans += newx[i] * newy[i]
    return ans


def cosine_distance(X: List[List[float]], Y: List[List[float]]) -> List[List[float]]:
    def norm(X: list[float]) -> float:
        sum = 0
        for x in X:
            sum += x**2
        return sum**(0.5)

    def cos_dist(X: List[float], Y: List[float]) -> float:
        if norm(X) == 0:
            return 1
        if norm(Y) == 0:
            return 1
        sum = 0
        for i in range(len(X)):
            sum += X[i] * Y[i]
        return sum/(norm(X)*norm(Y))
    
    ans = [[0 for j in range(len(Y))] for i in range(len(X))]
    for i in range(len(X)):
        for j in range(len(Y)):
            ans[i][j] = cos_dist(X[i], Y[j])
    return ans