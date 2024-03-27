#!/bin/python
# Longest CommonSubstring

import sys


res = []
def lcs(str1, str2):
    len_str1 = len(str1)
    len_str2 = len(str2)
    dp = [[0] * (len_str2+1) for _ in range(len_str1+1)]
    b = [[0] * (len_str2+1) for _ in range(len_str1+1)]
    for i in range(1, len_str1 + 1):
        for j in range(1, len_str2 + 1):
            if str1[i - 1] == str2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
                b[i][j] = 1
            elif dp[i-1][j] >= dp[i][j - 1]:
                dp[i][j] = dp[i - 1][j]
                b[i][j] = 3
            else:
                dp[i][j] = dp[i][j - 1]
                b[i][j] = 2
    return b


def print_lcs(str1, b, i, j):
    if i == 0 or j == 0:
        return
    if b[i][j] == 1:
        print_lcs(str1, b, i - 1, j - 1)
        res.append(str1[i - 1])
    elif b[i][j] == 3:
        print_lcs(str1, b, i - 1, j)
    else:
        print_lcs(str1, b, i, j - 1)


def main():
    str1 = sys.stdin.readline().strip()
    str2 = sys.stdin.readline().strip()

    b = lcs(str1, str2)

    print_lcs(str1, b, len(str1), len(str2))

    print("Longest CommonSubstring:")
    if len(res) == 0:
        print(0)
    else:
        print(''.join(res))


if __name__ == '__main__':
    main()
