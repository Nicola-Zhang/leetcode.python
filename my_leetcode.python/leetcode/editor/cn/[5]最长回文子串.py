
# leetcode submit region begin(Prohibit modification and deletion)
class Solution(object):
    def longestPalindrome(self, s):
        """
        :type s: str
        :rtype: str
        """
        # 现有算法仍旧有优化空间，
        # 当已找到的回文字符串长度是剩余待遍历字符串长度2倍的时候，
        # 则无需再遍历
        start_idx, end_idx = 0, 0
        for i in range(len(s)):
            slow, fast = self.find_palindrome(s, i, i)
            print("奇数: start:{}, end:{}, str:{}".format(slow, fast, s[slow:fast+1]))
            if fast-slow > end_idx-start_idx:
                start_idx = slow
                end_idx = fast

            if i+1 < len(s) and s[i+1] == s[i]:
                slow, fast = self.find_palindrome(s, i, i+1)
                print("偶数: start:{}, end:{}, str:{}".format(slow, fast, s[slow:fast + 1]))
                if fast - slow > end_idx - start_idx:
                    start_idx = slow
                    end_idx = fast

        return s[start_idx:end_idx+1]

    def find_palindrome(self, s, i, j):
        slow, fast = i, j
        while slow >= 0 and fast <= len(s) - 1:
            if s[slow] != s[fast]:
                break
            slow -= 1
            fast += 1

        return slow+1, fast-1
# leetcode submit region end(Prohibit modification and deletion)
