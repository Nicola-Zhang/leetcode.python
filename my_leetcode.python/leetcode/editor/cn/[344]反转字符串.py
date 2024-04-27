
# leetcode submit region begin(Prohibit modification and deletion)
class Solution(object):
    def reverseString(self, s):
        """
        :type s: List[str]
        :rtype: None Do not return anything, modify s in-place instead.
        """
        slow, fast = 0, len(s)-1
        while slow < fast:
            tmp = s[slow]
            s[slow] = s[fast]
            s[fast] = tmp

            slow += 1
            fast -= 1

# leetcode submit region end(Prohibit modification and deletion)
