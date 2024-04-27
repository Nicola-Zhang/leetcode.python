
# leetcode submit region begin(Prohibit modification and deletion)
class Solution(object):
    def twoSum(self, numbers, target):
        """
        :type numbers: List[int]
        :type target: int
        :rtype: List[int]
        """
        slow, fast = 0, len(numbers)-1
        while slow<fast:
            # print("slow index:{}, val:{}; fast index:{}, val:{}".format(slow, numbers[slow], fast, numbers[fast]))
            two_sum = numbers[slow] + numbers[fast]
            if two_sum == target:
                return [slow+1, fast+1]
            elif two_sum < target:
                slow += 1
            else:
                fast -= 1
        return [-1, -1]

# leetcode submit region end(Prohibit modification and deletion)
