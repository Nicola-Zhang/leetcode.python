
# leetcode submit region begin(Prohibit modification and deletion)
class Solution(object):
    def removeDuplicates(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        if len(nums) == 0:
            return 0

        slow, fast = 0, 1
        while fast < len(nums):
            if nums[slow] == nums[fast]:
                fast += 1
            else:
                slow += 1
                nums[slow] = nums[fast]
                fast += 1

        # slow is the index starting from 0
        return slow+1

# leetcode submit region end(Prohibit modification and deletion)
