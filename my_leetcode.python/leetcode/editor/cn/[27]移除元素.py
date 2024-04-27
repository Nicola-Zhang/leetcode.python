
# leetcode submit region begin(Prohibit modification and deletion)
class Solution(object):
    def removeElement(self, nums, val):
        """
        :type nums: List[int]
        :type val: int
        :rtype: int
        """

        slow, fast = 0, 0
        while fast < len(nums):
            # 先赋值，再 slow+=1
            if nums[fast] != val:
                nums[slow] = nums[fast]
                slow += 1
            fast += 1

        return slow
# leetcode submit region end(Prohibit modification and deletion)
