
# leetcode submit region begin(Prohibit modification and deletion)
# Definition for a binary tree node.
class TreeNode(object):
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class Solution(object):
    def constructMaximumBinaryTree(self, nums):
        """
        :type nums: List[int]
        :rtype: TreeNode
        """
        if len(nums) == 0:
            return None

        max_val = max(nums)
        max_idx = nums.index(max_val)

        root = TreeNode(max_val)

        left_list = nums[:max_idx]
        right_list = nums[max_idx+1:]
        left_node = self.constructMaximumBinaryTree(left_list)
        right_node = self.constructMaximumBinaryTree(right_list)

        root.left = left_node
        root.right = right_node
        return root
        
# leetcode submit region end(Prohibit modification and deletion)
