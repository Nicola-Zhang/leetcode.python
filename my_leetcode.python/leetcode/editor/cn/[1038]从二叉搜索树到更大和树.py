
# leetcode submit region begin(Prohibit modification and deletion)
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def __init__(self):
        self.cur_sum = 0

    def bstToGst(self, root):
        """
        :type root: TreeNode
        :rtype: TreeNode
        """
        self.traverse(root)
        return root

    def traverse(self, root):
        if root is None:
            return

        self.traverse(root.right)

        self.cur_sum += root.val
        root.val = self.cur_sum

        self.traverse(root.left)

# leetcode submit region end(Prohibit modification and deletion)
