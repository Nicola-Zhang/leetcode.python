
# leetcode submit region begin(Prohibit modification and deletion)
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def invertTree(self, root):
        """
        :type root: TreeNode
        :rtype: TreeNode
        """
        self.traverse(root)
        return root

    def traverse(self, root):
        if root is None:
            return

        tmp = root.left
        root.left = root.right
        root.right = tmp

        self.traverse(root.left)
        self.traverse(root.right)

# leetcode submit region end(Prohibit modification and deletion)
