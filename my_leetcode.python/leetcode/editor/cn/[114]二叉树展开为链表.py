
# leetcode submit region begin(Prohibit modification and deletion)
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def flatten(self, root):
        """
        :type root: TreeNode
        :rtype: None Do not return anything, modify root in-place instead.
        """
        self.traverse(root)
        return root

    def traverse(self, root):
        if root is None:
            return

        self.traverse(root.left)
        self.traverse(root.right)

        left_node = root.left
        right_node = root.right

        root.left = None
        root.right = left_node

        p = root
        while p.right is not None:
            p = p.right
        p.right = right_node

        return
        
# leetcode submit region end(Prohibit modification and deletion)
