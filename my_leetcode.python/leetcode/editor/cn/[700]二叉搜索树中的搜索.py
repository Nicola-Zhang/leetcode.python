
# leetcode submit region begin(Prohibit modification and deletion)
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def searchBST(self, root, val):
        """
        :type root: TreeNode
        :type val: int
        :rtype: TreeNode
        """
        return self.traverse_BST(root, val)

    def traverse_BST(self, root, val):
        if root is None:
            return None

        if root.val == val:
            res = root
        elif root.val > val:
            res = self.traverse_BST(root.left, val)
        else:
            res = self.traverse_BST(root.right, val)

        return res

    def traverse_all(self, root, val):
        if root is None:
            return None
        if root.val == val:
            return root

        left = self.traverse_all(root.left, val)
        right = self.traverse_all(root.right, val)
        return left if left is not None else right
        
# leetcode submit region end(Prohibit modification and deletion)
