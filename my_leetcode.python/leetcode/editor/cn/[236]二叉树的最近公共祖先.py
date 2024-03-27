
# leetcode submit region begin(Prohibit modification and deletion)
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None


class Solution(object):
    def lowestCommonAncestor(self, root, p, q):
        """
        :type root: TreeNode
        :type p: TreeNode
        :type q: TreeNode
        :rtype: TreeNode
        """
        return self.__find(root, p.val, q.val)

    def __find(self, root, val1, val2):
        if root is None:
            print("root is None!")
            return None

        if root.val == val1 or root.val == val2:
            print("found val: {}".format(root.val))
            return root

        left = self.__find(root.left, val1, val2)
        right = self.__find(root.right, val1, val2)

        print("left_node: {}, right_node: {}\n".format(
            left.val if left is not None else None,
            right.val if right is not None else None))

        if left is not None and right is not None:
            return root

        return left if left is not None else right
# leetcode submit region end(Prohibit modification and deletion)
