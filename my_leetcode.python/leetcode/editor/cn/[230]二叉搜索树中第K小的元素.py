
# leetcode submit region begin(Prohibit modification and deletion)
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right


class Solution(object):
    def __init__(self):
        self.index = 0
        self.res_val = -1

    def kthSmallest(self, root, k):
        """
        :type root: TreeNode
        :type k: int
        :rtype: int
        """
        self.__find(root, k)
        return self.res_val

    def __find(self, root, k):
        if root is None:
            return

        self.__find(root.left, k)

        self.index += 1
        if self.index == k:
            self.res_val = root.val
            return

        self.__find(root.right, k)






# leetcode submit region end(Prohibit modification and deletion)
