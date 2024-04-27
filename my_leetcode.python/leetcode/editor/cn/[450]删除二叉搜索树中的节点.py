
# leetcode submit region begin(Prohibit modification and deletion)
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def deleteNode(self, root, key):
        """
        :type root: TreeNode
        :type key: int
        :rtype: TreeNode
        """
        if root is None:
            return None

        # step1: 寻找key
        # step2: 删除该节点，同时保持删除后的仍旧是二叉搜索树
        if root.val == key:
            # 1.如果root为空或者其中一个子节点为空
            if root.left is None:
                return root.right
            if root.right is None:
                return root.left

            # 2.右子树最小值，并进行替换
            min_node = self.__get_right_min(root.right)
            root.right = self.deleteNode(root.right, min_node.val)

            min_node.left = root.left
            min_node.right = root.right
            root = min_node

        elif root.val < key:
            root.right = self.deleteNode(root.right, key)

        else:
            root.left = self.deleteNode(root.left, key)

        return root

    def __get_right_min(self, node):
        while node.left is not None:
            node = node.left
        return node
        
# leetcode submit region end(Prohibit modification and deletion)
