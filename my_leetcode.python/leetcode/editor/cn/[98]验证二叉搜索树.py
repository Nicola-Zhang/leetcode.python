
# leetcode submit region begin(Prohibit modification and deletion)
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def isValidBST(self, root):
        """
        :type root: TreeNode
        :rtype: bool
        """
        return self.traverse(root, None, None)

    def traverse(self, root, left, right):
        if root is None:
            return True

        # 二叉搜索树需满足的条件为：left_node<root<right_node
        # 其中，left_node / right_node 既可能是子节点也可能是父节点

        if left is not None and root.val <= left.val:
            return False
        if right is not None and root.val >= right.val:
            return False

        # root.left包含有两种情况：
        # 1.root.left.left 该情况下，该节点无左侧节点，但有右侧节点(root.left)
        # 2.root.right.left 该情况下，该节点既有左侧节点（root）,又有右侧节点（root.right）
        left_res = self.traverse(root.left, left, root)

        right_res = self.traverse(root.right, root, right)

        return left_res and right_res



# leetcode submit region end(Prohibit modification and deletion)
