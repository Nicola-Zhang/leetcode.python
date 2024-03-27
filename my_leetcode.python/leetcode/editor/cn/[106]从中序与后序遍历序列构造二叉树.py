
# leetcode submit region begin(Prohibit modification and deletion)
# Definition for a binary tree node.
class TreeNode(object):
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class Solution(object):
    def buildTree(self, inorder, postorder):
        """
        :type inorder: List[int]
        :type postorder: List[int]
        :rtype: TreeNode
        """
        if len(inorder)==0 or len(postorder)==0:
            return None
        assert len(inorder) == len(postorder)

        root_val = postorder[-1]
        # find the left / right inorder list according to the index of the root value
        in_idx = inorder.index(root_val)
        left_inorder = inorder[:in_idx]
        right_inorder = inorder[in_idx+1:]

        # find the left/right postorder list according to the size of the left/right inorder list
        left_size = len(left_inorder)
        right_size = len(right_inorder)
        left_postorder = postorder[:left_size]
        right_postorder = postorder[left_size:left_size+right_size]

        # build root node
        root = TreeNode(root_val)

        left_node = self.buildTree(left_inorder, left_postorder)
        right_node = self.buildTree(right_inorder, right_postorder)

        root.left = left_node
        root.right = right_node

        return root
# leetcode submit region end(Prohibit modification and deletion)
