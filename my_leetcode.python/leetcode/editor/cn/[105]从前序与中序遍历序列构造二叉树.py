
# leetcode submit region begin(Prohibit modification and deletion)
# Definition for a binary tree node.
class TreeNode(object):
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class Solution(object):
    def buildTree(self, preorder, inorder):
        """
        :type preorder: List[int]
        :type inorder: List[int]
        :rtype: TreeNode
        """
        if len(preorder) == 0 or len(inorder) == 0:
            return None
        assert len(preorder) == len(inorder)

        root_val = preorder[0]
        # root val index in the inorder list
        root_idx = inorder.index(root_val)

        # find the left/right inorder list according to the root index
        l_inorder = [] if root_idx == 0 else inorder[:root_idx]
        r_inorder = [] if root_idx == len(inorder)-1 else inorder[root_idx+1:]

        # find the left/right preorder list according to length of the left/right inorder list
        l_preorder = preorder[1:1+len(l_inorder)]
        r_preorder = preorder[-len(r_inorder):]

        # build root node
        root = TreeNode(root_val)

        # build the left and right node (left and right sub-trees)
        left_node = self.buildTree(l_preorder, l_inorder)
        right_node = self.buildTree(r_preorder, r_inorder)

        root.left = left_node
        root.right = right_node

        return root

        
# leetcode submit region end(Prohibit modification and deletion)
