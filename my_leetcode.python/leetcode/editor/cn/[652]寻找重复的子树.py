
# leetcode submit region begin(Prohibit modification and deletion)
# Definition for a binary tree node.
class TreeNode(object):
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class Solution(object):
    def findDuplicateSubtrees(self, root):
        """
        :type root: TreeNode
        :rtype: List[TreeNode]
        """
        res_nodes = []
        tree_str_dict = {}
        _ = self.traverse(root, res_nodes, tree_str_dict)
        return res_nodes

    def traverse(self, root, res_nodes, tree_str_dict):
        if root is None:
            return "#"

        left_str = self.traverse(root.left, res_nodes, tree_str_dict)
        right_str = self.traverse(root.right, res_nodes, tree_str_dict)

        # build tree str
        tree_str = "{},{},{}".format(root.val, left_str, right_str)

        # deduplicate
        if tree_str not in tree_str_dict:
            tree_str_dict[tree_str] = 1
        else:
            if tree_str_dict[tree_str] == 1:
                res_nodes.append(root)
            tree_str_dict[tree_str] += 1

        return tree_str

# leetcode submit region end(Prohibit modification and deletion)
