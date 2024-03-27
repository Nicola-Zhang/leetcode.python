
# leetcode submit region begin(Prohibit modification and deletion)
"""
# Definition for a Node.
class Node:
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None
        self.parent = None
"""

class Solution(object):
    def lowestCommonAncestor(self, p, q):
        """
        :type node: Node
        :rtype: Node
        """
        p_root, q_root = p, q
        while (p_root != q_root):
            if p_root is None:
                p_root = q
            else:
                p_root = p_root.parent

            if q_root is None:
                q_root = p
            else:
                q_root = q_root.parent
        return p_root
        
# leetcode submit region end(Prohibit modification and deletion)
