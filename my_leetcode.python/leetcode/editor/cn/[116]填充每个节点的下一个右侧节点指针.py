
# leetcode submit region begin(Prohibit modification and deletion)
"""
# Definition for a Node.
class Node(object):
    def __init__(self, val=0, left=None, right=None, next=None):
        self.val = val
        self.left = left
        self.right = right
        self.next = next
"""

class Solution(object):
    def connect(self, root):
        """
        :type root: Node
        :rtype: Node
        """
        if root is None:
            return root

        self.traverse(root.left, root.right)
        return root

    def traverse(self, node1, node2):
        if node1 is None or node2 is None:
            return

        node1.next = node2

        self.traverse(node1.left, node1.right)
        self.traverse(node2.left, node2.right)

        self.traverse(node1.right, node2.left)

# leetcode submit region end(Prohibit modification and deletion)
