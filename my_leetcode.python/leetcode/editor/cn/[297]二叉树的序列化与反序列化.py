
# leetcode submit region begin(Prohibit modification and deletion)
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None


class Codec:

    def serialize(self, root):
        """Encodes a tree to a single string.
        
        :type root: TreeNode
        :rtype: str
        """
        if root is None:
            return ""
        # traverse the tree
        res_list = []
        self.__traverse_tree(root, res_list)
        # convert list to str
        res_str = ",".join(res_list)
        print("ser str: {}".format(res_str))
        return res_str

    def __traverse_tree(self, root, res_list):
        if root is None:
            res_list.append("#")
            return

        res_list.append(str(root.val))
        self.__traverse_tree(root.left, res_list)
        self.__traverse_tree(root.right, res_list)
        return

    def deserialize(self, data):
        """Decodes your encoded data to tree.
        
        :type data: str
        :rtype: TreeNode
        """
        if len(data) == 0:
            val_list = []
        else:
            val_list = data.split(",")

        print("deser list: {}".format(val_list))
        root = self.__traverse_list(val_list)
        return root

    def __traverse_list(self, val_list):
        if len(val_list) == 0:
            return None

        first_val = val_list[0]
        del val_list[0]
        if first_val == "#":
            return None

        root = TreeNode(int(first_val))
        root.left = self.__traverse_list(val_list)
        root.right = self.__traverse_list(val_list)

        return root


# Your Codec object will be instantiated and called as such:
# ser = Codec()
# deser = Codec()
# ans = deser.deserialize(ser.serialize(root))
# leetcode submit region end(Prohibit modification and deletion)
