
# leetcode submit region begin(Prohibit modification and deletion)
# Definition for singly-linked list.
class ListNode(object):
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


class Solution(object):
    def mergeTwoLists(self, list1, list2):
        """
        :type list1: Optional[ListNode]
        :type list2: Optional[ListNode]
        :rtype: Optional[ListNode]
        """
        dummy_node = ListNode()
        head = dummy_node

        p, q = list1, list2
        while p is not None and q is not None:
            if p.val < q.val:
                dummy_node.next = p
                p = p.next
            else:
                dummy_node.next = q
                q = q.next
            dummy_node = dummy_node.next

        if p is not None:
            dummy_node.next = p

        if q is not None:
            dummy_node.next = q

        return head.next

# leetcode submit region end(Prohibit modification and deletion)
