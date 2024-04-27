
# leetcode submit region begin(Prohibit modification and deletion)
# Definition for singly-linked list.
class ListNode(object):
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


class Solution(object):
    def removeNthFromEnd(self, head, n):
        """
        :type head: ListNode
        :type n: int
        :rtype: ListNode
        """
        dummy_node = ListNode(0, head)
        slow, fast = dummy_node, dummy_node
        for _ in range(n+1):
            fast = fast.next

        while fast is not None:
            fast = fast.next
            slow = slow.next

        slow.next = slow.next.next

        return dummy_node.next

# leetcode submit region end(Prohibit modification and deletion)
