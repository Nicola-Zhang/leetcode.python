
# leetcode submit region begin(Prohibit modification and deletion)
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def kthToLast(self, head, k):
        """
        :type head: ListNode
        :type k: int
        :rtype: int
        """
        if head is None:
            return head

        slow, fast = head, head
        for _ in range(k):
            fast = fast.next

        while fast is not None:
            slow = slow.next
            fast = fast.next

        return slow.val
# leetcode submit region end(Prohibit modification and deletion)
