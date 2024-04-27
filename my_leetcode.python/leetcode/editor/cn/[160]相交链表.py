
# leetcode submit region begin(Prohibit modification and deletion)
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def getIntersectionNode(self, headA, headB):
        """
        :type head1, head1: ListNode
        :rtype: ListNode
        """
        dummyA, dummyB = headA, headB
        while dummyA is not None or dummyB is not None:
            if dummyA == dummyB:
                return dummyA

            if dummyA is not None:
                dummyA = dummyA.next
            else:
                dummyA = headB

            if dummyB is not None:
                dummyB = dummyB.next
            else:
                dummyB = headA

        return None
# leetcode submit region end(Prohibit modification and deletion)
