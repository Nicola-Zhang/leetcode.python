
# leetcode submit region begin(Prohibit modification and deletion)
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None


class Solution(object):
    def detectCycle(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        is_cycle = False
        slow, fast = head, head
        while fast is not None and fast.next is not None:
            slow = slow.next
            fast = fast.next.next

            if slow == fast:
                is_cycle = True
                break

        if not is_cycle:
            return None

        slow = head
        while slow != fast:
            slow = slow.next
            fast = fast.next
        return slow


# leetcode submit region end(Prohibit modification and deletion)
