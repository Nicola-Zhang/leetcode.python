
# leetcode submit region begin(Prohibit modification and deletion)
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next


class Solution(object):
    def mergeInBetween(self, list1, a, b, list2):
        """
        :type list1: ListNode
        :type a: int
        :type b: int
        :type list2: ListNode
        :rtype: ListNode
        """
        # list1 为空
        if list1 is None:
            return list2

        # find a_pre and b_next
        dummy_list1 = ListNode(0, list1)
        a_pre = dummy_list1
        for _ in range(a):
            a_pre = a_pre.next
        b_next = a_pre
        for _ in range(b+2-a):
            b_next = b_next.next

        if list2 is None:
            a_pre.next = b_next
        else:
            # find head and tail nodes of the list2
            head_list2 = list2
            tail_list2 = list2
            while list2 is not None:
                tail_list2 = list2
                list2 = list2.next

            a_pre.next = head_list2
            tail_list2.next = b_next

        return dummy_list1.next

        
# leetcode submit region end(Prohibit modification and deletion)
