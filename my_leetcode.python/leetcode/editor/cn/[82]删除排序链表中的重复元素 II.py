
# leetcode submit region begin(Prohibit modification and deletion)
# Definition for singly-linked list.
class ListNode(object):
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


class Solution(object):
    def deleteDuplicates(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        if head is None:
            return None
        dummy = ListNode(0, head)
        slow, fast = dummy, head
        while fast is not None:
            if fast.next is not None and fast.val == fast.next.val:
                fast = fast.next
                while fast is not None:
                    if fast.next is not None and fast.val != fast.next.val:
                        fast = fast.next
                        break
                    fast = fast.next

                # 如果fast is None，无法在进入循环中，则需要对该情况单独处理
                if fast is None:
                    slow.next = fast

            else:
                slow.next = fast
                slow = slow.next
                fast = fast.next

        return dummy.next
# leetcode submit region end(Prohibit modification and deletion)
