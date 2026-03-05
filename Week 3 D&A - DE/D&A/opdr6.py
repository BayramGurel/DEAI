from __future__ import annotations
from dataclasses import dataclass
from typing import Generic, TypeVar

T = TypeVar("T")


class LinkedList(Generic[T]):
    def is_empty(self) -> bool:
        raise NotImplementedError

    def add_first(self, value: T) -> LinkedList[T]:
        raise NotImplementedError

    def to_string(self) -> str:
        raise NotImplementedError

    def remove(self, value: T) -> LinkedList[T]:
        raise NotImplementedError

    def smallest(self) -> T:
        raise NotImplementedError

    def sort_simple(self) -> LinkedList[T]:
        raise NotImplementedError


@dataclass(frozen=True)
class LinkedListEmpty(LinkedList[T]):
    def is_empty(self) -> bool:
        return True

    def add_first(self, value: T) -> LinkedList[T]:
        return LinkedListPopulated(value, self)

    def to_string(self) -> str:
        return ""

    def remove(self, value: T) -> LinkedList[T]:
        return self

    def smallest(self) -> T:
        raise ValueError("smallest() called on empty list")

    def sort_simple(self) -> LinkedList[T]:
        return self


@dataclass(frozen=True)
class LinkedListPopulated(LinkedList[T]):
    head: T
    next: LinkedList[T]

    def is_empty(self) -> bool:
        return False

    def add_first(self, value: T) -> LinkedList[T]:
        return LinkedListPopulated(value, self)

    def to_string(self) -> str:
        return f"{self.head} " + self.next.to_string()

    def remove(self, value: T) -> LinkedList[T]:
        if self.head == value:
            return self.next
        return LinkedListPopulated(self.head, self.next.remove(value))

    def smallest(self) -> T:
        if self.next.is_empty():
            return self.head
        rest_smallest = self.next.smallest()
        return self.head if self.head <= rest_smallest else rest_smallest

    def sort_simple(self) -> LinkedList[T]:
        if self.next.is_empty():
            return self
        s = self.smallest()
        rest = self.remove(s)
        sorted_rest = rest.sort_simple()
        return sorted_rest.add_first(s)


def list_from_python_list(values: list[T]) -> LinkedList[T]:
    lst: LinkedList[T] = LinkedListEmpty()
    for v in reversed(values):
        lst = lst.add_first(v)
    return lst


if __name__ == "__main__":
    lst = list_from_python_list([5, 4, 7, 4])
    print(f"Originele lijst: {lst.to_string()}")
    sorted_lst = lst.sort_simple()
    print(f"Gesorteerde lijst: {sorted_lst.to_string()}")