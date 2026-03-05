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


def list_from_python_list(values: list[T]) -> LinkedList[T]:
    lst: LinkedList[T] = LinkedListEmpty()
    for v in reversed(values):
        lst = lst.add_first(v)
    return lst


if __name__ == "__main__":
    lst = list_from_python_list([5, 4, 7, 4])
    print(f"Originele lijst: {lst.to_string()}")
    lst2 = lst.remove(4)
    print(f"Na remove(4): {lst2.to_string()}")
    lst3 = lst2.remove(4)
    print(f"Nogmaals remove(4): {lst3.to_string()}")