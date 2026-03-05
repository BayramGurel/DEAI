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

    def uniq(self) -> int:
        raise NotImplementedError


@dataclass(frozen=True)
class LinkedListEmpty(LinkedList[T]):
    def is_empty(self) -> bool:
        return True

    def add_first(self, value: T) -> LinkedList[T]:
        return LinkedListPopulated(value, self)

    def to_string(self) -> str:
        return ""

    def uniq(self) -> int:
        return 0


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

    def uniq(self) -> int:
        if self.next.is_empty():
            return 1
        if self.head == self.next.head:
            return self.next.uniq()
        return 1 + self.next.uniq()


def list_from_python_list(values: list[T]) -> LinkedList[T]:
    lst: LinkedList[T] = LinkedListEmpty()
    for v in reversed(values):
        lst = lst.add_first(v)
    return lst


if __name__ == "__main__":
    lst = list_from_python_list([4, 4, 5, 7])
    print(f"Lijst: {lst.to_string()} -> unieke elementen: {lst.uniq()}")