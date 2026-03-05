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


@dataclass(frozen=True)
class LinkedListEmpty(LinkedList[T]):
    def is_empty(self) -> bool:
        return True

    def add_first(self, value: T) -> LinkedList[T]:
        return LinkedListPopulated(value, self)

    def to_string(self) -> str:
        return ""


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


if __name__ == "__main__":
    empty = LinkedListEmpty()
    print(f"Lege lijst -> '{empty.to_string()}'")

    one = empty.add_first(4)
    print(f"[4] -> '{one.to_string()}'")

    two = one.add_first(7)
    print(f"[7, 4] -> '{two.to_string()}'")