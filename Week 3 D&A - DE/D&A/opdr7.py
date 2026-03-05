from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeVar
import sys

sys.setrecursionlimit(2000)

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

    def length(self) -> int:
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

    def length(self) -> int:
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
        if self.next.is_empty():
            return f"{self.head}"
        return f"{self.head}\n{self.next.to_string()}"

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

    def length(self) -> int:
        return 1 + self.next.length()


def read_plates_file(path: str) -> LinkedList[str]:
    """Lees een tekstbestand (één item per regel) in een linked list."""
    with open(path, "r", encoding="utf8") as f:
        lines = [line.strip() for line in f.read().splitlines() if line.strip()]

    lst: LinkedList[str] = LinkedListEmpty()
    for line in reversed(lines):
        lst = lst.add_first(line)
    return lst


if __name__ == "__main__":
    path_to_file = "kentekens1000.txt"
    try:
        lst = read_plates_file(path_to_file)
    except FileNotFoundError:
        print(f"Het bestand '{path_to_file}' bestaat niet op deze locatie.")
    else:
        print(f"Aantal kentekens (ongefilterd): {lst.length()}")
        sorted_lst = lst.sort_simple()
        print(f"Gesorteerde kentekens:\n{sorted_lst.to_string()}")