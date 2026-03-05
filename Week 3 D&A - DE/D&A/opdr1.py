from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, TypeVar

T = TypeVar("T")  # type van de waarden in de lijst


class LinkedList(ABC, Generic[T]):  # abstracte class
    @abstractmethod
    def is_empty(self) -> bool: ...
    @abstractmethod
    def add_first(self, v: T) -> "LinkedList[T]": ...


@dataclass(frozen=True)
class Empty(LinkedList[T]):  # concrete subclass: lege lijst
    def is_empty(self) -> bool: return True
    def add_first(self, v: T) -> LinkedList[T]: return Node(v, self)  # leeg + v => node


@dataclass(frozen=True)
class Node(LinkedList[T]):  # concrete subclass: niet-lege lijst
    value: T                # huidige waarde
    next: LinkedList[T]     # rest van de lijst
    def is_empty(self) -> bool: return False
    def add_first(self, v: T) -> LinkedList[T]: return Node(v, self)  # vooraan toevoegen