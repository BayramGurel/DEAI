from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeVar, Iterator, Tuple
import urllib.request
import urllib.parse
import csv
import io
import json

T = TypeVar("T")

# -----------------------------
# Linked list (sneller sorteren)
# -----------------------------
class LinkedList(Generic[T]):
    def is_empty(self) -> bool:
        raise NotImplementedError

    def add_first(self, value: T) -> LinkedList[T]:
        raise NotImplementedError

    def reverse(self) -> LinkedList[T]:
        raise NotImplementedError

    def sort(self) -> LinkedList[T]:
        """Merge sort: O(n log n) (veel sneller dan sort_simple)."""
        raise NotImplementedError

    def uniq_sorted(self) -> int:
        """Tel unieke waarden in een *gesorteerde* lijst (iteratief, snel)."""
        raise NotImplementedError

    def __iter__(self) -> Iterator[T]:
        cur: LinkedList[T] = self
        while isinstance(cur, LinkedListPopulated):
            yield cur.head
            cur = cur.next


@dataclass(frozen=True)
class LinkedListEmpty(LinkedList[T]):
    def is_empty(self) -> bool:
        return True

    def add_first(self, value: T) -> LinkedList[T]:
        return LinkedListPopulated(value, self)

    def reverse(self) -> LinkedList[T]:
        return self

    def sort(self) -> LinkedList[T]:
        return self

    def uniq_sorted(self) -> int:
        return 0


@dataclass(frozen=True)
class LinkedListPopulated(LinkedList[T]):
    head: T
    next: LinkedList[T]

    def is_empty(self) -> bool:
        return False

    def add_first(self, value: T) -> LinkedList[T]:
        return LinkedListPopulated(value, self)

    def reverse(self) -> LinkedList[T]:
        out: LinkedList[T] = LinkedListEmpty()
        cur: LinkedList[T] = self
        while isinstance(cur, LinkedListPopulated):
            out = out.add_first(cur.head)
            cur = cur.next
        return out

    def _split_alternating(self) -> Tuple[LinkedList[T], LinkedList[T]]:
        """Split in 2 lijsten door om-en-om te verdelen (iteratief)."""
        a: LinkedList[T] = LinkedListEmpty()
        b: LinkedList[T] = LinkedListEmpty()
        cur: LinkedList[T] = self
        toggle = True
        while isinstance(cur, LinkedListPopulated):
            if toggle:
                a = a.add_first(cur.head)
            else:
                b = b.add_first(cur.head)
            toggle = not toggle
            cur = cur.next
        return a.reverse(), b.reverse()

    @staticmethod
    def _merge_sorted(left: LinkedList[T], right: LinkedList[T]) -> LinkedList[T]:
        out: LinkedList[T] = LinkedListEmpty()

        l = left
        r = right
        while isinstance(l, LinkedListPopulated) and isinstance(r, LinkedListPopulated):
            if l.head <= r.head:
                out = out.add_first(l.head)
                l = l.next
            else:
                out = out.add_first(r.head)
                r = r.next

        while isinstance(l, LinkedListPopulated):
            out = out.add_first(l.head)
            l = l.next

        while isinstance(r, LinkedListPopulated):
            out = out.add_first(r.head)
            r = r.next

        return out.reverse()

    def sort(self) -> LinkedList[T]:
        if self.next.is_empty():
            return self
        left, right = self._split_alternating()
        left_s = left.sort()
        right_s = right.sort()
        return LinkedListPopulated._merge_sorted(left_s, right_s)

    def uniq_sorted(self) -> int:
        # iteratief (geen recursie), verwacht gesorteerd
        count = 0
        prev = None
        first = True
        for x in self:
            if first or x != prev:
                count += 1
                prev = x
                first = False
        return count


def list_from_iter(values: Iterator[T]) -> LinkedList[T]:
    # Bouw linked list (omgekeerd opbouwen is het snelst)
    lst: LinkedList[T] = LinkedListEmpty()
    buf = []
    for v in values:
        buf.append(v)
    for v in reversed(buf):
        lst = lst.add_first(v)
    return lst


# -----------------------------
# RDW / Socrata helpers (snel)
# -----------------------------
BASE = "https://opendata.rdw.nl/resource/m9d7-ebf2"

def _open_url(url: str):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    return urllib.request.urlopen(req, timeout=60)

def _build_url(fmt: str, select: str, limit: int | None = None, offset: int | None = None) -> str:
    # encode alleen de value; laat keys met $ intact
    q = [f"$select={urllib.parse.quote(select, safe='(),* ')}"]
    if limit is not None:
        q.append(f"$limit={limit}")
    if offset is not None:
        q.append(f"$offset={offset}")
    return f"{BASE}.{fmt}?" + "&".join(q)

def count_total_rows() -> int:
    # Supersnel: server telt voor je
    url = _build_url("json", "count(*)")
    with _open_url(url) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return int(data[0]["count"])

def iter_kentekens(limit: int) -> Iterator[str]:
    # kleine sample (voor linked list demo)
    url = _build_url("csv", "kenteken", limit=limit, offset=0)
    with _open_url(url) as resp:
        text = io.TextIOWrapper(resp, encoding="utf-8")
        reader = csv.DictReader(text)
        for row in reader:
            k = row.get("kenteken")
            if k:
                yield k.strip()

def count_distinct_kentekens(page_size: int = 50_000) -> int:
    """
    Exact uniek tellen via 'distinct' + pagineren.
    Let op: dit kan nog steeds lang duren omdat er heel veel unieke kentekens zijn.
    """
    total = 0
    offset = 0
    while True:
        url = _build_url("csv", "distinct kenteken", limit=page_size, offset=offset)
        with _open_url(url) as resp:
            text = io.TextIOWrapper(resp, encoding="utf-8")
            reader = csv.DictReader(text)
            n = 0
            for _ in reader:
                n += 1
        total += n
        if n < page_size:
            break
        offset += page_size
    return total


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    # 1) DIRECT SNEL: aantal records (geen download)
    rows = count_total_rows()
    print(f"Aantal rijen in dataset (server-side count): {rows}")

    # 2) Linked list demo op kleine sample (wel snel)
    SAMPLE_N = 5000
    lst = list_from_iter(iter_kentekens(SAMPLE_N))
    sorted_lst = lst.sort()
    print(f"Unieke kentekens in sample van {SAMPLE_N} (linked list merge sort): {sorted_lst.uniq_sorted()}")

    # 3) Exact uniek tellen over alles (kan lang duren) -> zet aan als je dit echt nodig hebt
    DO_EXACT_DISTINCT = False
    if DO_EXACT_DISTINCT:
        uniq_all = count_distinct_kentekens(page_size=50_000)
        print(f"Aantal unieke kentekens (distinct + paging): {uniq_all}")