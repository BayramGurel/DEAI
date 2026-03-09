import time
import random
import sys

from opdracht6 import SortLinkedList, Node

# Verhoog de recursie-limiet naar bijvoorbeeld 10.000
# Zo weet Python dat onze diepe recursie met opzet is.
sys.setrecursionlimit(10000)

# 1. Maak een grote, willekeurige lijst
N = 2000
test_lijst = SortLinkedList()

# Voeg N willekeurige getallen toe
for _ in range(N):
    test_lijst.append(random.randint(1, 10000))

print(f"Lijst van {N} elementen is aangemaakt. Starten met sorteren...")

# 2. Start de stopwatch
start_tijd = time.perf_counter()

# 3. Voer het sorteeralgoritme uit
gesorteerd = test_lijst.sortMerge()

# 4. Stop de stopwatch
eind_tijd = time.perf_counter()

# Bereken en print de duur
tijdsduur = eind_tijd - start_tijd
print(f"Sorteertijd met sortMerge: {tijdsduur:.5f} seconden")