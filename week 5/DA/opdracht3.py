# Importeer de klassen uit ons vorige bestand
from opdracht1 import Node, LinkedList


# nieuwe klasse LinkedList erft
class MergeLinkedList(LinkedList):

    def merge(self, andere_lijst):
        """
        Voegt twee gesorteerde lijsten samen tot een nieuwe gesorteerde lijst
        door middel van recursie.
        """

        def fuseer_knoopjes(node1, node2):
            # Als beide lijsten leeg zijn, stop.
            if node1 is None and node2 is None:
                return None

            # Als lijst 1 op is, kopieer dan de rest van lijst 2
            if node1 is None:
                return Node(node2.waarde, fuseer_knoopjes(None, node2.volgende))

            # Als lijst 2 op is, kopieer dan de rest van lijst 1
            if node2 is None:
                return Node(node1.waarde, fuseer_knoopjes(node1.volgende, None))

            # 2. Recursie: Kies de kleinste waarde en maak daar een nieuwe Node van
            if node1.waarde <= node2.waarde:
                # Node 1 is kleiner of gelijk, dus die komt eerst
                return Node(node1.waarde, fuseer_knoopjes(node1.volgende, node2))
            else:
                # Node 2 is kleiner, dus die komt eerst
                return Node(node2.waarde, fuseer_knoopjes(node1, node2.volgende))

        # Start de recursie met de beginpunten (heads) van beide lijsten
        nieuwe_head = fuseer_knoopjes(self.head, andere_lijst.head)

        # Geef een compleet nieuwe lijst terug
        return MergeLinkedList(nieuwe_head)


if __name__ == "__main__":
    # Bouw lijst 1: 4 -> 4 -> 5 -> 7 -> None
    l1_knoop4 = Node(7)
    l1_knoop3 = Node(5, l1_knoop4)
    l1_knoop2 = Node(4, l1_knoop3)
    l1_knoop1 = Node(4, l1_knoop2)
    lijst1 = MergeLinkedList(l1_knoop1)

    # Bouw lijst 2: 2 -> 6 -> 7 -> None
    l2_knoop3 = Node(7)
    l2_knoop2 = Node(6, l2_knoop3)
    l2_knoop1 = Node(2, l2_knoop2)
    lijst2 = MergeLinkedList(l2_knoop1)

    print(f"Lijst 1: {lijst1}")
    print(f"Lijst 2: {lijst2}")

    # Test de merge methode (Dit is het voorbeeld uit je opdracht)
    # Verwachting: 2 -> 4 -> 4 -> 5 -> 6 -> 7 -> 7 -> None
    samengevoegd = lijst1.merge(lijst2)
    print(f"Resultaat na merge: {samengevoegd}")