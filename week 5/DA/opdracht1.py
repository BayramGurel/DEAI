class Node:
    def __init__(self, waarde, volgende=None):
        self.waarde = waarde
        self.volgende = volgende


class LinkedList:
    def __init__(self, head=None):
        self.head = head

    # huidige_node=False = allereerste aanroep
    # 'None' = einde van de lijst
    def subList(self, start, end, huidige_node=False, index=0):
        # eerste aanroep
        if huidige_node is False:
            huidige_node = self.head

        # 'end' is bereikt
        if huidige_node is None or index == end:
            return LinkedList()  # lege lijst terug

        # Recursie: De methode roept zichzelf aan voor het volgende knoopje
        rest_lijst = self.subList(start, end, huidige_node.volgende, index + 1)

        # 3. OPBOUWEN: Zitten we binnen het start-bereik? Voeg de node dan toe!
        if index >= start:
            # huidige node en plak de rest erachter
            nieuwe_node = Node(huidige_node.waarde, rest_lijst.head)
            rest_lijst.head = nieuwe_node

        return rest_lijst

    # Hulpfunctie om de lijst netjes te printen (bijv: "5 -> 4 -> None")
    def __str__(self):
        elementen = []
        huidig = self.head
        while huidig is not None:
            elementen.append(str(huidig.waarde))
            huidig = huidig.volgende
        elementen.append("None")
        return " -> ".join(elementen)


if __name__ == "__main__":

    # Lijst: 5 -> 4 -> 7 -> 4 -> None
    knoop4 = Node(4) # index 3 - eind
    knoop3 = Node(7, knoop4) # index 2
    knoop2 = Node(4, knoop3) # index 1
    knoop1 = Node(5, knoop2) # index 0

    mijn_lijst = LinkedList(knoop1)
    print(f"Lijst: {mijn_lijst}")

    # Verwachting: 4 -> 7 -> None
    test1 = mijn_lijst.subList(1, 3)
    print(f"subList(1, 3) levert:  {test1}")

    # Verwachting: 5 -> 4 -> None
    test2 = mijn_lijst.subList(0, 2)
    print(f"subList(0, 2) levert:  {test2}")

    # Verwachting: 7 -> 4 -> None
    test3 = mijn_lijst.subList(2, 10)
    print(f"subList(2, 10) levert: {test3}")

    # Verwachting: None
    test4 = mijn_lijst.subList(2, 2)
    print(f"subList(2, 2) levert:  {test4}")