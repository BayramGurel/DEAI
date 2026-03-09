# We importeren de Node en de lijst met de merge-methode uit opdracht 2
from opdracht3 import Node, MergeLinkedList


class SortLinkedList(MergeLinkedList):

    # Hulpje: we moeten weten hoe lang de lijst is om het midden te bepalen
    def bepreek_lengte(self):
        teller = 0
        huidig = self.head
        while huidig:
            teller += 1
            huidig = huidig.volgende
        return teller

    def sortMerge(self):
        n = self.bepreek_lengte()

        # 1. BASISGEVAL: Lijsten van 0 of 1 element hoeven niet gesorteerd te worden
        if n <= 1:
            # We gebruiken subList (Opdracht 1) om een veilige kopie te maken
            basis_kopie = self.subList(0, n)
            return SortLinkedList(basis_kopie.head)

        # 2. SPLITSEN: Gebruik subList (Opdracht 1) om de lijst te halveren
        midden = n // 2
        linker_deel = self.subList(0, midden)
        rechter_deel = self.subList(midden, n)

        # We 'upgraden' de geknipte lijstjes naar SortLinkedLists,
        # anders kunnen we de methode sortMerge er niet op aanroepen.
        links = SortLinkedList(linker_deel.head)
        rechts = SortLinkedList(rechter_deel.head)

        # 3. RECURSIE: Sorteer de linker- en rechterhelft afzonderlijk
        gesorteerd_links = links.sortMerge()
        gesorteerd_rechts = rechts.sortMerge()

        # 4. SAMENVOEGEN: Gebruik merge (Opdracht 2) om ze weer in elkaar te ritsen
        # merge() geeft een MergeLinkedList terug, dus we upgraden het eindresultaat weer.
        samengevoegd = gesorteerd_links.merge(gesorteerd_rechts)
        return SortLinkedList(samengevoegd.head)

    def append(self, param):
        nieuwe_node = Node(param)

        # Als de lijst nog leeg is, wordt dit het eerste element (de head)
        if self.head is None:
            self.head = nieuwe_node
            return

        # Loop door de lijst tot we bij het allerlaatste knoopje zijn
        huidig = self.head
        while huidig.volgende is not None:
            huidig = huidig.volgende

        # Plak het nieuwe knoopje aan het einde vast
        huidig.volgende = nieuwe_node


if __name__ == "__main__":
    mijn_lijst = SortLinkedList()

    # Gebruik de nieuwe append methode om de lijst te vullen
    mijn_lijst.append(5)
    mijn_lijst.append(4)
    mijn_lijst.append(7)
    mijn_lijst.append(4)

    print(f"Oorspronkelijke lijst: {mijn_lijst}")

    # Test het sorteren
    # Verwachting: 4 -> 4 -> 5 -> 7 -> None
    gesorteerd = mijn_lijst.sortMerge()
    print(f"Na sortMerge():        {gesorteerd}")