def ajouter(element, cible=[]):
    cible.append(element)
    return cible

print(ajouter(5))
structure = [1, [2, 3], [], [4, [5, 6]]]
premier, *reste = structure
print(premier)
print(reste)