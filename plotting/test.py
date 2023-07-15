def print_passthrough(elem):
    print(elem)
    return elem


with open("../data/test.txt") as fp:
    data = [
        (int(tsc, 16), int(pc, 16))
        for tsc, _, pc, *_ in (line.strip.split() for line in fp)
    ]
    print(data[0])
