from negmas_rl.utils import enumerate_similar


def test_enumerate_similar():
    my_list = [[1, 2, 3, 4], ["a", "b", "c", None], [True, False]]
    paths = enumerate_similar(my_list)
    assert len(paths) == len(list(set(paths))), "No repetitions"
    n = 1
    for x in my_list:
        n *= len(x)
    assert len(paths) == n, f"{n=} but I got {len(paths)} tuples"
    for a, b in zip(paths[:-1], paths[1:]):
        diffs = sum(x != y for x, y in zip(a, b, strict=True))
        assert diffs == 1, f"{a=}, {b=}"
