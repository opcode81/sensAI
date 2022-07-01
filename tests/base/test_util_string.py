from sensai.util.string import ToStringMixin


class A(ToStringMixin):
    def __init__(self, p1):
        self.p1 = p1
        self.p2 = 2
        self.p3 = self
        self.p5 = ["foo", self]

    def _toStringExcludes(self):
        return ["p2"]


class B(ToStringMixin):
    def __init__(self, a):
        self.a = a


def test_ToStringMixin_recursion():
    s = str(A("foo"))
    assert s == "A[p1=foo, p3=A[<<], p5=[foo, A[<<]]]"
    s = str(B(A("foo")))
    assert s == "B[a=A[p1=foo, p3=A[<<], p5=[foo, A[<<]]]]"
