class Not():
    def __init__(self, pred):
        self.pred = pred

    def pretty_str(self):
        return f'Not({self.pred.pretty_str()})'

    def make_ldips_dict(self):
        temp = {}
        temp['dim'] = [0, 0, 0]
        temp['input'] = self.pred.make_ldips_dict()
        temp['node'] = 'UnOp'
        temp['op'] = 'Not'
        temp['symbolic'] = False
        temp['type'] = 4

        return temp


class And():
    def __init__(self, left=None, right=None):
        self.left = left
        self.right = right

    def pretty_str(self):
        return f'And({self.left.pretty_str()}, {self.right.pretty_str()})'

    def make_ldips_dict(self):
        temp = {}
        temp['dim'] = [0, 0, 0]
        temp['left'] = self.left.make_ldips_dict()
        temp['node'] = 'BinOp'
        temp['op'] = 'And'
        temp['right'] = self.right.make_ldips_dict()
        temp['symbolic'] = False
        temp['type'] = 4

        return temp


class Or():
    def __init__(self, left=None, right=None):
        self.left = left
        self.right = right

    def pretty_str(self):
        return f'Or({self.left.pretty_str()}, {self.right.pretty_str()})'

    def make_ldips_dict(self):
        temp = {}
        temp['dim'] = [0, 0, 0]
        temp['left'] = self.left.make_ldips_dict()
        temp['node'] = 'BinOp'
        temp['op'] = 'Or'
        temp['right'] = self.right.make_ldips_dict()
        temp['symbolic'] = False
        temp['type'] = 4

        return temp


class Value():
    def __init__(self, left, right, op):
        self.left = left
        self.right = right
        self.op = op

    def pretty_str(self):
        return f'{self.op}({self.left}, {self.right})'

    def make_feature_dict(self):
        temp = {}

        temp['dim'] = [0, 0, 0]
        temp['name'] = self.left
        temp['node'] = 'Feature'
        temp['symbolic'] = False
        temp['type'] = 2
        temp['value'] = {
            "dim": [1, 0, 0],
            "name": self.left,
            "node": "Var",
            "symbolic": False,
            "type": 2
        }

        return temp

    def make_param_dict(self):
        temp = {}

        if self.op == 'Gt':
            temp['dim'] = [0, 0, 0]
            temp['name'] = self.right
            temp['node'] = "Param"
            temp['symbolic'] = True
            temp['type'] = 2
            temp['value'] = 'null'
        elif self.op == 'Lt':
            temp['dim'] = [0, 0, 0]
            temp['name'] = self.right
            temp['node'] = "Param"
            temp['symbolic'] = True
            temp['type'] = 2
            temp['value'] = 'null'
        else:
            temp['dim'] = [0, 0, 0]
            temp['node'] = "Num"
            temp['value'] = self.right

        return temp

    def make_ldips_dict(self):
        temp = {}

        temp['dim'] = [0, 0, 0]
        temp['left'] = self.make_feature_dict()
        temp['node'] = 'BinOp'
        temp['op'] = self.op
        temp['right'] = self.make_param_dict()
        temp['symbolic'] = False
        temp['type'] = 4

        return temp


class ConstComp():
    def __init__(self, left, right, op):
        self.left = left
        self.right = right
        self.op = op

    def pretty_str(self):
        return f'{self.op}({self.left}, {self.right})'

    def make_feature_dict(self):
        temp = {}

        temp['dim'] = [0, 0, 0]
        temp['name'] = self.left
        temp['node'] = 'Feature'
        temp['symbolic'] = False
        temp['type'] = 2
        temp['value'] = {
            "dim": [1, 0, 0],
            "name": self.left,
            "node": "Var",
            "symbolic": False,
            "type": 2
        }

        return temp

    def make_const_dict(self):
        temp = {}

        if self.op == 'Gt':
            temp['dim'] = [0, 0, 0]
            temp['name'] = self.right
            temp['node'] = "Param"
            temp['symbolic'] = True
            temp['type'] = 2
            temp['value'] = 'null'
        elif self.op == 'Lt':
            temp['dim'] = [0, 0, 0]
            temp['name'] = self.right
            temp['node'] = "Param"
            temp['symbolic'] = True
            temp['type'] = 2
            temp['value'] = 'null'
        else:
            temp['dim'] = [0, 0, 0]
            temp['node'] = "Num"
            temp['value'] = self.right

        return temp

    def make_ldips_dict(self):
        temp = {}

        temp['dim'] = [0, 0, 0]
        temp['left'] = self.make_feature_dict()
        temp['node'] = 'BinOp'
        temp['op'] = self.op
        temp['right'] = self.make_const_dict()
        temp['symbolic'] = False
        temp['type'] = 4

        return temp
