class Pipeline(object):
    def __init__(self):
        self.data = dict()
        self.components = dict()
        self.provides = dict()
        self.depends = dict()
        self.whoprovides = dict()
        self.data["__data__"] = self.data
        self.data["__pipeline__"] = self

    def add_component(self, name, callable, provides=None, depends=None):
        provides = provides or getattr(callable, "__provides__", [])
        depends = depends or getattr(callable, "__depends__", [])
        for p in provides:
            if p in self.whoprovides:
                raise Exception("There is already a component that provides %s" % p)
        self.provides[name] = provides
        self.depends[name] = depends
        self.components[name] = callable
        for p in provides:
            self.whoprovides[p] = name

    def remove_component(self, name):
        if name not in self.components:
            raise Exception("No component named %s" % name)
        del self.components[name]
        del self.depends[name]
        for p in self.provides[name]:
            del self.whoprovides[p]
            self.invalidate(p)
        del self.provides[name]

    def replace_component(self, name, callable, provides=None, depends=None):
        self.remove_component(name)
        self.add_component(name, callable, provides, depends)

    def invalidate(self, key):
        if key not in self.data:
            return
        del self.data[key]
        for cname in self.components:
            if key in self.depends[cname]:
                for downstream_key in self.provides[cname]:
                    self.invalidate(downstream_key)

    def __setitem__(self, key, value):
        self.data[key] = value

    def __getitem__(self, key):
        self._compute(key)
        return self.data[key]

    def _compute(self, key):
        if key not in self.data:
            cname = self.whoprovides[key]
            for d in self.depends[cname]:
                self._compute(d)
            inputs = [self.data[d] for d in self.depends[cname]]
            results = self.components[cname](*inputs)
            if len(self.provides[cname]) == 1:
                self.data[self.provides[cname][0]] = results
            else:
                for k, v in zip(self.provides[cname], results):
                    self.data[k] = v

