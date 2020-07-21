
class LazyWrapper(object):
    def __init__(self, initializer):
        self.initializer = initializer
        self.obj = None
        self.initialized = False

    def get(self):
        if not self.initialized:
            self.obj = self.initializer()
            self.initialized = True

        return self.obj


class LazyLoader(object):
    def __init__(self, initializer):
        self.wrapper = LazyWrapper(initializer)

    def __getattribute__(self, name):
        obj = object.__getattribute__(self, "wrapper").get()

        attr = object.__getattribute__(obj, name)
        if hasattr(attr, '__call__'):
            def newfunc(*args, **kwargs):
                result = attr(*args, **kwargs)
                return result

            return newfunc
        else:
            return attr