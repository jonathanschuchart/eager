class FileAndConsole(object):
    def __init__(self, *filenames):
        self.filenames = filenames
        self.files = [open(f, "w") if type(f) == str else f for f in self.filenames]

    def __enter__(self):
        self.files = [open(f, "w") if type(f) == str else f for f in self.filenames]

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()
        print(obj, end="")

    def flush(self):
        for f in self.files:
            f.flush()

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def close(self):
        for f in self.files:
            f.close()
