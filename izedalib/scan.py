from fnmatch import fnmatch
import os


class Scan:

    @staticmethod
    def scan(root, pattern="*.csv", verbose=0):
        res = []
        for path, subdirs, files in os.walk(root):
            for name in files:
                if fnmatch(name, pattern):
                    r = os.path.join(path, name)
                    res.append(r)
                    if verbose == 1:
                        print(r)
        return res, (path, subdirs, files)
