#!/usr/bin/env python
# -*- coding: utf-8 -*-
#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os


_CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))


class Config(object):
    def __init__(self):
        self.CURRENT_DIR = _CURRENT_DIR

        self.DATA_PATH = os.path.abspath(os.path.join(_CURRENT_DIR, "data"))

        self.DOC_PATH = os.path.abspath(os.path.join(_CURRENT_DIR, "docs"))

        self.DRIVING_LOG_PATH = os.path.join(self.DATA_PATH, "driving_log.csv")

        self.IMAGE_PATH = os.path.join(self.DATA_PATH, "IMG")

        self.RESULT_IMAGE_PATH = os.path.join(self.DATA_PATH, "results")

        self.SAVED_MODELS_PATH = os.path.join(self.CURRENT_DIR, "saved_models")

        self.SAVED_MODELS = os.path.join(self.SAVED_MODELS_PATH, "model-038.h5")

        self.EPOCHS = 100

        self.BATCH_SIZE = 64

    def display(self):
        """
        Display Configuration values.
        """
        print("\nConfigurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("{:30} {}".format(a, getattr(self, a)))
                print("\n")
