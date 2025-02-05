class NotInstalled:
    """This object is used to notify the user that additional dependencies need to be
    installed in order to use the string matching model.
    """

    def __init__(self, tool, dep, custom_msg=None):
        self.tool = tool
        self.dep = dep

        msg = f"In order to use {self.tool} you will need to install via;\n\n"
        if custom_msg is not None:
            msg += custom_msg
        else:
            msg += f"pip install keybert[{self.dep}]\n\n"
        self.msg = msg

    def __getattr__(self, *args, **kwargs):
        raise ModuleNotFoundError(self.msg)

    def __call__(self, *args, **kwargs):
        raise ModuleNotFoundError(self.msg)
