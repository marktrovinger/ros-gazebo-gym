"""Meta class that allows you to create python singletons. Base don the answers in
https://stackoverflow.com/questions/6760685/creating-a-singleton-in-python?page=1&tab=active#tab-top.
"""

from bayesian_learning_control.utils.log_utils import colorize


class Singleton(type):
    """Meta class that allows you to create a singleton. This is used to prevent certain
    environment from being created multiple times.
    """

    _instances = {}
    _was_warned = False

    def __call__(cls, *args, **kwargs):
        """Make sure only one instance is created and this same instance is returned
        when somebody tries to create another instance.
        """
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        else:
            if not cls._was_warned:
                print(
                    colorize(
                        "WARNING: The task environment you choose has been implemented "
                        "as a singleton since it does not yet allow multiple active "
                        "versions. As a result, the previously created instance was "
                        "returned. Please open a pull issue/pull request if your use "
                        "case simultaneously needs various versions of the tasks "
                        "environment.",
                        color="yellow",
                        bold=True,
                    )
                )
        return cls._instances[cls]
