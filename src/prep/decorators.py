from functools import wraps
from src.prep.prep_globals import lock, counter


def call_counter(func):
    """
    Decorator to count the number of texts processed and print progress at intervals.

    Args:
        func (callable): The function to be decorated.

    Returns:
        callable: The wrapped function that increments the counter each time it is called.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        """
        Wrapper function to increment the counter and print progress every 5000 texts.

        Args:
            *args: Positional arguments to pass to the wrapped function.
            **kwargs: Keyword arguments to pass to the wrapped function.

        Returns:
            Any: The return value of the wrapped function.
        """
        with lock:
            counter.value += 1
            if counter.value % 5000 == 0:
                print(f"Processed {counter.value} texts")
        return func(*args, **kwargs)

    return wrapper

