import random
import time


def process_candidate_keywords(documents, candidate_keywords):
    """Create a common format for candidate keywords."""
    if candidate_keywords is None:
        candidate_keywords = [None for _ in documents]
    elif isinstance(candidate_keywords[0][0], str) and not isinstance(candidate_keywords[0], list):
        candidate_keywords = [[keyword for keyword, _ in candidate_keywords]]
    elif isinstance(candidate_keywords[0][0], tuple):
        candidate_keywords = [[keyword for keyword, _ in keywords] for keywords in candidate_keywords]
    return candidate_keywords


def retry_with_exponential_backoff(
    func,
    initial_delay: float = 1,
    exponential_base: float = 2,
    jitter: bool = True,
    max_retries: int = 10,
    errors: tuple = None,
):
    """Retry a function with exponential backoff."""

    def wrapper(*args, **kwargs):
        # Initialize variables
        num_retries = 0
        delay = initial_delay

        # Loop until a successful response or max_retries is hit or an exception is raised
        while True:
            try:
                return func(*args, **kwargs)

            # Retry on specific errors
            except errors:
                # Increment retries
                num_retries += 1

                # Check if max retries has been reached
                if num_retries > max_retries:
                    raise Exception(f"Maximum number of retries ({max_retries}) exceeded.")

                # Increment the delay
                delay *= exponential_base * (1 + jitter * random.random())

                # Sleep for the delay
                time.sleep(delay)

            # Raise exceptions for any errors not specified
            except Exception as e:
                raise e

    return wrapper
