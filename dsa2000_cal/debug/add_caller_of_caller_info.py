import inspect


def get_grandparent_info():
    # Get the grandparent frame (caller of the caller)
    depth = min(2, len(inspect.stack()) - 1)
    caller_frame = inspect.stack()[depth]
    caller_file = caller_frame.filename
    caller_line = caller_frame.lineno
    caller_func = caller_frame.function

    return f"at {caller_file}:{caller_line} in {caller_func}"


def foo():
    def bar():
        print(get_grandparent_info())

    bar()
def main():
    foo()

if __name__ == '__main__':
    main()