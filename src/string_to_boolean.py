def string_to_boolean(input_str):
    """
    Convert a string representing a boolean value into an actual boolean value.
    
    Args:
    input_str (str): A string that is expected to be "true" or "false" (case insensitive).
    
    Returns:
    bool: The boolean value represented by the input string.
    """
    if not isinstance(input_str, str):
        if isinstance(input_str, bool):
            return input_str
        return False
    
    if input_str.lower() == "true":
        return True
    elif input_str.lower() == "false":
        return False
    else:
        return False