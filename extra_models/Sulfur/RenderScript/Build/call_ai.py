
def _call_ai_class(class_name,is_main):

    """
    Dynamically loads an AI class script using the provided class name.

    Args:
        class_name (str): The short identifier for the AI class. Currently only "CD" is supported.

    Returns:
        dict: Executed AI class script as a dictionary of its variables and functions.
    """
    import os, runpy
    if class_name == "CD":
        # Get the absolute path to the project root
        def _get_call_file_path():
            from extra_models.Sulfur.TrainingScript.Build import call_file_path
            return call_file_path.Call()
        call = _get_call_file_path()
        check_device_path = call.check_device_s()
        # Use the absolute path with runpy
        Check_device_s = runpy.run_path(
            check_device_path,
            init_globals={"__caller__": __name__}
        )
        return Check_device_s