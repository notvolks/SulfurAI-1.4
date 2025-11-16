def handle_sulfur_exception(e, call):
    import os

    if isinstance(e, NameError):
        raise NameError(
            "SULFUR SETUP EXCEPTION (run_locally): Previous variables were not defined. This is most-likely a Sulfur-Side issue."
        ) from e

    elif isinstance(e, TypeError):
        raise TypeError("SULFUR EXCEPTION (run_locally): input_string must be a *string*!") from e

    elif isinstance(e, FileNotFoundError):
        file_path_cache_LocalHost = call.cache_LocalScriptHost()
        try:
            with open(file_path_cache_LocalHost, "r", encoding="utf-8", errors="ignore") as file:
                cached_local_check = file.read()

            if not cached_local_check.startswith("LL"):
                raise FileNotFoundError("SulfurAI cache file is corrupted or invalid.")

            base_path = cached_local_check[2:].strip()
            if base_path not in cached_local_check:
                raise FileNotFoundError(
                    "SULFUR EXCEPTION (run_locally): You did not set up SulfurAI via SulfurAI.setup_local(). "
                    "DEBUG FIX: Run SulfurAI.setup_local() once and then delete it."
                ) from e

        except FileNotFoundError:
            raise FileNotFoundError(
                "SULFUR EXCEPTION (run_locally): You did not set up SulfurAI via SulfurAI.setup_local(). "
                "DEBUG FIX: Run SulfurAI.setup_local() once and then delete it."
            ) from e

        raise FileNotFoundError(
            "SULFUR EXCEPTION (run_locally): A required file was not found. Was it deleted?"
        ) from e

    elif isinstance(e, IOError):
        raise IOError(
            "SULFUR EXCEPTION (run_locally): A file could not be accessed. Was it deleted?"
        ) from e

    elif isinstance(e, ValueError):
        raise ValueError(
            "SULFUR EXCEPTION (run_locally): Sulfur could not convert a value. Likely a Sulfur-Side issue."
        ) from e

    elif isinstance(e, AttributeError):
        raise AttributeError(
            "SULFUR EXCEPTION (run_locally): A call failed due to a missing attribute. Likely a Sulfur-Side issue."
        ) from e

    else:
        raise e  # Re-raise unexpected exceptions