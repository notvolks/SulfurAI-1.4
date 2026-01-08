def handle_sulfur_exception(e, call):
    import os

    if isinstance(e, NameError):

        from scripts.ai_renderer_sentences.error import SulfurError
        raise SulfurError(message=f"SULFUR SETUP EXCEPTION (run_locally): Previous variables were not defined. This is most-likely a Sulfur-Side issue. {e}")

    elif isinstance(e, TypeError):
        from scripts.ai_renderer_sentences.error import SulfurError
        raise SulfurError(
            message=f"SULFUR EXCEPTION (run_locally): input_string must be a *string*! {e}")

    elif isinstance(e, FileNotFoundError):
        file_path_cache_LocalHost = call.cache_LocalScriptHost()
        try:
            with open(file_path_cache_LocalHost, "r", encoding="utf-8", errors="ignore") as file:
                cached_local_check = file.read()

            if not cached_local_check.startswith("LL"):

                from scripts.ai_renderer_sentences.error import SulfurError
                raise SulfurError(
                    message=f"SulfurAI cache file is corrupted or invalid.")

            base_path = cached_local_check[2:].strip()
            if base_path not in cached_local_check:
                from scripts.ai_renderer_sentences.error import SulfurError
                raise SulfurError(
                    message=f"SULFUR EXCEPTION (run_locally): You did not set up SulfurAI via SulfurAI.setup_local(). "
                    "DEBUG FIX: Run SulfurAI.setup_local() once and then delete it.")

        except FileNotFoundError:
            from scripts.ai_renderer_sentences.error import SulfurError
            raise SulfurError(
                message=f"SULFUR EXCEPTION (run_locally): You did not set up SulfurAI via SulfurAI.setup_local(). "
                        "DEBUG FIX: Run SulfurAI.setup_local() once and then delete it.")

        from scripts.ai_renderer_sentences.error import SulfurError
        raise SulfurError(
            message=f"SULFUR EXCEPTION (run_locally): A required file was not found. Was it deleted? {e}")

    elif isinstance(e, IOError):
        from scripts.ai_renderer_sentences.error import SulfurError
        raise SulfurError(
            message=f"SULFUR EXCEPTION (run_locally): A required file was not found. Was it deleted? {e}")

    elif isinstance(e, ValueError):

        from scripts.ai_renderer_sentences.error import SulfurError
        raise SulfurError(
            message=f"SULFUR EXCEPTION (run_locally): Sulfur could not convert a value. Likely a Sulfur-Side issue. {e}")

    elif isinstance(e, AttributeError):

        from scripts.ai_renderer_sentences.error import SulfurError
        raise SulfurError(
            message=f"SULFUR EXCEPTION (run_locally): A call failed due to a missing attribute. Likely a Sulfur-Side issue. {e}")

    else:
        from scripts.ai_renderer_sentences.error import SulfurError
        raise SulfurError(
            message=f"{e}")
