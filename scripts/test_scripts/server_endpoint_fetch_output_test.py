import SulfurAI

SulfurAI.server.clear_local_endpoint_cache()
print(SulfurAI.server.get_output_endpoint("test_endpoint"))