import SulfurAI,time
#SulfurAI.server.add_input_endpoint("test_endpoint",extra_debug=True)
#SulfurAI.server.add_input_endpoint("test_endpoint1",extra_debug=True)
#SulfurAI.server.add_input_endpoint("test_endpoint2",extra_debug=True)
SulfurAI.server.wait_for_output_endpoint("test_endpoint",max_timeout=1000)
print(SulfurAI.server.get_output_endpoint("test_endpoint"))