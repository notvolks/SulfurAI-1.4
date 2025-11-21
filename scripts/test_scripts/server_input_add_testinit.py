import SulfurAI, os, time

srv = SulfurAI.server.get()  # create server (once)
print("Server created:", srv)

resp = srv.add_input_endpoint(input_string="test_endpoint_with_trace")
srv.wait_for_output_endpoint(input_string="test_endpoint_with_trace")
print("RESP:", resp)

time.sleep(0.5)  # give worker a moment
print("Queue size after enqueue:", srv.queue.qsize())

# poll cache for the id that was returned
id_ = resp[0]["id"] if isinstance(resp, tuple) and isinstance(resp[0], dict) else (resp.get("id") if isinstance(resp, dict) else None)
cache_folder = os.path.join(srv.cache_base, id_) if id_ else None
print("expected cache folder:", cache_folder)

# Show output file if present
if cache_folder and os.path.exists(cache_folder):
    print("Cache files:", os.listdir(cache_folder))
    out_json = os.path.join(cache_folder, "output.json")
    out_txt = os.path.join(cache_folder, "output.txt")
    if os.path.exists(out_json):
        import json
        with open(out_json, "r", encoding="utf-8") as f:
            print("output.json content:", json.load(f))
    elif os.path.exists(out_txt):
        with open(out_txt, "r", encoding="utf-8") as f:
            print("output.txt content:", f.read())
else:
    print("No cache folder found yet; worker probably hasn't written; check stdout for worker debug prints.")
