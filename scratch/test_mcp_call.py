import subprocess
import json
import sys

def call_mcp_tool(tool_name, arguments={}):
    # Command to run the MCP server
    cmd = [
        r"tools\lightrag\.venv\Scripts\python.exe",
        "-m", "kryonix_brain_lightrag.server"
    ]
    
    # JSON-RPC request
    # Note: Modern MCP uses 'tools/call'
    request = {
        "jsonrpc": "2.0",
        "id": "1",
        "method": "tools/call",
        "params": {
            "name": tool_name,
            "arguments": arguments
        }
    }
    
    # Run server and pipe request
    process = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding='utf-8'
    )
    
    # In a real MCP scenario, there's an initialization handshake.
    # For a quick tool call test, we might need to send initialize first.
    # But let's see if our server handles a direct call (it might not without handshake).
    
    init_request = {
        "jsonrpc": "2.0",
        "id": "0",
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "test-client", "version": "1.0.0"}
        }
    }
    
    process.stdin.write(json.dumps(init_request) + "\n")
    process.stdin.write(json.dumps(request) + "\n")
    process.stdin.flush()
    
    # Read responses
    # First response should be initialize result
    # Second might be a notification
    # Third should be our tool result
    
    responses = []
    try:
        for _ in range(5): # Read up to 5 lines
            line = process.stdout.readline()
            if line:
                responses.append(json.loads(line))
                if "id" in responses[-1] and responses[-1]["id"] == "1":
                    break
    except Exception as e:
        print(f"Error reading: {e}")
    finally:
        process.terminate()
        
    return responses

if __name__ == "__main__":
    results = call_mcp_tool("rag_stats")
    print(json.dumps(results, indent=2))
