import requests
import json

def test_chat_api():
    url = "http://127.0.0.1:8000/chat"

    payload = json.dumps({
        "query": "what is my name again?",
        "history": [
            {
                "role": "assistant",
                "content": "hi how can help you ?"
            },
            {
                "role": "user",
                "content": "My name is Kendrick"
            },
            {
                "role": "assistant",
                "content": "hi Man how can help you ?"
            }
        ]
    })
    headers = {
        'accept': 'application/json',
        'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload, stream=True)

    assert response.status_code == 200
    
    full_response = ""
    for chunk in response.iter_content(chunk_size=None):
        if chunk:
            full_response += chunk.decode('utf-8')

    assert "data:" in full_response