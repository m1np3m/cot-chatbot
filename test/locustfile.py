from locust import HttpUser, task, between

class WebsiteUser(HttpUser):
    wait_time = between(1, 5)

    @task
    def index_page(self):
        payload = {
            "query": "Tell me about coach on tap ?",
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
        }
        headers = {
            'accept': 'application/json',
            'Content-Type': 'application/json'
        }
        self.client.post("/chat", json=payload, headers=headers)