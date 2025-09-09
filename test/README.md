## Install dependencies

```bash
pip install -r requirements.txt
```
## Start server

```bash
uvicorn server:app --workers=5
```

## How to run the locust stress test

1. Open a terminal and navigate to the `test` directory.
2. Run the following command to start the locust web interface:

```bash
locust -f locustfile.py --host=http://127.0.0.1:8000
```

3. Open a web browser and go to `http://localhost:8089`.
4. Enter the number of users to simulate and the spawn rate, then start the test.