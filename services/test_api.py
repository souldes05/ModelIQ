import requests
import json
from datetime import datetime

# -----------------------------
# Configuration
# -----------------------------
API_URL = "http://127.0.0.1:8001/predict"  # Update to your API port
TIMEOUT = 10  # seconds for API request timeout
LOG_FILE = "test_api_results.log"

# -----------------------------
# Test payloads
# -----------------------------
test_payloads = [
    {
        "name": "Missing fields",
        "payload": {"data": [{"gender": "Male"}]}
    },
    {
        "name": "Unseen categorical values",
        "payload": {"data": [{"gender": "Alien", "Partner": "Maybe", "tenure": 10}]}
    },
    {
        "name": "Invalid numeric values",
        "payload": {"data": [{"tenure": "twelve", "MonthlyCharges": -50, "TotalCharges": "abc"}]}
    },
    {
        "name": "Valid input",
        "payload": {"data": [
            {
                "gender_binary": 1,
                "SeniorCitizen": 0,
                "Partner_binary": 1,
                "Dependents_binary": 0,
                "tenure": 12,
                "PhoneService_binary": 1,
                "PaperlessBilling_binary": 0,
                "MonthlyCharges": 56.7,
                "TotalCharges": 700.5
            }
        ]}
    }
]

# -----------------------------
# Helper function to log results
# -----------------------------
def log_result(test_name, status, details):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, "a") as f:
        f.write(f"{timestamp} | {test_name} | {status} | {details}\n")

# -----------------------------
# Run tests
# -----------------------------
results = []

for test in test_payloads:
    name = test["name"]
    payload = test["payload"]
    try:
        response = requests.post(API_URL, json=payload, timeout=TIMEOUT)
        response.raise_for_status()  # HTTP errors
        try:
            data = response.json()
        except json.JSONDecodeError as e:
            results.append((name, "FAIL"))
            log_result(name, "FAIL", f"Invalid JSON response: {e}")
            print(f"[FAIL] {name} → Invalid JSON response: {e}")
            continue

        # Check if 'predictions' key exists
        if "predictions" in data:
            results.append((name, "PASS"))
            log_result(name, "PASS", f"Response: {data}")
            print(f"[PASS] {name} → Response: {data}")
        else:
            results.append((name, "FAIL"))
            log_result(name, "FAIL", f"Missing 'predictions' in response: {data}")
            print(f"[FAIL] {name} → Missing 'predictions' in response: {data}")

    except requests.Timeout:
        results.append((name, "FAIL"))
        log_result(name, "FAIL", "Request timed out")
        print(f"[FAIL] {name} → Request timed out")
    except requests.ConnectionError:
        results.append((name, "FAIL"))
        log_result(name, "FAIL", "Connection error")
        print(f"[FAIL] {name} → Connection error")
    except requests.HTTPError as e:
        results.append((name, "FAIL"))
        log_result(name, "FAIL", f"HTTP error: {e}")
        print(f"[FAIL] {name} → HTTP error: {e}")
    except Exception as e:
        results.append((name, "FAIL"))
        log_result(name, "FAIL", f"Unexpected error: {e}")
        print(f"[FAIL] {name} → Unexpected error: {e}")

# -----------------------------
# Summary
# -----------------------------
print("\n" + "="*40)
print("TEST SUMMARY")
for name, status in results:
    print(f"{name}: {status}")
print("="*40)

log_result("SUMMARY", "INFO", f"Total tests: {len(results)}, Passed: {sum(1 for _, s in results if s=='PASS')}, Failed: {sum(1 for _, s in results if s=='FAIL')}")
