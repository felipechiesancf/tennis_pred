# Usage: python test_project.py
import sys
import requests

URL = "https://web-production-3021d.up.railway.app/api/predict"
PAYLOAD = {
    "p1_name": "Carlos Alcaraz",
    "p2_name": "Jannik Sinner",
    "surface": "Clay",
    "best_of_5": True,
}


def main() -> int:
    try:
        resp = requests.post(URL, json=PAYLOAD, timeout=30)
    except requests.RequestException as e:
        print(f"✗ Request failed: {e}")
        return 1

    if resp.status_code != 200:
        print(f"✗ Expected status 200, got {resp.status_code}")
        print(f"  Body: {resp.text[:500]}")
        return 1
    print("✓ API reachable")

    try:
        data = resp.json()
    except ValueError as e:
        print(f"✗ Response is not valid JSON: {e}")
        return 1

    if "p1_win_prob" not in data or "p2_win_prob" not in data:
        print(f"✗ Missing required keys in response: {data}")
        return 1

    p1 = data["p1_win_prob"]
    p2 = data["p2_win_prob"]

    if not (isinstance(p1, (int, float)) and 0 <= p1 <= 1):
        print(f"✗ p1_win_prob out of range or wrong type: {p1!r}")
        return 1
    if not (isinstance(p2, (int, float)) and 0 <= p2 <= 1):
        print(f"✗ p2_win_prob out of range or wrong type: {p2!r}")
        return 1

    print(
        f"✓ Prediction received: Alcaraz {p1*100:.1f}% vs Sinner {p2*100:.1f}%"
    )
    print("✓ All checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
