"""
Quick Gemini API connectivity test.
Usage: python3 test_gemini.py [API_KEY]
       If no key provided, reads from ~/.quantoricv_settings.json
"""
import sys
import json
import os
import time
import urllib.request
import urllib.error

def load_key():
    if len(sys.argv) > 1:
        return sys.argv[1].strip()
    settings = os.path.expanduser("~/.quantoricv_settings.json")
    if os.path.exists(settings):
        with open(settings) as f:
            return json.load(f).get("api_key", "")
    return ""

def check_network():
    print("1. Network connectivity...")
    for url in ["https://www.google.com", "https://www.cloudflare.com"]:
        try:
            urllib.request.urlopen(url, timeout=4)
            print(f"   ✅ Internet reachable ({url})")
            return True
        except urllib.error.HTTPError:
            print(f"   ✅ Internet reachable ({url})")
            return True
        except:
            pass
    print("   ❌ No internet (or HTTPS blocked)")
    return False

def check_google_apis():
    print("2. Google APIs reachable...")
    url = "https://generativelanguage.googleapis.com/"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "test"})
        urllib.request.urlopen(req, timeout=5)
        print(f"   ✅ generativelanguage.googleapis.com reachable")
        return True
    except urllib.error.HTTPError as e:
        # Any HTTP error means the host is reachable
        print(f"   ✅ generativelanguage.googleapis.com reachable (HTTP {e.code})")
        return True
    except Exception as e:
        print(f"   ❌ Cannot reach googleapis.com: {e}")
        print("   ⚠️  This is the typical symptom of Russian IP blocking.")
        print("   💡 Solutions: VPN, proxy, or use a server outside Russia.")
        return False

def check_api_key(key):
    print("3. API key format...")
    if not key:
        print("   ❌ No API key found")
        return False
    if not key.startswith("AIza"):
        print(f"   ⚠️  Key doesn't start with 'AIza': {key[:8]}...")
    else:
        print(f"   ✅ Key looks valid: {key[:8]}...{key[-4:]}")
    return True

def test_api_call(key):
    print("4. Test API call (simple prompt)...")
    try:
        from google import genai
        client = genai.Client(api_key=key)
        t0 = time.time()
        resp = client.models.generate_content(
            model="gemini-2.0-flash",
            contents="Reply with exactly: OK"
        )
        elapsed = time.time() - t0
        text = resp.text.strip()
        print(f"   ✅ Response in {elapsed:.1f}s: {repr(text)}")
        in_tok = getattr(resp.usage_metadata, 'prompt_token_count', '?')
        out_tok = getattr(resp.usage_metadata, 'candidates_token_count', '?')
        print(f"   Tokens: {in_tok} in / {out_tok} out")
        return True
    except Exception as e:
        err = str(e)
        print(f"   ❌ API call failed: {err}")
        if "403" in err or "API_KEY_INVALID" in err:
            print("   💡 Invalid or expired API key.")
        elif "429" in err:
            print("   💡 Rate limit hit — wait and retry.")
        elif "timeout" in err.lower() or "connection" in err.lower():
            print("   💡 Connection timeout — likely geo-blocked. Try VPN.")
        return False

def main():
    print("=" * 50)
    print("Gemini API connectivity test")
    print("=" * 50)

    key = load_key()
    check_network()
    reachable = check_google_apis()
    key_ok = check_api_key(key)

    if reachable and key_ok:
        test_api_call(key)
    else:
        print("\n4. Skipping API call (prerequisites failed)")

    print("=" * 50)

if __name__ == "__main__":
    main()
