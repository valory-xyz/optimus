import json

try:
    # Read the Gitleaks leak report
    with open("leak_report", "r") as report_file:
        report = json.load(report_file)
        fps = [leak["Fingerprint"] + "\n" for leak in report]

    # Append fingerprints to .gitleaksignore
    with open(".gitleaksignore", "a") as ignore_file:
        ignore_file.writelines(fps)

    print("✅ Successfully added detected leaks to .gitleaksignore")

except Exception as e:
    print(f"❌ Error: {e}")
